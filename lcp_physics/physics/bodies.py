import math

import ode
import pygame

import torch

from .utils import Indices, Defaults, get_tensor, cross_2d, rotation_matrix

X = Indices.X
Y = Indices.Y
DIM = Defaults.DIM


class Body(object):
    """Base class for bodies.
    """
    def __init__(self, pos, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff=Defaults.FRIC_COEFF, eps=Defaults.EPSILON,
                 col=(255, 0, 0), thickness=1):
        # get base tensor to define dtype, device and layout for others
        self._set_base_tensor(locals().values())

        self.eps = get_tensor(eps, base_tensor=self._base_tensor)
        # rotation & position vectors
        pos = get_tensor(pos, base_tensor=self._base_tensor)
        if pos.size(0) == 2:
            self.p = torch.cat([pos.new_zeros(1), pos])
        else:
            self.p = pos
        self.rot = self.p[0:1]
        self.pos = self.p[1:]

        # linear and angular velocity vector
        vel = get_tensor(vel, base_tensor=self._base_tensor)
        if vel.size(0) == 2:
            self.v = torch.cat([vel.new_zeros(1), vel])
        else:
            self.v = vel

        self.mass = get_tensor(mass, self._base_tensor)
        self.ang_inertia = self._get_ang_inertia(self.mass)
        # M can change if object rotates, not the case for now
        self.M = self.v.new_zeros(len(self.v), len(self.v))
        ang_sizes = [1, 1]
        self.M[:ang_sizes[0], :ang_sizes[1]] = self.ang_inertia
        self.M[ang_sizes[0]:, ang_sizes[1]:] = torch.eye(DIM).type_as(self.M) * self.mass

        self.fric_coeff = get_tensor(fric_coeff, base_tensor=self._base_tensor)
        self.restitution = get_tensor(restitution, base_tensor=self._base_tensor)
        self.forces = []

        self.col = col
        self.thickness = thickness

        self._create_geom()

    def _set_base_tensor(self, args):
        """Check if any tensor provided and if so set as base tensor to
           use as base for other tensors' dtype, device and layout.
        """
        if hasattr(self, '_base_tensor') and self._base_tensor is not None:
            return

        for arg in args:
            if isinstance(arg, torch.Tensor):
                self._base_tensor = arg
                return

        # if no tensor provided, use defaults
        self._base_tensor = get_tensor(0, base_tensor=None)
        return

    def _create_geom(self):
        raise NotImplementedError

    def _get_ang_inertia(self, mass):
        raise NotImplementedError

    def move(self, dt, update_geom_rotation=True):
        new_p = self.p + self.v * dt
        self.set_p(new_p, update_geom_rotation)

    def set_p(self, new_p, update_geom_rotation=True):
        self.p = new_p
        # Reset memory pointers
        self.rot = self.p[0:1]
        self.pos = self.p[1:]

        self.geom.setPosition([self.pos[0], self.pos[1], 0.0])
        if update_geom_rotation:
            # XXX sign correction
            s = math.sin(-self.rot.item() / 2)
            c = math.cos(-self.rot.item() / 2)
            quat = [s, 0, 0, c]  # Eq 2.3
            self.geom.setQuaternion(quat)

    def apply_forces(self, t):
        if len(self.forces) == 0:
            return self.v.new_zeros(len(self.v))
        else:
            return sum([f.force(t) for f in self.forces])

    def add_no_contact(self, other):
        self.geom.no_contact.add(other.geom)
        other.geom.no_contact.add(self.geom)

    def add_force(self, f):
        self.forces.append(f)
        f.set_body(self)

    def draw(self, screen, pixels_per_meter=1):
        raise NotImplementedError


class Circle(Body):
    def __init__(self, pos, rad, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff=Defaults.FRIC_COEFF, eps=Defaults.EPSILON,
                 col=(255, 0, 0), thickness=1):
        self._set_base_tensor(locals().values())
        self.rad = get_tensor(rad, base_tensor=self._base_tensor)
        super().__init__(pos, vel=vel, mass=mass, restitution=restitution,
                         fric_coeff=fric_coeff, eps=eps, col=col, thickness=thickness)

    def _get_ang_inertia(self, mass):
        return mass * self.rad * self.rad / 2

    def _create_geom(self):
        self.geom = ode.GeomSphere(None, self.rad.item() + self.eps.item())
        self.geom.setPosition(torch.cat([self.pos,
                                         self.pos.new_zeros(1)]))
        self.geom.no_contact = set()

    def move(self, dt, update_geom_rotation=False):
        super().move(dt, update_geom_rotation=update_geom_rotation)

    def set_p(self, new_p, update_geom_rotation=False):
        super().set_p(new_p, update_geom_rotation=update_geom_rotation)

    def draw(self, screen, pixels_per_meter=1):
        center = (self.pos.detach().numpy() * pixels_per_meter).astype(int)
        rad = int(self.rad.item() * pixels_per_meter)
        # draw radius to visualize orientation
        r = pygame.draw.line(screen, (0, 0, 255), center,
                             center + [math.cos(self.rot.item()) * rad,
                                       math.sin(self.rot.item()) * rad],
                             self.thickness)
        # draw circle
        c = pygame.draw.circle(screen, self.col, center,
                               rad, self.thickness)
        return [c, r]


class Hull(Body):
    """Body's position will not necessarily match reference point.
       Reference point is used as a world frame reference for setting the position
       of vertices, which maintain the world frame position that was passed in.
       After vertices are positioned in world frame using reference point, centroid
       of hull is calculated and the vertices' representation is adjusted to the
       centroid's frame. Object position is set to centroid.
    """
    def __init__(self, ref_point, vertices, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff=Defaults.FRIC_COEFF, eps=Defaults.EPSILON,
                 col=(255, 0, 0), thickness=1):
        self._set_base_tensor(locals().values())
        ref_point = get_tensor(ref_point, base_tensor=self._base_tensor)
        # center vertices around centroid
        verts = [get_tensor(v, base_tensor=self._base_tensor) for v in vertices]
        assert len(verts) > 2 and self._is_clockwise(verts)
        centroid = self._get_centroid(verts)
        self.verts = [v - centroid for v in verts]
        # center position at centroid
        pos = ref_point + centroid
        # store last separating edge for SAT
        self.last_sat_idx = 0
        super().__init__(pos, vel=vel, mass=mass, restitution=restitution,
                         fric_coeff=fric_coeff, eps=eps, col=col, thickness=thickness)

    def _get_ang_inertia(self, mass):
        numerator = 0
        denominator = 0
        for i in range(len(self.verts)):
            v1 = self.verts[i]
            v2 = self.verts[(i+1) % len(self.verts)]
            norm_cross = torch.norm(cross_2d(v2, v1))
            numerator = numerator + norm_cross * \
                (torch.dot(v1, v1) + torch.dot(v1, v2) + torch.dot(v2, v2))
            denominator = denominator + norm_cross
        return 1 / 6 * mass * numerator / denominator

    def _create_geom(self):
        # find vertex furthest from centroid
        max_rad = max([v.dot(v).item() for v in self.verts])
        max_rad = math.sqrt(max_rad)

        # XXX Using sphere with largest vertex ray for broadphase for now
        self.geom = ode.GeomSphere(None, max_rad + self.eps.item())
        self.geom.setPosition(torch.cat([self.pos,
                                         self.pos.new_zeros(1)]))
        self.geom.no_contact = set()

    def set_p(self, new_p, update_geom_rotation=False):
        rot = new_p[0] - self.p[0]
        if rot.item() != 0:
            self.rotate_verts(rot)
        super().set_p(new_p, update_geom_rotation=update_geom_rotation)

    def move(self, dt, update_geom_rotation=False):
        super().move(dt, update_geom_rotation=update_geom_rotation)

    def rotate_verts(self, rot):
        rot_mat = rotation_matrix(rot)
        for i in range(len(self.verts)):
            self.verts[i] = rot_mat.matmul(self.verts[i])

    @staticmethod
    def _get_centroid(verts):
        numerator = 0
        denominator = 0
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            cross = cross_2d(v2, v1)
            numerator = numerator + cross * (v1 + v2)
            denominator = denominator + cross / 2
        return 1 / 6 * numerator / denominator

    @staticmethod
    def _is_clockwise(verts):
        total = 0
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i+1) % len(verts)]
            total = total + ((v2[X] - v1[X]) * (v2[Y] + v1[Y])).item()
        return total < 0

    def draw(self, screen, draw_center=True, pixels_per_meter=1):
        # vertices in global frame
        pts = [(v + self.pos).detach().cpu().numpy() * pixels_per_meter
               for v in self.verts]

        # draw hull
        p = pygame.draw.polygon(screen, self.col, pts, self.thickness)
        # draw center
        if draw_center:
            c_pos = (self.pos.data.numpy() * pixels_per_meter).astype(int)
            c = pygame.draw.circle(screen, (0, 0, 255), c_pos, 2)
            return [p, c]
        else:
            return [p]


class Rect(Hull):
    def __init__(self, pos, dims, vel=(0, 0, 0), mass=1, restitution=Defaults.RESTITUTION,
                 fric_coeff=Defaults.FRIC_COEFF, eps=Defaults.EPSILON,
                 col=(255, 0, 0), thickness=1):
        self._set_base_tensor(locals().values())
        self.dims = get_tensor(dims, base_tensor=self._base_tensor)
        pos = get_tensor(pos, base_tensor=self._base_tensor)
        half_dims = self.dims / 2
        v0, v1 = half_dims, half_dims * half_dims.new_tensor([-1, 1])
        verts = [v0, v1, -v0, -v1]
        ref_point = pos[-2:]
        super().__init__(ref_point, verts, vel=vel, mass=mass, restitution=restitution,
                         fric_coeff=fric_coeff, eps=eps, col=col, thickness=thickness)
        if pos.size(0) == 3:
            self.set_p(pos)

    def _get_ang_inertia(self, mass):
        return mass * torch.sum(self.dims ** 2) / 12

    def _create_geom(self):
        self.geom = ode.GeomBox(None, torch.cat([self.dims + 2 * self.eps.item(),
                                                 self.dims.new_ones(1)]))
        self.geom.setPosition(torch.cat([self.pos, self.pos.new_zeros(1)]))
        self.geom.no_contact = set()

    def rotate_verts(self, rot):
        rot_mat = rotation_matrix(rot)
        self.verts[0] = rot_mat.matmul(self.verts[0])
        self.verts[1] = rot_mat.matmul(self.verts[1])
        self.verts[2] = -self.verts[0]
        self.verts[3] = -self.verts[1]

    def set_p(self, new_p, update_geom_rotation=True):
        super().set_p(new_p, update_geom_rotation=update_geom_rotation)

    def move(self, dt, update_geom_rotation=True):
        super().move(dt, update_geom_rotation=update_geom_rotation)

    def draw(self, screen, pixels_per_meter=1):
        # draw diagonals
        verts = [(v + self.pos).detach().cpu().numpy() * pixels_per_meter
                 for v in self.verts]
        l1 = pygame.draw.line(screen, (0, 0, 255), verts[0], verts[2])
        l2 = pygame.draw.line(screen, (0, 0, 255), verts[1], verts[3])

        # draw rectangle
        p = super().draw(screen, pixels_per_meter=pixels_per_meter,
                         draw_center=False)
        return [l1, l2] + p
