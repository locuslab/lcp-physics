from functools import reduce
import math

import ode
import pygame

import torch
from torch.autograd import Variable

from .utils import Indices, Params, wrap_variable, polar_to_cart, cart_to_polar

X = Indices.X
Y = Indices.Y
DIM = Params.DIM

Tensor = Params.TENSOR_TYPE


class Body(object):
    def __init__(self, pos, mass=1, restitution=Params.DEFAULT_RESTITUTION,
                 fric_coeff=Params.DEFAULT_FRIC_COEFF, eps=Params.DEFAULT_EPSILON,
                 col=(255, 0, 0), thickness=1):
        self.eps = Variable(Tensor([eps]))
        # rotation & position vector
        self.p = torch.cat([Variable(Tensor(1).zero_()), wrap_variable(pos)])
        self.rot = self.p[0:1]
        self.pos = self.p[1:]

        # linear and angular velocity vector
        lin_vel = Variable(Tensor([0, 0]))
        ang_vel = Variable(Tensor([0]))
        self.v = torch.cat([ang_vel, lin_vel])

        self.mass = wrap_variable(mass)
        self.ang_inertia = self._get_ang_inertia(self.mass)
        # M can change if object rotates, not the case for now
        self.M = Variable(Tensor(len(self.v), len(self.v)).zero_())
        s = [self.ang_inertia.size(0), self.ang_inertia.size(0)]
        self.M[:s[0], :s[1]] = self.ang_inertia
        self.M[s[0]:, s[1]:] = Variable(torch.eye(DIM).type_as(self.M.data)) * self.mass

        self.fric_coeff = wrap_variable(fric_coeff)
        self.restitution = wrap_variable(restitution)
        self.forces = []

        self.col = col
        self.thickness = thickness

        self._create_geom()

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
            s = math.sin(-self.rot.data[0] / 2)
            c = math.cos(-self.rot.data[0] / 2)
            quat = [s, 0, 0, c]  # Eq 2.3
            self.geom.setQuaternion(quat)

    def apply_forces(self, t):
        return reduce(sum, [f.force(t) for f in self.forces],
                      Variable(Tensor(len(self.v)).zero_()))

    def add_no_collision(self, other):
        self.geom.no_collision.add(other.geom)
        other.geom.no_collision.add(self.geom)

    def add_force(self, f):
        self.forces.append(f)

    def draw(self, screen):
        raise NotImplementedError


class Rect(Body):
    def __init__(self, pos, dims, mass=1, restitution=Params.DEFAULT_RESTITUTION,
                 fric_coeff=Params.DEFAULT_FRIC_COEFF, eps=Params.DEFAULT_EPSILON,
                 col=(255, 0, 0), thickness=1):
        self.dims = wrap_variable(dims)
        super().__init__(pos, mass=mass, restitution=restitution, fric_coeff=fric_coeff,
                         eps=eps, col=col, thickness=thickness)

    def _get_ang_inertia(self, mass):
        return mass * torch.sum(self.dims ** 2) / 12

    def _create_geom(self):
        self.geom = ode.GeomBox(None, torch.cat([self.dims.data + 2 * self.eps.data[0],
                                                 torch.ones(1).type_as(self.M.data)]))
        self.geom.setPosition(torch.cat([self.pos.data, Tensor(1).zero_()]))
        self.geom.no_collision = set()

    def draw(self, screen):
        # counter clockwise vertices, p1 is top right, origin center of mass
        half_dims = self.dims / 2
        r, theta = cart_to_polar(half_dims)
        p1 = polar_to_cart(r, self.rot.data[0] + theta[0])
        p2 = polar_to_cart(r, self.rot.data[0] + math.pi - theta[0])
        p3 = polar_to_cart(r, self.rot.data[0] + math.pi + theta[0])
        p4 = polar_to_cart(r, self.rot.data[0] + 2 * math.pi - theta[0])

        # points in global frame
        pts = [(p1 + self.pos).data.numpy(), (p2 + self.pos).data.numpy(),
               (p3 + self.pos).data.numpy(), (p4 + self.pos).data.numpy()]

        # draw diagonals
        l1 = pygame.draw.line(screen, (0, 0, 255), pts[0], pts[2])
        l2 = pygame.draw.line(screen, (0, 0, 255), pts[1], pts[3])
        # draw center
        c = pygame.draw.circle(screen, (0, 0, 255),
                               self.pos.data.numpy().astype(int), 1)

        # draw rectangle
        r = pygame.draw.polygon(screen, self.col, pts, self.thickness)
        return [r, l1, l2, c]


class Circle(Body):
    def __init__(self, pos, rad, mass=1, restitution=Params.DEFAULT_RESTITUTION,
                 fric_coeff=Params.DEFAULT_FRIC_COEFF, eps=Params.DEFAULT_EPSILON,
                 col=(255, 0, 0), thickness=1):
        self.rad = wrap_variable(rad)
        super().__init__(pos, mass=mass, restitution=restitution, fric_coeff=fric_coeff,
                         eps=eps, col=col, thickness=thickness)

    def _get_ang_inertia(self, mass):
        return mass * self.rad * self.rad / 2

    def _create_geom(self):
        # XXX Change to cylinder?
        self.geom = ode.GeomSphere(None, self.rad.data[0] + self.eps.data[0])
        self.geom.setPosition(torch.cat([self.pos.data,
                                         Tensor(1).zero_()]))
        self.geom.no_collision = set()

    def move(self, dt, update_geom_rotation=False):
        super().move(dt, update_geom_rotation=update_geom_rotation)

    def set_p(self, new_p, update_geom_rotation=False):
        super().set_p(new_p, update_geom_rotation=update_geom_rotation)

    def draw(self, screen):
        center = self.pos.data.numpy().astype(int)
        rad = int(self.rad.data[0])
        # draw radius to visualize orientation
        r = pygame.draw.line(screen, (0, 0, 255), center,
                             center + [math.cos(self.rot.data[0]) * rad,
                                       math.sin(self.rot.data[0]) * rad],
                             self.thickness)
        # draw circle
        c = pygame.draw.circle(screen, self.col, center,
                               rad, self.thickness)
        return [c, r]
