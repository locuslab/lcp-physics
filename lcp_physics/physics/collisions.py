import math

import ode

import torch
from torch.autograd import Variable

from .bodies import Circle
from .utils import Indices, Params, cart_to_polar, polar_to_cart


X = Indices.X
Y = Indices.Y
DIM = Params.DIM

Tensor = Params.TENSOR_TYPE


class CollisionHandler:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class OdeCollisionHandler(CollisionHandler):
    def __call__(self, args, geom1, geom2):
        if geom1 in geom2.no_collision:
            return
        world = args[0]

        contacts = ode.collide(geom1, geom2)
        for c in contacts:
            point, normal, penetration, geom1, geom2 = c.getContactGeomParams()
            # XXX Simple disambiguation of 3D repetition of contacts
            if point[2] > 0:
                continue
            normal = Variable(Tensor(normal[:DIM]))
            point = Variable(Tensor(point))
            penetration = Variable(Tensor([penetration]))
            penetration -= 2 * world.eps
            if penetration.data[0] < -2 * world.eps:
                return
            p1 = point - Variable(Tensor(geom1.getPosition()))
            p2 = point - Variable(Tensor(geom2.getPosition()))
            world.collisions.append(((normal, p1[:DIM], p2[:DIM], penetration),
                                    geom1.body, geom2.body))
            world.collisions_debug = world.collisions  # XXX


class DiffCollisionHandler(CollisionHandler):
    def __init__(self):
        self.debug_callback = OdeCollisionHandler()

    def __call__(self, args, geom1, geom2):
        # XXX
        # self.debug_callback(args, geom1, geom2)

        if geom1 in geom2.no_collision:
            return
        world = args[0]

        b1 = world.bodies[geom1.body]
        b2 = world.bodies[geom2.body]
        is_circle_g1 = isinstance(b1, Circle)
        is_circle_g2 = isinstance(b2, Circle)
        if is_circle_g1 and is_circle_g2:
            r = b1.rad + b2.rad
            normal = b1.pos - b2.pos
            dist = normal.norm()
            penetration = r - dist
            if penetration.data[0] < -world.eps:
                return
            normal = normal / dist
            p1 = -normal * b1.rad
            p2 = normal * b2.rad
            pts = [(normal, p1, p2, penetration)]
        elif is_circle_g1 or is_circle_g2:
            # one rectangle and one circle
            if is_circle_g2:
                # set circle to b1
                b1, b2 = b2, b1
            half_dims = b2.dims / 2
            # four corners (counterclockwise, c1 = top left)
            c1 = -half_dims
            c3 = half_dims
            c2 = torch.cat([c1[X], c3[Y]])
            c4 = torch.cat([c3[X], c1[Y]])
            # positions in rect frame
            b1_pos = b1.pos - b2.pos
            # TODO Optimize rotation
            r, theta = cart_to_polar(b1_pos)
            b1_pos = polar_to_cart(r, theta - b2.rot)
            # TODO case where circle center is inside rect
            if c1.data[X] <= b1_pos.data[X] <= c3.data[X]:
                # top or bottom face contact
                if b1_pos.data[Y] < 0:
                    # center-above
                    normal = c4 - c3
                    normal = normal / normal.norm()
                    p2 = torch.cat([b1_pos[X], c1[Y]])
                else:
                    # center-below
                    normal = c3 - c4
                    normal = normal / normal.norm()
                    p2 = torch.cat([b1_pos[X], c3[Y]])
                penetration = b1.rad + half_dims[Y] - torch.abs(b1_pos[Y])
            elif c1.data[Y] <= b1_pos.data[Y] <= c3.data[Y]:
                # left or right face contact
                if b1_pos.data[X] < 0:
                    # center-left
                    normal = c1 - c4
                    normal = normal / normal.norm()
                    p2 = torch.cat([c1[X], b1_pos[Y]])
                else:
                    # center-right
                    normal = c4 - c1
                    normal = normal / normal.norm()
                    p2 = torch.cat([c3[X], b1_pos[Y]])
                penetration = b1.rad + half_dims[X] - torch.abs(b1_pos[X])
            else:
                # corner contact
                if b1_pos.data[X] < c1.data[X] and b1_pos.data[Y] < c1.data[Y]:
                    # top-left corner
                    normal = b1_pos - c1
                    normal = normal / normal.norm()
                    p2 = c1
                elif b1_pos.data[X] > c3.data[X] and b1_pos.data[Y] > c3.data[Y]:
                    # bottom-right corner
                    normal = b1_pos - c3
                    normal = normal / normal.norm()
                    p2 = c3
                elif b1_pos.data[X] < c1.data[X] and b1_pos.data[Y] > c3.data[Y]:
                    # bottom-left corner
                    normal = b1_pos - c2
                    normal = normal / normal.norm()
                    p2 = c2
                else:
                    # top-right corner
                    normal = b1_pos - c4
                    normal = normal / normal.norm()
                    p2 = c4
                penetration = b1.rad - (b1_pos - p2).norm()  # XXX
            if penetration.data[0] <= -world.eps:
                return
            # TODO Optimize rotations
            r, theta = cart_to_polar(p2)
            p2 = polar_to_cart(r, theta + b2.rot)
            r, theta = cart_to_polar(normal)
            normal = polar_to_cart(r, theta + b2.rot)

            p1 = -normal * b1.rad # TODO is this right?
            if is_circle_g2:
                # flip back values for circle as g2
                normal = -normal
                p1, p2 = p2, p1
            # penetration = Variable(Tensor([-1]))
            pts = [(normal, p1, p2, penetration)]
        else:
            # both are rectangles
            if b1.rot.data[0] % (math.pi / 2) == 0 and \
               b2.rot.data[0] % (math.pi / 2) == 0:
                # both rectangles are axis aligned
                b1_half_dims = b1.dims / 2
                b2_half_dims = b2.dims / 2
                b1_br = b1.pos + b1_half_dims
                b1_tl = b1.pos - b1_half_dims
                b2_br = b2.pos + b2_half_dims
                b2_tl = b2.pos - b2_half_dims
                delta_pos = b1.pos - b2.pos
                overlap = b1_half_dims + b2_half_dims - torch.abs(delta_pos)
                if overlap.data[X] > overlap.data[Y]:
                    penetration = overlap[Y]
                    if penetration.data[0] < -world.eps:
                        return
                    lx = torch.max(b1_tl[X], b2_tl[X])
                    rx = torch.min(b1_br[X], b2_br[X])
                    normal = b1_tl - torch.cat([b1_tl[X], b1_br[Y]])
                    if delta_pos.data[Y] <= 0:
                        # above
                        pts_1 = [torch.cat([lx, b1_br[Y]]), torch.cat(([rx, b1_br[Y]]))]
                        pts_2 = [torch.cat([lx, b2_tl[Y]]), torch.cat(([rx, b2_tl[Y]]))]
                    else:
                        # below
                        pts_1 = [torch.cat([lx, b1_tl[Y]]), torch.cat(([rx, b1_tl[Y]]))]
                        pts_2 = [torch.cat([lx, b2_br[Y]]), torch.cat(([rx, b2_br[Y]]))]
                        normal *= -1
                    normal = normal / normal.norm()
                elif overlap.data[X] < overlap.data[Y]:
                    penetration = overlap[X]
                    if penetration.data[0] < -world.eps:
                        return
                    ty = torch.max(b1_tl[Y], b2_tl[Y])
                    by = torch.min(b1_br[Y], b2_br[Y])
                    normal = b1_tl - torch.cat([b1_br[X], b1_tl[Y]])
                    if delta_pos.data[X] <= 0:
                        # left
                        pts_1 = [torch.cat([b1_tl[X]], ty), torch.cat(([b1_tl[X], by]))]
                        pts_2 = [torch.cat([b2_br[X], ty]), torch.cat(([b2_br[X], by]))]
                    else:
                        # right
                        pts_1 = [torch.cat([b1_br[X]], ty), torch.cat(([b1_br[X], by]))]
                        pts_2 = [torch.cat([b2_tl[X], ty]), torch.cat(([b2_tl[X], by]))]
                        normal *= -1
                    normal = normal / normal.norm()
                else:
                    assert False  # TODO normal that points 45 degrees from corner
                pts = [(normal, pt1 - b1.pos, pt2 - b1.pos, penetration)
                       for pt1, pt2 in zip(pts_1, pts_2)]
            else:
                # rectangles not axis aligned
                # TODO Simplify case where both are aligned with each other but not with world?
                delta_pos = b1.pos - b2.pos
                # Calculate separations for axis in b1's frame
                c, s = torch.cos(-b1.rot), torch.sin(-b1.rot)
                rot_mat_1 = torch.cat([torch.cat([c, -s]).unsqueeze(0),
                                       torch.cat([s, c]).unsqueeze(0)], 0)
                rot_delta_pos = torch.matmul(rot_mat_1, delta_pos)
                rot_2 = b2.rot - b1.rot
                c, s = torch.cos(rot_2), torch.sin(rot_2)
                b2_rot_mat = torch.cat([torch.cat([c, -s]).unsqueeze(0),
                                        torch.cat([s, c]).unsqueeze(0)], 0)
                rot_half_x_2 = torch.matmul(b2_rot_mat, torch.cat([b2.dims[X] / 2, Variable(Tensor(1).zero_())]))
                rot_half_y_2 = torch.matmul(b2_rot_mat, torch.cat([Variable(Tensor(1).zero_()), b2.dims[Y] / 2]))
                b2_extent = torch.cat([torch.abs(rot_half_x_2[X]) + torch.abs(rot_half_y_2[X]),
                                       torch.abs(rot_half_x_2[Y]) + torch.abs(rot_half_y_2[Y])])
                b1_extent = b1.dims / 2
                overlap_1 = b1_extent + b2_extent - torch.abs(rot_delta_pos)

                # Calculate separations for axis in b2's frame
                c, s = torch.cos(-b2.rot), torch.sin(-b2.rot)
                rot_mat_2 = torch.cat([torch.cat([c, -s]).unsqueeze(0),
                                       torch.cat([s, c]).unsqueeze(0)], 0)
                rot_delta_pos = torch.matmul(rot_mat_2, delta_pos)
                rot_1 = b1.rot - b2.rot
                c, s = torch.cos(rot_1), torch.sin(rot_1)
                b1_rot_mat = torch.cat([torch.cat([c, -s]).unsqueeze(0),
                                        torch.cat([s, c]).unsqueeze(0)], 0)
                rot_half_x_1 = torch.matmul(b1_rot_mat, torch.cat([b1.dims[X] / 2, Variable(Tensor(1).zero_())]))
                rot_half_y_1 = torch.matmul(b1_rot_mat, torch.cat([Variable(Tensor(1).zero_()), b1.dims[Y] / 2]))
                b1_extent = torch.cat([torch.abs(rot_half_x_1[X]) + torch.abs(rot_half_y_1[X]),
                                       torch.abs(rot_half_x_1[Y]) + torch.abs(rot_half_y_1[Y])])
                b2_extent = b2.dims / 2
                overlap_2 = b1_extent + b2_extent - torch.abs(rot_delta_pos)

                min_overlap_1 = torch.min(overlap_1)
                min_overlap_2 = torch.min(overlap_2)
                penetration = torch.min(min_overlap_1, min_overlap_2)
                if penetration.data[0] < -world.eps:
                    return
                if min_overlap_1.data[0] <= min_overlap_2.data[0]:
                    # Minimal penetration is in b1 frame
                    rot_b1_pos = torch.matmul(rot_mat_1, b1.pos)
                    rot_b2_pos = torch.matmul(rot_mat_1, b2.pos)
                    b2_half_dims = b2.dims / 2
                    c1 = -b2_half_dims
                    c3 = b2_half_dims
                    c2 = torch.cat([c1[X], c3[Y]])
                    c4 = torch.cat([c3[X], c1[Y]])
                    cs = [c1, c2, c3, c4]
                    rot_cs = [rot_b2_pos + torch.matmul(b2_rot_mat, c) for c in cs]
                    if overlap_1.data[X] < overlap_1.data[Y]:
                        # Minimal penetration is in X axis
                        normal = torch.cat([-b1.dims[X], b1.dims[Y]]) - b1.dims
                        normal = torch.matmul(rot_mat_1.t(), normal)
                        normal = normal / normal.norm()
                        if rot_b2_pos.data[X] < rot_b1_pos.data[X]:
                            # b2 is to the left of b1
                            normal = -normal
                            closest = sec_closest = None
                            for i in range(len(rot_cs)):
                                if closest is None or rot_cs[i].data[X] > rot_cs[closest].data[X]:
                                    sec_closest = closest
                                    closest = i
                                elif sec_closest is None or rot_cs[i].data[X] > rot_cs[sec_closest].data[X]:
                                    sec_closest = i
                            pen_diff = rot_cs[closest] - rot_cs[sec_closest]
                            if pen_diff.data[X] > world.par_eps + max(penetration.data[0], 0):
                                sec_closest = None
                            if rot_cs[closest].data[Y] > rot_b1_pos.data[Y] + b1.dims.data[Y] / 2:
                                pt1 = torch.cat([-b1.dims[X] / 2, b1.dims[Y] / 2])
                                pt2 = torch.cat([rot_cs[closest][X], (rot_b1_pos + pt1)[Y]])
                                pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            elif rot_cs[closest].data[Y] < rot_b1_pos.data[Y] - b1.dims.data[Y] / 2:
                                pt1 = torch.cat([-b1.dims[X] / 2, -b1.dims[Y] / 2])
                                pt2 = torch.cat([rot_cs[closest][X], (rot_b1_pos + pt1)[Y]])
                                pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            else:
                                pt2 = torch.matmul(rot_mat_2.t(), cs[closest])
                                pt1 = torch.cat([-b1.dims[X] / 2, rot_cs[closest][Y] - rot_b1_pos[Y]])
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            pts = [(normal, pt1, pt2, penetration)]
                            if sec_closest is not None:
                                if rot_cs[sec_closest].data[Y] > rot_b1_pos.data[Y] + b1.dims.data[Y] / 2:
                                    pt1 = torch.cat([-b1.dims[X] / 2, b1.dims[Y] / 2])
                                    pt2 = torch.cat([rot_cs[sec_closest][X], (rot_b1_pos + pt1)[Y]])
                                    pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                elif rot_cs[sec_closest].data[Y] < rot_b1_pos.data[Y] - b1.dims.data[Y] / 2:
                                    pt1 = torch.cat([-b1.dims[X] / 2, -b1.dims[Y] / 2])
                                    pt2 = torch.cat([rot_cs[sec_closest][X], (rot_b1_pos + pt1)[Y]])
                                    pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                else:
                                    pt2 = torch.matmul(rot_mat_2.t(), cs[sec_closest])
                                    pt1 = torch.cat([-b1.dims[X] / 2, rot_cs[sec_closest][Y] - rot_b1_pos[Y]])
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                pts.append((normal, pt1, pt2, penetration - pen_diff[X]))
                        else:
                            closest = sec_closest = None
                            for i in range(len(rot_cs)):
                                if closest is None or rot_cs[i].data[X] < rot_cs[closest].data[X]:
                                    sec_closest = closest
                                    closest = i
                                elif sec_closest is None or rot_cs[i].data[X] < rot_cs[sec_closest].data[X]:
                                    sec_closest = i
                            pen_diff = rot_cs[sec_closest] - rot_cs[closest]
                            if pen_diff.data[X] > world.par_eps + max(penetration.data[0], 0):
                                sec_closest = None
                            if rot_cs[closest].data[Y] > rot_b1_pos.data[Y] + b1.dims.data[Y] / 2:
                                pt1 = torch.cat([b1.dims[X] / 2, b1.dims[Y] / 2])
                                pt2 = torch.cat([rot_cs[closest][X], (rot_b1_pos + pt1)[Y]])
                                pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            elif rot_cs[closest].data[Y] < rot_b1_pos.data[Y] - b1.dims.data[Y] / 2:
                                pt1 = torch.cat([b1.dims[X] / 2, -b1.dims[Y] / 2])
                                pt2 = torch.cat([rot_cs[closest][X], (rot_b1_pos + pt1)[Y]])
                                pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            else:
                                pt2 = torch.matmul(rot_mat_2.t(), cs[closest])
                                pt1 = torch.cat([b1.dims[X] / 2, rot_cs[closest][Y] - rot_b1_pos[Y]])
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            pts = [(normal, pt1, pt2, penetration)]
                            if sec_closest is not None:
                                if rot_cs[sec_closest].data[Y] > rot_b1_pos.data[Y] + b1.dims.data[Y] / 2:
                                    pt1 = torch.cat([b1.dims[X] / 2, b1.dims[Y] / 2])
                                    pt2 = torch.cat([rot_cs[sec_closest][X], (rot_b1_pos + pt1)[Y]])
                                    pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                elif rot_cs[sec_closest].data[Y] < rot_b1_pos.data[Y] - b1.dims.data[Y] / 2:
                                    pt1 = torch.cat([b1.dims[X] / 2, -b1.dims[Y] / 2])
                                    pt2 = torch.cat([rot_cs[sec_closest][X], (rot_b1_pos + pt1)[Y]])
                                    pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                else:
                                    pt2 = torch.matmul(rot_mat_2.t(), cs[sec_closest])
                                    pt1 = torch.cat([b1.dims[X] / 2, rot_cs[sec_closest][Y] - rot_b1_pos[Y]])
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                pts.append((normal, pt1, pt2, penetration - pen_diff[X]))
                    elif overlap_1.data[X] > overlap_1.data[Y]:
                        # Minimal penetration is in Y axis
                        normal = torch.cat([b1.dims[X], -b1.dims[Y]]) - b1.dims
                        normal = torch.matmul(rot_mat_1.t(), normal)
                        normal = normal / normal.norm()
                        if rot_b2_pos.data[Y] < rot_b1_pos.data[Y]:
                            # b2 is above b1
                            normal = -normal
                            closest = sec_closest = None
                            for i in range(len(rot_cs)):
                                if closest is None or rot_cs[i].data[Y] > rot_cs[closest].data[Y]:
                                    sec_closest = closest
                                    closest = i
                                elif sec_closest is None or rot_cs[i].data[Y] > rot_cs[sec_closest].data[Y]:
                                    sec_closest = i
                            pen_diff = rot_cs[closest] - rot_cs[sec_closest]
                            if pen_diff.data[Y] > world.par_eps + max(penetration.data[0], 0):
                                sec_closest = None
                            if rot_cs[closest].data[X] > rot_b1_pos.data[X] + b1.dims.data[X] / 2:
                                pt1 = torch.cat([b1.dims[X] / 2, -b1.dims[Y] / 2])
                                pt2 = torch.cat([(rot_b1_pos + pt1)[X], rot_cs[closest][Y]])
                                pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            elif rot_cs[closest].data[X] < rot_b1_pos.data[X] - b1.dims.data[X] / 2:
                                pt1 = torch.cat([-b1.dims[X] / 2, -b1.dims[Y] / 2])
                                pt2 = torch.cat([(rot_b1_pos + pt1)[X], rot_cs[closest][Y]])
                                pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            else:
                                pt2 = torch.matmul(rot_mat_2.t(), cs[closest])
                                pt1 = torch.cat([rot_cs[closest][X] - rot_b1_pos[X], -b1.dims[Y] / 2])
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            pts = [(normal, pt1, pt2, penetration)]
                            if sec_closest is not None:
                                if rot_cs[sec_closest].data[X] > rot_b1_pos.data[X] + b1.dims.data[X] / 2:
                                    pt1 = torch.cat([b1.dims[X] / 2, -b1.dims[Y] / 2])
                                    pt2 = torch.cat([(rot_b1_pos + pt1)[X], rot_cs[sec_closest][Y]])
                                    pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                elif rot_cs[sec_closest].data[X] < rot_b1_pos.data[X] - b1.dims.data[X] / 2:
                                    pt1 = torch.cat([-b1.dims[X] / 2, -b1.dims[Y] / 2])
                                    pt2 = torch.cat([(rot_b1_pos + pt1)[X], rot_cs[sec_closest][Y]])
                                    pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                else:
                                    pt2 = torch.matmul(rot_mat_2.t(), cs[sec_closest])
                                    pt1 = torch.cat([rot_cs[sec_closest][X] - rot_b1_pos[X], -b1.dims[Y] / 2])
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                pts.append((normal, pt1, pt2, penetration - pen_diff[Y]))
                        else:
                            closest = sec_closest = None
                            for i in range(len(rot_cs)):
                                if closest is None or rot_cs[i].data[Y] < rot_cs[closest].data[Y]:
                                    sec_closest = closest
                                    closest = i
                                elif sec_closest is None or rot_cs[i].data[Y] < rot_cs[sec_closest].data[Y]:
                                    sec_closest = i
                            pen_diff = rot_cs[sec_closest] - rot_cs[closest]
                            if pen_diff.data[Y] > world.par_eps + max(penetration.data[0], 0):
                                sec_closest = None
                            if rot_cs[closest].data[X] > rot_b1_pos.data[X] + b1.dims.data[X] / 2:
                                pt1 = torch.cat([b1.dims[X] / 2, b1.dims[Y] / 2])
                                pt2 = torch.cat([(rot_b1_pos + pt1)[X], rot_cs[closest][Y]])
                                pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            elif rot_cs[closest].data[X] < rot_b1_pos.data[X] - b1.dims.data[X] / 2:
                                pt1 = torch.cat([-b1.dims[X] / 2, b1.dims[Y] / 2])
                                pt2 = torch.cat([(rot_b1_pos + pt1)[X], rot_cs[closest][Y]])
                                pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            else:
                                pt2 = torch.matmul(rot_mat_2.t(), cs[closest])
                                pt1 = torch.cat([rot_cs[closest][X] - rot_b1_pos[X], b1.dims[Y] / 2])
                                pt1 = torch.matmul(rot_mat_1.t(), pt1)
                            pts = [(normal, pt1, pt2, penetration)]
                            if sec_closest is not None:
                                if rot_cs[sec_closest].data[X] > rot_b1_pos.data[X] + b1.dims.data[X] / 2:
                                    pt1 = torch.cat([b1.dims[X] / 2, b1.dims[Y] / 2])
                                    pt2 = torch.cat([(rot_b1_pos + pt1)[X], rot_cs[sec_closest][Y]])
                                    pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                elif rot_cs[sec_closest].data[X] < rot_b1_pos.data[X] - b1.dims.data[X] / 2:
                                    pt1 = torch.cat([-b1.dims[X] / 2, b1.dims[Y] / 2])
                                    pt2 = torch.cat([(rot_b1_pos + pt1)[X], rot_cs[sec_closest][Y]])
                                    pt2 = torch.matmul(rot_mat_1.t(), pt2) - b2.pos
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                else:
                                    pt2 = torch.matmul(rot_mat_2.t(), cs[sec_closest])
                                    pt1 = torch.cat([rot_cs[sec_closest][X] - rot_b1_pos[X], b1.dims[Y] / 2])
                                    pt1 = torch.matmul(rot_mat_1.t(), pt1)
                                pts.append((normal, pt1, pt2, penetration - pen_diff[Y]))
                    else:
                        assert False    # TODO case where == ?
                else:
                    # Minimal penetration is in b2 frame
                    rot_b1_pos = torch.matmul(rot_mat_2, b1.pos)
                    rot_b2_pos = torch.matmul(rot_mat_2, b2.pos)
                    b1_half_dims = b1.dims / 2
                    c1 = -b1_half_dims
                    c3 = b1_half_dims
                    c2 = torch.cat([c1[X], c3[Y]])
                    c4 = torch.cat([c3[X], c1[Y]])
                    cs = [c1, c2, c3, c4]
                    rot_cs = [rot_b1_pos + torch.matmul(b1_rot_mat, c) for c in cs]
                    if overlap_2.data[X] < overlap_2.data[Y]:
                        # Minimal penetration is in X axis
                        normal = torch.cat([-b2.dims[X], b2.dims[Y]]) - b2.dims
                        normal = torch.matmul(rot_mat_2.t(), normal)
                        normal = normal / normal.norm()
                        if rot_b1_pos.data[X] < rot_b2_pos.data[X]:
                            # b1 is to the left of b2
                            closest = sec_closest = None
                            for i in range(len(rot_cs)):
                                if closest is None or rot_cs[i].data[X] > rot_cs[closest].data[X]:
                                    sec_closest = closest
                                    closest = i
                                elif sec_closest is None or rot_cs[i].data[X] > rot_cs[sec_closest].data[X]:
                                    sec_closest = i
                            pen_diff = rot_cs[closest] - rot_cs[sec_closest]
                            if pen_diff.data[X] > world.par_eps + max(penetration.data[0], 0):
                                sec_closest = None
                            if rot_cs[closest].data[Y] > rot_b2_pos.data[Y] + b2.dims.data[Y] / 2:
                                pt2 = torch.cat([-b2.dims[X] / 2, b2.dims[Y] / 2])
                                pt1 = torch.cat([rot_cs[closest][X], (rot_b2_pos + pt2)[Y]])
                                pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            elif rot_cs[closest].data[Y] < rot_b2_pos.data[Y] - b2.dims.data[Y] / 2:
                                pt2 = torch.cat([-b2.dims[X] / 2, -b2.dims[Y] / 2])
                                pt1 = torch.cat([rot_cs[closest][X], (rot_b2_pos + pt2)[Y]])
                                pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            else:
                                pt1 = torch.matmul(rot_mat_1.t(), cs[closest])
                                pt2 = torch.cat([-b2.dims[X] / 2, rot_cs[closest][Y] - rot_b2_pos[Y]])
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            pts = [(normal, pt1, pt2, penetration)]
                            if sec_closest is not None:
                                if rot_cs[sec_closest].data[Y] > rot_b2_pos.data[Y] + b2.dims.data[Y] / 2:
                                    pt2 = torch.cat([-b2.dims[X] / 2, b2.dims[Y] / 2])
                                    pt1 = torch.cat([rot_cs[sec_closest][X], (rot_b2_pos + pt2)[Y]])
                                    pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                elif rot_cs[sec_closest].data[Y] < rot_b2_pos.data[Y] - b2.dims.data[Y] / 2:
                                    pt2 = torch.cat([-b2.dims[X] / 2, -b2.dims[Y] / 2])
                                    pt1 = torch.cat([rot_cs[sec_closest][X], (rot_b2_pos + pt2)[Y]])
                                    pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                else:
                                    pt1 = torch.matmul(rot_mat_1.t(), cs[sec_closest])
                                    pt2 = torch.cat([-b2.dims[X] / 2, rot_cs[sec_closest][Y] - rot_b2_pos[Y]])
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                pts.append((normal, pt1, pt2, penetration - pen_diff[X]))
                        else:
                            normal = -normal
                            closest = sec_closest = None
                            for i in range(len(rot_cs)):
                                if closest is None or rot_cs[i].data[X] < rot_cs[closest].data[X]:
                                    sec_closest = closest
                                    closest = i
                                elif sec_closest is None or rot_cs[i].data[X] < rot_cs[sec_closest].data[X]:
                                    sec_closest = i
                            pen_diff = rot_cs[sec_closest] - rot_cs[closest]
                            if pen_diff.data[X] > world.par_eps + max(penetration.data[0], 0):
                                sec_closest = None
                            if rot_cs[closest].data[Y] > rot_b2_pos.data[Y] + b2.dims.data[Y] / 2:
                                pt2 = torch.cat([b2.dims[X] / 2, b2.dims[Y] / 2])
                                pt1 = torch.cat([rot_cs[closest][X], (rot_b2_pos + pt2)[Y]])
                                pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            elif rot_cs[closest].data[Y] < rot_b2_pos.data[Y] - b2.dims.data[Y] / 2:
                                pt2 = torch.cat([b2.dims[X] / 2, -b2.dims[Y] / 2])
                                pt1 = torch.cat([rot_cs[closest][X], (rot_b2_pos + pt2)[Y]])
                                pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            else:
                                pt1 = torch.matmul(rot_mat_1.t(), cs[closest])
                                pt2 = torch.cat([b2.dims[X] / 2, rot_cs[closest][Y] - rot_b2_pos[Y]])
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            pts = [(normal, pt1, pt2, penetration)]
                            if sec_closest is not None:
                                if rot_cs[sec_closest].data[Y] > rot_b2_pos.data[Y] + b2.dims.data[Y] / 2:
                                    pt2 = torch.cat([b2.dims[X] / 2, b2.dims[Y] / 2])
                                    pt1 = torch.cat([rot_cs[sec_closest][X], (rot_b2_pos + pt2)[Y]])
                                    pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                elif rot_cs[sec_closest].data[Y] < rot_b2_pos.data[Y] - b2.dims.data[Y] / 2:
                                    pt2 = torch.cat([b2.dims[X] / 2, -b2.dims[Y] / 2])
                                    pt1 = torch.cat([rot_cs[sec_closest][X], (rot_b2_pos + pt2)[Y]])
                                    pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                else:
                                    pt1 = torch.matmul(rot_mat_1.t(), cs[sec_closest])
                                    pt2 = torch.cat([b2.dims[X] / 2, rot_cs[sec_closest][Y] - rot_b2_pos[Y]])
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                pts.append((normal, pt1, pt2, penetration - pen_diff[X]))
                    elif overlap_2.data[X] > overlap_2.data[Y]:
                        # Minimal penetration is in Y axis
                        normal = torch.cat([b2.dims[X], -b2.dims[Y]]) - b2.dims
                        normal = torch.matmul(rot_mat_2.t(), normal)
                        normal = normal / normal.norm()
                        if rot_b1_pos.data[Y] < rot_b2_pos.data[Y]:
                            # b1 is above b2
                            closest = sec_closest = None
                            for i in range(len(rot_cs)):
                                if closest is None or rot_cs[i].data[Y] > rot_cs[closest].data[Y]:
                                    sec_closest = closest
                                    closest = i
                                elif sec_closest is None or rot_cs[i].data[Y] > rot_cs[sec_closest].data[Y]:
                                    sec_closest = i
                            pen_diff = rot_cs[closest] - rot_cs[sec_closest]
                            if pen_diff.data[Y] > world.par_eps + max(penetration.data[0], 0):
                                sec_closest = None
                            if rot_cs[closest].data[X] > rot_b2_pos.data[X] + b2.dims.data[X] / 2:
                                pt2 = torch.cat([b2.dims[X] / 2, -b2.dims[Y] / 2])
                                pt1 = torch.cat([(rot_b2_pos + pt2)[X], rot_cs[closest][Y]])
                                pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            elif rot_cs[closest].data[X] < rot_b2_pos.data[X] - b2.dims.data[X] / 2:
                                pt2 = torch.cat([-b2.dims[X] / 2, -b2.dims[Y] / 2])
                                pt1 = torch.cat([(rot_b2_pos + pt2)[X], rot_cs[closest][Y]])
                                pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            else:
                                pt1 = torch.matmul(rot_mat_1.t(), cs[closest])
                                pt2 = torch.cat([rot_cs[closest][X] - rot_b2_pos[X], -b2.dims[Y] / 2])
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            pts = [(normal, pt1, pt2, penetration)]
                            if sec_closest is not None:
                                if rot_cs[sec_closest].data[X] > rot_b2_pos.data[X] + b2.dims.data[X] / 2:
                                    pt2 = torch.cat([b2.dims[X] / 2, -b2.dims[Y] / 2])
                                    pt1 = torch.cat([(rot_b2_pos + pt2)[X], rot_cs[sec_closest][Y]])
                                    pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                elif rot_cs[sec_closest].data[X] < rot_b2_pos.data[X] - b2.dims.data[X] / 2:
                                    pt2 = torch.cat([-b2.dims[X] / 2, -b2.dims[Y] / 2])
                                    pt1 = torch.cat([(rot_b2_pos + pt2)[X], rot_cs[sec_closest][Y]])
                                    pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                else:
                                    pt1 = torch.matmul(rot_mat_1.t(), cs[sec_closest])
                                    pt2 = torch.cat([rot_cs[sec_closest][X] - rot_b2_pos[X], -b2.dims[Y] / 2])
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                pts.append((normal, pt1, pt2, penetration - pen_diff[Y]))
                        else:
                            normal = -normal
                            closest = sec_closest = None
                            for i in range(len(rot_cs)):
                                if closest is None or rot_cs[i].data[Y] < rot_cs[closest].data[Y]:
                                    sec_closest = closest
                                    closest = i
                                elif sec_closest is None or rot_cs[i].data[Y] < rot_cs[sec_closest].data[Y]:
                                    sec_closest = i
                            pen_diff = rot_cs[sec_closest] - rot_cs[closest]
                            if pen_diff.data[Y] > world.par_eps + max(penetration.data[0], 0):
                                sec_closest = None
                            if rot_cs[closest].data[X] > rot_b2_pos.data[X] + b2.dims.data[X] / 2:
                                pt2 = torch.cat([b2.dims[X] / 2, b2.dims[Y] / 2])
                                pt1 = torch.cat([(rot_b2_pos + pt2)[X], rot_cs[closest][Y]])
                                pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            elif rot_cs[closest].data[X] < rot_b2_pos.data[X] - b2.dims.data[X] / 2:
                                pt2 = torch.cat([-b2.dims[X] / 2, b2.dims[Y] / 2])
                                pt1 = torch.cat([(rot_b2_pos + pt2)[X], rot_cs[closest][Y]])
                                pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            else:
                                pt1 = torch.matmul(rot_mat_1.t(), cs[closest])
                                pt2 = torch.cat([rot_cs[closest][X] - rot_b2_pos[X], b2.dims[Y] / 2])
                                pt2 = torch.matmul(rot_mat_2.t(), pt2)
                            pts = [(normal, pt1, pt2, penetration)]
                            if sec_closest is not None:
                                if rot_cs[sec_closest].data[X] > rot_b2_pos.data[X] + b2.dims.data[X] / 2:
                                    pt2 = torch.cat([b2.dims[X] / 2, b2.dims[Y] / 2])
                                    pt1 = torch.cat([(rot_b2_pos + pt2)[X], rot_cs[sec_closest][Y]])
                                    pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                elif rot_cs[sec_closest].data[X] < rot_b2_pos.data[X] - b2.dims.data[X] / 2:
                                    pt2 = torch.cat([-b2.dims[X] / 2, b2.dims[Y] / 2])
                                    pt1 = torch.cat([(rot_b2_pos + pt2)[X], rot_cs[sec_closest][Y]])
                                    pt1 = torch.matmul(rot_mat_2.t(), pt1) - b1.pos
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                else:
                                    pt1 = torch.matmul(rot_mat_1.t(), cs[sec_closest])
                                    pt2 = torch.cat([rot_cs[sec_closest][X] - rot_b2_pos[X], b2.dims[Y] / 2])
                                    pt2 = torch.matmul(rot_mat_2.t(), pt2)
                                pts.append((normal, pt1, pt2, penetration - pen_diff[Y]))
                    else:
                        assert False  # TODO case where == ?

        for p in pts:
            world.collisions.append((p, geom1.body, geom2.body))
        world.collisions_debug = world.collisions  # XXX
