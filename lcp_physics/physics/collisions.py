import math
import random

import ode

import torch
from torch.autograd import Variable

from .bodies import Circle
from .utils import Indices, Params, wrap_variable, left_orthogonal


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
            # world.collisions_debug = world.collisions  # XXX


class DiffCollisionHandler(CollisionHandler):
    def __init__(self):
        self.debug_callback = OdeCollisionHandler()

    def __call__(self, args, geom1, geom2):
        # self.debug_callback(args, geom1, geom2)

        if geom1 in geom2.no_collision:
            return
        world = args[0]

        b1 = world.bodies[geom1.body]
        b2 = world.bodies[geom2.body]
        is_circle_g1 = isinstance(b1, Circle)
        is_circle_g2 = isinstance(b2, Circle)
        if is_circle_g1 and is_circle_g2:
            # Simple circle vs circle
            r = b1.rad + b2.rad
            normal = b1.pos - b2.pos
            dist = normal.norm()
            penetration = r - dist
            if penetration.data[0] < -world.eps:
                return
            normal = normal / dist
            p1 = -normal * (b1.rad - penetration / 2)
            p2 = normal * (b2.rad - penetration / 2)
            pts = [(normal, p1, p2, penetration)]
        elif is_circle_g1 or is_circle_g2:
            if is_circle_g2:
                # set circle to b1
                b1, b2 = b2, b1

            # Shallow penetration with GJK
            test_point = b1.pos - b2.pos
            simplex = [random.choice(b2.verts)]
            while True:
                closest, ids_used = self.get_closest(test_point, simplex)
                if len(ids_used) == 3:
                    break
                simplex = [simplex[idx] for idx in ids_used]  # remove unused points
                if len(ids_used) == 2:
                    # use orthogonal when closest is in segment
                    search_dir = left_orthogonal(simplex[0] - simplex[1])
                    if search_dir.dot(test_point - simplex[0]).data[0] < 0:
                        search_dir = -search_dir
                else:
                    search_dir = test_point - closest
                if search_dir.data[0] == 0 and search_dir.data[1] == 0:
                    break
                support, _ = DiffCollisionHandler.get_support(b2.verts, search_dir)
                if support in set(simplex):
                    break
                simplex.append(support)
            if len(ids_used) < 3:
                best_pt2 = closest
                closest = closest + b2.pos
                best_pt1 = closest - b1.pos
                best_dist = torch.norm(closest - b1.pos) - b1.rad
                if best_dist.data[0] > world.eps:
                    return
                # normal points from closest point to circle center
                best_normal = -best_pt1 / torch.norm(best_pt1)
            else:
                # SAT for circle vs hull if deep penetration
                best_dist = wrap_variable(-1e10)
                num_verts = len(b2.verts)
                start_edge = b2.last_sat_idx
                for i in range(start_edge, num_verts + start_edge):
                    idx = i % num_verts
                    edge = b2.verts[(idx+1) % num_verts] - b2.verts[idx]
                    edge_norm = edge.norm()
                    normal = left_orthogonal(edge) / edge_norm
                    # adjust to hull1's frame
                    center = b1.pos - b2.pos
                    # get distance from circle point to edge
                    dist = normal.dot(center - b2.verts[idx]) - b1.rad

                    if dist.data[0] > best_dist.data[0]:
                        b2.last_sat_idx = idx
                        if dist.data[0] > world.eps:
                            # exit early if separating axis found
                            return
                        best_dist = dist
                        best_normal = normal
                        best_pt2 = center + normal * -(dist + b1.rad)
                        best_pt1 = best_pt2 + b2.pos - b1.pos

            if is_circle_g2:
                # flip back values for circle as g2
                best_normal = -best_normal
                best_pt1, best_pt2 = best_pt2, best_pt1
            pts = [(best_normal, best_pt1, best_pt2, -best_dist)]
        else:
            # SAT for hull x hull contact
            # TODO Optimize for rectangle vs rectangle?
            contact1 = self.test_separations(b1, b2, eps=world.eps)
            b1.last_sat_idx = contact1[6]
            if contact1[0].data[0] > world.eps:
                return
            contact2 = self.test_separations(b2, b1, eps=world.eps)
            b2.last_sat_idx = contact2[6]
            if contact2[0].data[0] > world.eps:
                return
            if contact2[0].data[0] > contact1[0].data[0]:
                normal = -contact2[3]
                half_edge_norm = contact2[5] / 2
                ref_edge_idx = contact2[6]
                incident_vertex_idx = contact2[4]
                incident_edge_idx = self.get_incident_edge(normal, b1, incident_vertex_idx)
                incident_verts = [b1.verts[incident_edge_idx],
                                  b1.verts[(incident_edge_idx + 1) % len(b1.verts)]]
                incident_verts = [v + b1.pos - b2.pos for v in incident_verts]
                clip_plane = left_orthogonal(normal)
                clipped_verts = self.clip_segment_to_line(incident_verts, clip_plane,
                                                          half_edge_norm)
                if len(clipped_verts) < 2:
                    return
                clipped_verts = self.clip_segment_to_line(clipped_verts, -clip_plane,
                                                          half_edge_norm)
                pts = []
                for v in clipped_verts:
                    dist = normal.dot(v - b2.verts[ref_edge_idx])
                    if dist.data[0] <= world.eps:
                        pt1 = v + normal * -dist
                        pt2 = pt1 + b2.pos - b1.pos
                        pts.append((normal, pt2, pt1, -dist))
            else:
                normal = -contact1[3]
                half_edge_norm = contact1[5] / 2
                ref_edge_idx = contact1[6]
                incident_vertex_idx = contact1[4]
                incident_edge_idx = self.get_incident_edge(normal, b2, incident_vertex_idx)
                incident_verts = [b2.verts[incident_edge_idx],
                                  b2.verts[(incident_edge_idx+1) % len(b2.verts)]]
                incident_verts = [v + b2.pos - b1.pos for v in incident_verts]
                clip_plane = left_orthogonal(normal)
                clipped_verts = self.clip_segment_to_line(incident_verts, clip_plane,
                                                          half_edge_norm)
                if len(clipped_verts) < 2:
                    return
                clipped_verts = self.clip_segment_to_line(clipped_verts, -clip_plane,
                                                          half_edge_norm)
                pts = []
                for v in clipped_verts:
                    dist = normal.dot(v - b1.verts[ref_edge_idx])
                    if dist.data[0] <= world.eps:
                        pt1 = v + normal * -dist
                        pt2 = pt1 + b1.pos - b2.pos
                        pts.append((-normal, pt1, pt2, -dist))

        for p in pts:
            world.collisions.append((p, geom1.body, geom2.body))
        # world.collisions_debug = world.collisions  # XXX

    @staticmethod
    def get_support(points, direction):
        best_point = None
        best_norm = -1.
        for i, p in enumerate(points):
            cur_norm = p.dot(direction).data[0]
            if cur_norm >= best_norm:
                best_point = p
                best_idx = i
                best_norm = cur_norm
        return best_point, best_idx

    @staticmethod
    def test_separations(hull1, hull2, eps=0):
        verts1, verts2 = hull1.verts, hull2.verts
        num_verts = len(verts1)
        best_dist = wrap_variable(-1e10)
        best_normal = None
        best_vertex = -1
        start_edge = hull1.last_sat_idx
        for i in range(start_edge, num_verts + start_edge):
            idx = i % num_verts
            edge = verts1[(idx+1) % num_verts] - verts1[idx]
            edge_norm = edge.norm()
            normal = left_orthogonal(edge) / edge_norm
            support_point, support_idx = DiffCollisionHandler.get_support(verts2, -normal)
            # adjust to hull1's frame
            support_point = support_point + hull2.pos - hull1.pos
            # get distance from support point to edge
            dist = normal.dot(support_point - verts1[idx])

            if dist.data[0] > best_dist.data[0]:
                if dist.data[0] > eps:
                    # exit early if separating axis found
                    return dist, None, None, None, None, None, idx
                best_dist = dist
                best_normal = normal
                best_pt1 = support_point + normal * -dist
                best_pt2 = best_pt1 + hull1.pos - hull2.pos
                best_vertex = support_idx
                best_edge_norm = edge_norm
                best_edge = idx
        return best_dist, best_pt1, best_pt2, -best_normal, \
            best_vertex, best_edge_norm, best_edge

    @staticmethod
    def get_incident_edge(ref_normal, inc_hull, inc_vertex):
        inc_verts = inc_hull.verts
        # two possible incident edges (pointing to and from incident vertex)
        edges = [(inc_vertex-1) % len(inc_verts), inc_vertex]
        min_dot = 1e10
        best_edge = -1
        for i in edges:
            edge = inc_verts[(i+1) % len(inc_verts)] - inc_verts[i]
            edge_norm = edge.norm()
            inc_normal = left_orthogonal(edge) / edge_norm
            dot = ref_normal.dot(inc_normal).data[0]
            if dot < min_dot:
                min_dot = dot
                best_edge = i
        return best_edge

    @staticmethod
    def clip_segment_to_line(verts, normal, offset):
        clipped_verts = []

        # Calculate the distance of end points to the line
        distance0 = normal.dot(verts[0]) + offset
        distance1 = normal.dot(verts[1]) + offset

        # If the points are behind the plane
        if distance0.data[0] >= 0.0:
            clipped_verts.append(verts[0])
        if distance1.data[0] >= 0.0:
            clipped_verts.append(verts[1])

        # If the points are on different sides of the plane
        if distance0.data[0] * distance1.data[0] < 0.0 or len(clipped_verts) < 2:
            # Find intersection point of edge and plane
            interp = distance0 / (distance0 - distance1)

            # Vertex is hitting edge.
            cv = verts[0] + interp * (verts[1] - verts[0])
            clipped_verts.append(cv)

        return clipped_verts

    @staticmethod
    def get_closest(point, simplex):
        if len(simplex) == 1:
            return simplex[0], [0]
        elif len(simplex) == 2:
            u, v = DiffCollisionHandler.get_barycentric_coords(point, simplex)
            if u.data[0] <= 0:
                return simplex[1], [1]
            elif v.data[0] <= 0:
                return simplex[0], [0]
            else:
                return u * simplex[0] + v * simplex[1], [0, 1]
        elif len(simplex) == 3:
            uAB, vAB = DiffCollisionHandler.get_barycentric_coords(point, simplex[0:2])
            uBC, vBC = DiffCollisionHandler.get_barycentric_coords(point, simplex[1:])
            uCA, vCA = DiffCollisionHandler.get_barycentric_coords(point, [simplex[2], simplex[0]])
            uABC, vABC, wABC = DiffCollisionHandler.get_barycentric_coords(point, simplex)

            if vAB.data[0] <= 0 and uCA.data[0] <= 0:
                return simplex[0], [0]
            elif vBC.data[0] <= 0 and uAB.data[0] <= 0:
                return simplex[1], [1]
            elif vCA.data[0] <= 0 and uBC.data[0] <= 0:
                return simplex[2], [2]
            elif uAB.data[0] > 0 and vAB.data[0] > 0 and wABC.data[0] <= 0:
                return uAB * simplex[0] + vAB * simplex[1], [0, 1]
            elif uBC.data[0] > 0 and vBC.data[0] > 0 and uABC.data[0] <= 0:
                return uBC * simplex[1] + vBC * simplex[2], [1, 2]
            elif uCA.data[0] > 0 and vCA.data[0] > 0 and vABC.data[0] <= 0:
                return uCA * simplex[2] + vCA * simplex[0], [2, 0]
            elif uABC.data[0] > 0 and vABC.data[0] > 0 and wABC.data[0] > 0:
                return point, [0, 1, 2]
            else:
                print(uAB, vAB, uBC, vBC, uCA, vCA, uABC, vABC, wABC)
                raise ValueError('Point does not satisfy any condition in get_closest()')
        else:
            raise ValueError('Simplex should not have more than 3 points in GJK.')

    @staticmethod
    def get_barycentric_coords(point, verts):
        if len(verts) == 2:
            diff = verts[1] - verts[0]
            diff_norm = torch.norm(diff)
            normalized_diff = diff / diff_norm
            u = torch.dot(verts[1] - point, normalized_diff) / diff_norm
            v = torch.dot(point - verts[0], normalized_diff) / diff_norm
            return u, v
        elif len(verts) == 3:
            # TODO Area method instead of LinAlg
            M = torch.cat([
                torch.cat([verts[0], Variable(torch.ones(1)).type_as(verts[0])]).unsqueeze(1),
                torch.cat([verts[1], Variable(torch.ones(1)).type_as(verts[1])]).unsqueeze(1),
                torch.cat([verts[2], Variable(torch.ones(1)).type_as(verts[2])]).unsqueeze(1),
            ], dim=1)
            invM = torch.inverse(M)
            uvw = torch.matmul(invM, torch.cat([point, Variable(torch.ones(1)).type_as(point)]).unsqueeze(1))
            return uvw
        else:
            raise ValueError('Barycentric coords only works for 2 or 3 points')
