import random

import ode

import torch

from .bodies import Circle
from .utils import Indices, Defaults, left_orthogonal


X = Indices.X
Y = Indices.Y
DIM = Defaults.DIM


class ContactHandler:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class OdeContactHandler(ContactHandler):
    def __call__(self, args, geom1, geom2):
        if geom1 in geom2.no_contact:
            return
        world = args[0]
        base_tensor = world.bodies[0].p

        contacts = ode.collide(geom1, geom2)
        for c in contacts:
            point, normal, penetration, geom1, geom2 = c.getContactGeomParams()
            # XXX Simple disambiguation of 3D repetition of contacts
            if point[2] > 0:
                continue
            normal = base_tensor.new_tensor(normal[:DIM])
            point = base_tensor.new_tensor(point)
            penetration = base_tensor.new_tensor([penetration])
            penetration -= 2 * world.eps
            if penetration.item() < -2 * world.eps:
                return
            p1 = point - base_tensor.new_tensor(geom1.getPosition())
            p2 = point - base_tensor.new_tensor(geom2.getPosition())
            world.contacts.append(((normal, p1[:DIM], p2[:DIM], penetration),
                                    geom1.body, geom2.body))

            world.contacts_debug = world.contacts  # XXX


class DiffContactHandler(ContactHandler):
    """Differentiable contact handler, operations to calculate contact manifold
    are done in autograd.
    """
    def __init__(self):
        self.debug_callback = OdeContactHandler()

    def __call__(self, args, geom1, geom2):
        # self.debug_callback(args, geom1, geom2)

        if geom1 in geom2.no_contact:
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
            if penetration.item() < -world.eps:
                return
            normal = normal / dist

            # contact points on surface of object if not interpenetrating,
            #  otherwise its the point  midway between two objects inside of them
            p1 = -normal * b1.rad
            p2 = normal * b2.rad
            if penetration > 0:
                p1 = p1 + normal * penetration / 2  # p1 = -normal * (b1.rad - penetration / 2)
                p2 = p2 - normal * penetration / 2  # p2 = normal * (b2.rad - penetration / 2)

            pts = [[normal, p1, p2, penetration]]
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
                if len(ids_used) == 2:
                    # use orthogonal when closest is in segment
                    search_dir = left_orthogonal(simplex[ids_used[0]] - simplex[ids_used[1]])
                    if search_dir.dot(test_point - simplex[ids_used[0]]).item() < 0:
                        search_dir = -search_dir
                else:
                    search_dir = test_point - closest
                if search_dir[0].item() == 0 and search_dir[1].item() == 0:
                    break
                support, _ = self.get_support(b2.verts, search_dir)
                if support in set(simplex):
                    break
                simplex = [simplex[idx] for idx in ids_used]  # remove unused points
                simplex.append(support)
            if len(ids_used) < 3:
                best_pt2 = closest
                closest = closest + b2.pos
                best_pt1 = closest - b1.pos
                best_dist = torch.norm(closest - b1.pos) - b1.rad
                if best_dist.item() > world.eps:
                    return
                # normal points from closest point to circle center
                best_normal = -best_pt1 / torch.norm(best_pt1)
            else:
                # SAT for circle vs hull if deep penetration
                best_dist = torch.tensor(-1e10)
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

                    if dist.item() > best_dist.item():
                        b2.last_sat_idx = idx
                        if dist.item() > world.eps:
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
            pts = [[best_normal, best_pt1, best_pt2, -best_dist]]
        else:
            # SAT for hull x hull contact
            # TODO Optimize for rectangle vs rectangle?
            contact1 = self.test_separations(b1, b2, eps=world.eps)
            b1.last_sat_idx = contact1[6]
            if contact1[0].item() > world.eps:
                return
            contact2 = self.test_separations(b2, b1, eps=world.eps)
            b2.last_sat_idx = contact2[6]
            if contact2[0].item() > world.eps:
                return
            if contact2[0].item() > contact1[0].item():
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
                    if dist.item() <= world.eps:
                        pt1 = v + normal * -dist
                        pt2 = pt1 + b2.pos - b1.pos
                        pts.append([normal, pt2, pt1, -dist])
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
                    if dist.item() <= world.eps:
                        pt1 = v + normal * -dist
                        pt2 = pt1 + b1.pos - b2.pos
                        pts.append([-normal, pt1, pt2, -dist])

        for p in pts:
            world.contacts.append([p, geom1.body, geom2.body])

        # smooth contact hack
        for i, contact in enumerate(world.contacts):
            # at 0 penetration (objects exact contact) we want p percent of contact normal.
            # compute adjustment with inverse of sigmoid
            p = torch.tensor(0.975)
            delta = torch.log(p / (1 - p))

            # contact[0] = (normal, pt1, pt2, penetration_dist)
            contact[0][0] *= torch.sigmoid(contact[0][3] + delta)

        world.contacts_debug = world.contacts  # XXX

    @staticmethod
    def get_support(points, direction):
        best_point = None
        best_norm = -1.
        for i, p in enumerate(points):
            cur_norm = p.dot(direction).item()
            if cur_norm >= best_norm:
                best_point = p
                best_idx = i
                best_norm = cur_norm
        return best_point, best_idx

    @staticmethod
    def test_separations(hull1, hull2, eps=0):
        verts1, verts2 = hull1.verts, hull2.verts
        num_verts = len(verts1)
        best_dist = torch.tensor(-1e10)
        best_normal = None
        best_vertex = -1
        start_edge = hull1.last_sat_idx
        for i in range(start_edge, num_verts + start_edge):
            idx = i % num_verts
            edge = verts1[(idx+1) % num_verts] - verts1[idx]
            edge_norm = edge.norm()
            normal = left_orthogonal(edge) / edge_norm
            support_point, support_idx = DiffContactHandler.get_support(verts2, -normal)
            # adjust to hull1's frame
            support_point = support_point + hull2.pos - hull1.pos
            # get distance from support point to edge
            dist = normal.dot(support_point - verts1[idx])

            if dist.item() > best_dist.item():
                if dist.item() > eps:
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
            dot = ref_normal.dot(inc_normal).item()
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
        if distance0.item() >= 0.0:
            clipped_verts.append(verts[0])
        if distance1.item() >= 0.0:
            clipped_verts.append(verts[1])

        # If the points are on different sides of the plane
        if distance0.item() * distance1.item() < 0.0 or len(clipped_verts) < 2:
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
            u, v = DiffContactHandler.get_barycentric_coords(point, simplex)
            if u.item() <= 0:
                return simplex[1], [1]
            elif v.item() <= 0:
                return simplex[0], [0]
            else:
                return u * simplex[0] + v * simplex[1], [0, 1]
        elif len(simplex) == 3:
            uAB, vAB = DiffContactHandler.get_barycentric_coords(point, simplex[0:2])
            uBC, vBC = DiffContactHandler.get_barycentric_coords(point, simplex[1:])
            uCA, vCA = DiffContactHandler.get_barycentric_coords(point, [simplex[2], simplex[0]])
            uABC, vABC, wABC = DiffContactHandler.get_barycentric_coords(point, simplex)

            if vAB.item() <= 0 and uCA.item() <= 0:
                return simplex[0], [0]
            elif vBC.item() <= 0 and uAB.item() <= 0:
                return simplex[1], [1]
            elif vCA.item() <= 0 and uBC.item() <= 0:
                return simplex[2], [2]
            elif uAB.item() > 0 and vAB.item() > 0 and wABC.item() <= 0:
                return uAB * simplex[0] + vAB * simplex[1], [0, 1]
            elif uBC.item() > 0 and vBC.item() > 0 and uABC.item() <= 0:
                return uBC * simplex[1] + vBC * simplex[2], [1, 2]
            elif uCA.item() > 0 and vCA.item() > 0 and vABC.item() <= 0:
                return uCA * simplex[2] + vCA * simplex[0], [2, 0]
            elif uABC.item() > 0 and vABC.item() > 0 and wABC.item() > 0:
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
                torch.cat([verts[0], verts[0].new_ones(1)]).unsqueeze(1),
                torch.cat([verts[1], verts[1].new_ones(1)]).unsqueeze(1),
                torch.cat([verts[2], verts[2].new_ones(1)]).unsqueeze(1),
            ], dim=1)
            invM = torch.inverse(M)
            uvw = torch.matmul(invM, torch.cat([point, point.new_ones(1)]).unsqueeze(1))
            return uvw
        else:
            raise ValueError('Barycentric coords only works for 2 or 3 points')
