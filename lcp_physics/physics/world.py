import time
from functools import lru_cache

import ode
import torch
from torch.autograd import Variable

import lcp_physics.physics.engines as engines_module
import lcp_physics.physics.collisions as collisions_module
from lcp_physics.physics.constraints import Joint
from .utils import Indices, Params, cross_2d, get_instance


X, Y = Indices.X, Indices.Y
DIM = Params.DIM

Tensor = Params.TENSOR_TYPE


class World:
    def __init__(self, bodies, constraints=[], dt=Params.DEFAULT_DT, engine=Params.DEFAULT_ENGINE,
                 collision_callback=Params.DEFAULT_COLLISION, eps=Params.DEFAULT_EPSILON,
                 fric_dirs=Params.DEFAULT_FRIC_DIRS, post_stab=Params.POST_STABILIZATION):
        self.collisions_debug = None  # XXX

        # Load classes from string name defined in utils
        self.engine = get_instance(engines_module, engine)
        self.collision_callback = get_instance(collisions_module, collision_callback)

        self.t = 0
        self.dt = dt
        self.eps = eps
        self.fric_dirs = fric_dirs
        self.post_stab = post_stab

        self.bodies = bodies
        self.vec_len = len(self.bodies[0].v)

        self.space = ode.HashSpace()
        for i, b in enumerate(bodies):
            b.geom.body = i
            self.space.add(b.geom)

        self.static_inverse = True
        self.num_constraints = 0
        self.joints = []
        for j in constraints:
            b1, b2 = j.body1, j.body2
            i1 = bodies.index(b1)
            i2 = bodies.index(b2) if b2 else None
            self.joints.append((j, i1, i2))
            self.num_constraints += j.num_constraints
            if isinstance(j, Joint):
                self.static_inverse = False

        M_size = bodies[0].M.size(0)
        self._M = Variable(Tensor(M_size * len(bodies), M_size * len(bodies)).zero_())
        # TODO Better way for diagonal block matrix?
        for i, b in enumerate(bodies):
            self._M[i * M_size:(i + 1) * M_size, i * M_size:(i + 1) * M_size] = b.M

        self.set_v(torch.cat([b.v for b in bodies]))

        self.restitutions = Variable(Tensor(len(self.v)))
        for i in range(len(bodies)):
            self.restitutions[i * self.vec_len:(i + 1) * self.vec_len] = \
                bodies[i].restitution.repeat(3)

        self.collisions = None
        self.find_collisions()

    def step(self):
        dt = self.dt
        start_v = self.v
        start_p = torch.cat([b.p for b in self.bodies])
        start_rot_joints = [(j[0].rot1, j[0].rot2) for j in self.joints]
        start_collisions = self.collisions
        assert all([c[0][3].data[0] <= 0 for c in self.collisions]), \
            'Interpenetration at beginning of step'
        while True:
            new_v = self.engine.solve_dynamics(self, dt, self.post_stab)
            self.set_v(new_v)
            # try step with current dt
            for body in self.bodies:
                body.move(dt)
            for joint in self.joints:
                joint[0].move(dt)
            self.find_collisions()
            if all([c[0][3].data[0] <= 0 for c in self.collisions]):
                break
            else:
                dt /= 2
                # reset state to beginning of step
                # XXX Avoid clones?
                self.set_v(start_v.clone())
                self.set_p(start_p.clone())
                for j, c in zip(self.joints, start_rot_joints):
                    # XXX Clone necessary?
                    j[0].rot1 = c[0].clone()
                    j[0].rot2 = c[1].clone() if j[0].rot2 is not None else None
                    j[0].update_pos()
                self.collisions = start_collisions
        self.t += dt

    def get_v(self):
        return self.v

    def set_v(self, new_v):
        self.v = new_v
        for i, b in enumerate(self.bodies):
            b.v = self.v[i * len(b.v):(i + 1) * len(b.v)]

    def set_p(self, new_p):
        for i, b in enumerate(self.bodies):
            b.set_p(new_p[i * self.vec_len:(i + 1) * self.vec_len])

    def apply_forces(self, t):
        return torch.cat([b.apply_forces(t) for b in self.bodies])

    def find_collisions(self):
        self.collisions = []
        # ODE collision detection
        self.space.collide([self], self.collision_callback)

    def M(self):
        return self._M

    def Je(self):
        Je = Variable(Tensor(self.num_constraints,
                             self.vec_len * len(self.bodies)).zero_())
        row = 0
        for joint in self.joints:
            J1, J2 = joint[0].J()
            i1 = joint[1]
            i2 = joint[2]
            Je[row:row + J1.size(0),
               i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
            if J2 is not None:
                Je[row:row + J2.size(0),
                   i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2
            row += J1.size(0)
        return Je

    def Jc(self):
        Jc = Variable(Tensor(len(self.collisions), self.vec_len * len(self.bodies)).zero_())
        for i, collision in enumerate(self.collisions):
            c = collision[0]  # c = (normal, collision_pt_1, collision_pt_2)
            i1 = collision[1]
            i2 = collision[2]
            J1 = torch.cat([cross_2d(c[1], c[0]).unsqueeze(1),
                            c[0].unsqueeze(0)], dim=1)
            J2 = -torch.cat([cross_2d(c[2], c[0]).unsqueeze(1),
                             c[0].unsqueeze(0)], dim=1)
            Jc[i, i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
            Jc[i, i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2
        return Jc

    def Jf(self):
        Jf = Variable(Tensor(len(self.collisions) * self.fric_dirs,
                             self.vec_len * len(self.bodies)).zero_())
        for i, collision in enumerate(self.collisions):
            c = collision[0]  # c = (normal, collision_pt_1, collision_pt_2)
            # find orthogonal vector in 2D
            dir1 = torch.cross(torch.cat([c[0], Variable(Tensor(1).zero_())]),
                               Variable(Tensor([0, 0, 1])))[:DIM]
            dir2 = -dir1
            i1 = collision[1]  # body 1 index
            i2 = collision[2]  # body 2 index
            J1 = torch.cat([
                torch.cat([cross_2d(c[1], dir1).unsqueeze(1),
                           dir1.unsqueeze(0)], dim=1),
                torch.cat([cross_2d(c[1], dir2).unsqueeze(1),
                           dir2.unsqueeze(0)], dim=1),
            ], dim=0)
            J2 = torch.cat([
                torch.cat([cross_2d(c[2], dir1).unsqueeze(1),
                           dir1.unsqueeze(0)], dim=1),
                torch.cat([cross_2d(c[2], dir2).unsqueeze(1),
                           dir2.unsqueeze(0)], dim=1),
            ], dim=0)
            Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs,
               i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
            Jf[i * self.fric_dirs:(i + 1) * self.fric_dirs,
               i2 * self.vec_len:(i2 + 1) * self.vec_len] = -J2
        return Jf

    def mu(self):
        return self._memoized_mu(*[(c[1], c[2]) for c in self.collisions])

    @lru_cache()
    def _memoized_mu(self, *collisions):
        # collisions is argument so that lru_cache works
        mu = Variable(Tensor(len(self.collisions)).zero_())
        for i, collision in enumerate(self.collisions):
            i1 = collision[1]
            i2 = collision[2]
            mu[i] = self.bodies[i1].fric_coeff * self.bodies[i2].fric_coeff
        return torch.diag(mu)

    def E(self):
        return self._memoized_E(len(self.collisions))

    @lru_cache()
    def _memoized_E(self, num_collisions):
        n = self.fric_dirs * num_collisions
        E = Tensor(n, num_collisions).zero_()
        for i in range(num_collisions):
            E[i * self.fric_dirs: (i + 1) * self.fric_dirs, i] += 1
        return Variable(E)

    def save_state(self):
        raise NotImplementedError
        # p = torch.cat([Variable(b.p.data) for b in self.bodies])
        # state_dict = {'p': p, 'v': Variable(self.v.data), 't': self.t}
        # return state_dict

    def load_state(self, state_dict):
        raise NotImplementedError
        # self.set_p(state_dict['p'])
        # self.set_v(state_dict['v'])
        # self.t = state_dict['t']
        # self._M.detach_()
        # self.restitutions.detach_()
        # import inspect
        # for b in self.bodies:
        #     for m in inspect.getmembers(b, lambda x: isinstance(x, Variable)):
        #         m[1].detach_()
        # for j in self.joints:
        #     for m in inspect.getmembers(j, lambda x: isinstance(x, Variable)):
        #         m[1].detach_()
        # self.find_collisions()

    def reset_engine(self):
        raise NotImplementedError
        # self.engine = self.engine.__class__()


class BatchWorld:
    def __init__(self, bodies, constraints=[], dt=Params.DEFAULT_DT, engine=Params.DEFAULT_ENGINE,
                 collision_callback=Params.DEFAULT_COLLISION, eps=Params.DEFAULT_EPSILON,
                 fric_dirs=Params.DEFAULT_FRIC_DIRS, post_stab=Params.POST_STABILIZATION):
        self.t = 0.
        self.dt = dt
        self.engine = get_instance(engines_module, engine)
        self.post_stab = post_stab

        self.worlds = []
        for i in range(len(bodies)):
            w = World(bodies[i], constraints[i], dt=dt, engine=engine,
                      collision_callback=collision_callback, eps=eps,
                      fric_dirs=fric_dirs,
                      post_stab=post_stab)
            self.worlds.append(w)

        self.vec_len = self.worlds[0].vec_len
        self._v = None
        self.v_changed = True
        self.collisions = self.has_collisions()
        self._restitutions = torch.cat([w.restitutions.unsqueeze(0) for w in self.worlds], dim=0)

    def step(self):
        dt = self.dt
        start_vs = self.get_v()
        self.v_changed = True
        start_ps = torch.cat([torch.cat([b.p for b in w.bodies]).unsqueeze(0) for w in self.worlds], dim=0)
        start_rot_joints = [[(j[0].rot1, j[0].rot2) for j in w.joints] for w in self.worlds]
        start_collisions = [w.collisions for w in self.worlds]
        for w in self.worlds:
            assert all([c[0][3].data[0] <= 0 for c in w.collisions]), \
                'Interpenetration at beginning of step'
        while True:
            self.collisions = self.has_collisions()
            new_v = self.engine.batch_solve_dynamics(self, dt, self.post_stab)
            self.set_v(new_v)
            # try step with current dt
            done = []
            for w in self.worlds:
                for body in w.bodies:
                    body.move(dt)
                for joint in w.joints:
                    joint[0].move(dt)
                w.find_collisions()
                done.append(all([c[0][3].data[0] <= 0 for c in w.collisions]))
            if all(done):
                break
            else:
                dt /= 2
                # reset state to beginning of step
                # XXX Avoid clones?
                self.set_v(start_vs.clone())
                self.set_p(start_ps.clone())
                for i, w in enumerate(self.worlds):
                    for j, c in zip(w.joints, start_rot_joints[i]):
                        # XXX Clone necessary?
                        j[0].rot1 = c[0].clone()
                        j[0].rot2 = c[1].clone() if j[0].rot2 is not None else None
                        j[0].update_pos()
                    w.collisions = start_collisions[i]
        self.t += dt
        for w in self.worlds:
            w.t += dt

    def get_v(self, num_colls=None):
        if self.v_changed:
            self._v = torch.cat([w.v.unsqueeze(0) for w in self.worlds], dim=0)
            self.v_changed = False
        # TODO Optimize / organize
        if num_colls is not None:
            v = torch.cat([w.v.unsqueeze(0) for w in self.worlds if len(w.collisions) == num_colls], dim=0)
            return v
        return self._v

    def set_v(self, new_v):
        for i, w in enumerate(self.worlds):
            w.set_v(new_v[i])

    def restitutions(self, num_colls=None):
        # TODO Organize / consolidate on other class
        if num_colls is not None:
            r = torch.cat([w.restitutions.unsqueeze(0) for w in self.worlds if len(w.collisions) == num_colls], dim=0)
            return r
        else:
            return self._restitutions

    def set_p(self, new_p):
        for i, w in enumerate(self.worlds):
            w.set_p(new_p[i])

    def has_collisions(self):
        return any([w.collisions for w in self.worlds])

    def has_n_collisions(self, num_colls):
        ret = torch.ByteTensor([len(w.collisions) == num_colls for w in self.worlds])
        if self.worlds[0]._M.is_cuda:
            ret = ret.cuda()
        return ret
    
    def apply_forces(self, t):
        forces = []
        for w in self.worlds:
            forces.append(torch.cat([b.apply_forces(t) for b in w.bodies]).unsqueeze(0))
        return torch.cat(forces, dim=0)

    def find_collisions(self):
        self.collisions = []
        # ODE collision detection
        self.space.collide([self], self.collision_callback)

    # def gather_batch(self, func):
    #     gather = []
    #     for w in self.worlds:
    #         gather.append(func().unsqueeze(0))
    #     return torch.cat(gather, dim=0)

    def M(self, num_colls=None):
        Ms = []
        for w in self.worlds:
            if num_colls is None or len(w.collisions) == num_colls:
                Ms.append(w.M().unsqueeze(0))
        M = torch.cat(Ms, dim=0)
        return M

    def invM(self, num_colls=None):
        invMs = []
        for w in self.worlds:
            if num_colls is None or len(w.collisions) == num_colls:
                invMs.append(w.invM().unsqueeze(0))
        invM = torch.cat(invMs, dim=0)
        return invM

    def Je(self, num_colls=None):
        jes = []
        for w in self.worlds:
            if num_colls is None or len(w.collisions) == num_colls:
                tmp = w.Je()
                tmp = tmp.unsqueeze(0) if tmp.dim() > 0 else tmp
                jes.append(tmp)
        if jes[0].dim() > 0:
            Je = torch.cat(jes, dim=0)
        else:
            Je = Variable(Tensor([]))
        return Je

    def Jc(self, num_colls=None):
        # max_collisions = max([len(w.collisions) for w in self.worlds])
        jcs = []
        for w in self.worlds:
            if len(w.collisions) == num_colls:
                jcs.append(w.Jc().unsqueeze(0))
            # else:
            #     jcs.append(Variable(w._M.data.new(1, max_collisions,
            #         self.vec_len * len(w.bodies)).zero_()))
        Jc = torch.cat(jcs, dim=0)
        return Jc

    def Jf(self, num_colls=None):
        # max_collisions = max([len(w.collisions) for w in self.worlds])
        jfs = []
        for w in self.worlds:
            if num_colls is None or len(w.collisions) == num_colls:
                jfs.append(w.Jf().unsqueeze(0))
            # else:
            #     jfs.append(Variable(w._M.data.new(1, 2 * max_collisions,
            #         self.vec_len * len(w.bodies)).zero_()))
        Jf = torch.cat(jfs, dim=0)
        return Jf

    def mu(self, num_colls=None):
        # max_collisions = max([len(w.collisions) for w in self.worlds])
        mus = []
        for w in self.worlds:
            if num_colls is None or len(w.collisions) == num_colls:
                mus.append(w.mu().unsqueeze(0))
            # else:
            #     mus.append(Variable(w._M.data.new(1, max_collisions,
            #         max_collisions).zero_()))
        mu = torch.cat(mus, dim=0)
        return mu

    def E(self, num_colls=None):
        # max_collisions = max([len(w.collisions) for w in self.worlds])
        Es = []
        for w in self.worlds:
            if num_colls is None or len(w.collisions) == num_colls:
                Es.append(w.E().unsqueeze(0))
            # else:
            #     Es.append(Variable(w._M.data.new(1, 2 * max_collisions,
            #         max_collisions).zero_()))
        E = torch.cat(Es, dim=0)
        return E

    def save_state(self):
        raise NotImplementedError

    def load_state(self, state_dict):
        raise NotImplementedError

    def reset_engine(self):
        raise NotImplementedError


def run_world(world, dt=Params.DEFAULT_DT, run_time=10,
              screen=None, recorder=None):
    """Helper function to run a simulation forward once a world is created.
    """
    # If in batched mode don't display simulation
    if hasattr(world, 'worlds'):
        screen = None

    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    animation_dt = dt
    elapsed_time = 0.
    prev_frame_time = -animation_dt
    start_time = time.time()

    while world.t < run_time:
        world.step()

        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            if elapsed_time - prev_frame_time >= animation_dt or recorder:
                prev_frame_time = elapsed_time

                screen.blit(background, (0, 0))
                update_list = []
                for body in world.bodies:
                    update_list += body.draw(screen)
                for joint in world.joints:
                    update_list += joint[0].draw(screen)

                # XXX visualize collision points and normal for debug
                if world.collisions_debug:
                    for c in world.collisions_debug:
                        (normal, p1, p2, penetration), b1, b2 = c
                        b1_pos = world.bodies[b1].pos
                        b2_pos = world.bodies[b2].pos
                        p1 = p1 + b1_pos
                        p2 = p2 + b2_pos
                        pygame.draw.circle(screen, (0, 255, 0), p1.data.numpy().astype(int), 5)
                        pygame.draw.circle(screen, (0, 0, 255), p2.data.numpy().astype(int), 5)
                        pygame.draw.line(screen, (0, 255, 0), p1.data.numpy().astype(int),
                                         (p1.data.numpy() + normal.data.numpy() * 100).astype(int), 3)

                if not recorder:
                    # Don't refresh screen if recording
                    pygame.display.update(update_list)
                    # pygame.display.flip()  # XXX
                else:
                    recorder.record(world.t)

            elapsed_time = time.time() - start_time
            if not recorder:
                # Adjust frame rate dynamically to keep real time
                wait_time = world.t - elapsed_time
                if wait_time >= 0 and not recorder:
                    wait_time += animation_dt  # XXX
                    time.sleep(max(wait_time - animation_dt, 0))
                #     animation_dt -= 0.005 * wait_time
                # elif wait_time < 0:
                #     animation_dt += 0.005 * -wait_time
                # elapsed_time = time.time() - start_time

        elapsed_time = time.time() - start_time
        print('\r ', '{} / {}  {} '.format(int(world.t), int(elapsed_time),
                                           1 / animation_dt), end='')
