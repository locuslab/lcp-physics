import time
from functools import lru_cache

import ode
import pygame
import torch
from torch.autograd import Variable

import lcp_physics.physics.engines as engines_module
import lcp_physics.physics.collisions as collisions_module
from .utils import Indices, Params, cross_2d, get_instance


X, Y = Indices.X, Indices.Y
DIM = Params.DIM

Tensor = Params.TENSOR_TYPE


class World:
    def __init__(self, bodies, joints, dt=Params.DEFAULT_DT, engine=Params.DEFAULT_ENGINE,
                 collision_callback=Params.DEFAULT_COLLISION, eps=Params.DEFAULT_EPSILON,
                 par_eps=Params.DEFAULT_PAR_EPS, fric_dirs=Params.DEFAULT_FRIC_DIRS,
                 post_stab=Params.POST_STABILIZATION):
        self.collisions_debug = None  # XXX

        # Load classes from string name defined in utils
        self.engine = get_instance(engines_module, engine)
        self.collision_callback = get_instance(collisions_module, collision_callback)

        self.t = 0
        self.dt = dt
        self.eps = eps
        self.par_eps = par_eps
        self.fric_dirs = fric_dirs
        self.post_stab = post_stab

        self.bodies = bodies
        self.vec_len = len(self.bodies[0].v)

        self.space = ode.HashSpace()
        for i, b in enumerate(bodies):
            b.geom.body = i
            self.space.add(b.geom)

        self.joints = []
        for j in joints:
            b1, b2 = j.body1, j.body2
            i1 = bodies.index(b1)
            i2 = None
            if b2 is not None:
                i2 = bodies.index(b2)
            self.joints.append((j, i1, i2))

        M_size = bodies[0].M.size(0)
        self.M = Variable(Tensor(M_size * len(bodies), M_size * len(bodies)).zero_())
        # TODO Better way for diagonal block matrix?
        for i, b in enumerate(bodies):
            self.M[i * M_size:(i+1) * M_size, i * M_size:(i+1) * M_size] = b.M
        self.set_v(torch.cat([b.v for b in bodies]))

        self.restitutions = Variable(Tensor(len(self.v)))
        for i in range(len(bodies)):
            # XXX why is 1/2 correction needed?
            self.restitutions[i * self.vec_len:(i + 1) * self.vec_len] = \
                bodies[i].restitution.repeat(3) / 2

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
            new_v = self.engine.solve_dynamics(self, dt, self.post_stab).squeeze()
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
                self.set_v(start_v)
                self.set_p(start_p.clone())  # XXX Avoid clone?
                for j, c in zip(self.joints, start_rot_joints):
                    # XXX Clone necessary?
                    j[0].rot1 = c[0].clone()
                    j[0].rot2 = c[1].clone() if j[0].rot2 is not None else None
                    j[0].update_pos()
                self.collisions = start_collisions
        self.t += dt

    def set_v(self, new_v):
        self.v = new_v
        for i, b in enumerate(self.bodies):
            b.v = self.v[i * len(b.v):(i+1) * len(b.v)]

    def set_p(self, new_p):
        for i, b in enumerate(self.bodies):
            b.set_p(new_p[i * self.vec_len:(i + 1) * self.vec_len])

    def apply_forces(self, t):
        return torch.cat([b.apply_forces(t) for b in self.bodies])

    def find_collisions(self):
        self.collisions = []
        # ODE collision detection
        self.space.collide([self], self.collision_callback)

    def Je(self):
        Je = Variable(Tensor(DIM * len(self.joints), self.vec_len * len(self.bodies)).zero_())
        for i, joint in enumerate(self.joints):
            J1, J2 = joint[0].J()
            i1 = joint[1]
            i2 = joint[2]
            Je[i * DIM:(i+1) * DIM, i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
            if i2 is not None:
                Je[i * DIM:(i+1) * DIM,
                    i2 * self.vec_len:(i2 + 1) * self.vec_len] = J2
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
            Jf[i * self.fric_dirs:(i+1) * self.fric_dirs,
                i1 * self.vec_len:(i1 + 1) * self.vec_len] = J1
            Jf[i * self.fric_dirs:(i+1) * self.fric_dirs,
                i2 * self.vec_len:(i2 + 1) * self.vec_len] = -J2
        return Jf

    def mu(self):
        return self._memoized_mu(*[(c[1], c[2]) for c in self.collisions])

    @lru_cache()
    def _memoized_mu(self, *collisions):
        # collisions is argument so that lru_cache works
        mu = Variable(torch.zeros(len(self.collisions)))
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
        E = torch.zeros(n, num_collisions)
        for i in range(num_collisions):
            E[i * self.fric_dirs: (i + 1) * self.fric_dirs, i] += 1
        return Variable(E)

    def save_state(self):
        p = torch.cat([Variable(b.p.data) for b in self.bodies])
        state_dict = {'p': p, 'v': Variable(self.v.data), 't': self.t}
        return state_dict

    def load_state(self, state_dict):
        self.set_p(state_dict['p'])
        self.set_v(state_dict['v'])
        self.t = state_dict['t']
        self.M.detach_()
        self.restitutions.detach_()
        import inspect
        for b in self.bodies:
            for m in inspect.getmembers(b, lambda x: isinstance(x, Variable)):
                m[1].detach_()
        for j in self.joints:
            for m in inspect.getmembers(j, lambda x: isinstance(x, Variable)):
                m[1].detach_()
        self.find_collisions()

    def reset_engine(self):
        self.engine = self.engine.__class__()


def run_world(world, dt=Params.DEFAULT_DT, run_time=10,
              screen=None, recorder=None):
    """Helper function to run a simulation forward once a world is created.
    """
    if screen is not None:
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
                # if world.collisions_debug:
                #     for c in world.collisions_debug:
                #         (normal, p1, p2, penetration), b1, b2 = c
                #         b1_pos = world.bodies[b1].pos
                #         b2_pos = world.bodies[b2].pos
                #         p1 = p1 + b1_pos
                #         p2 = p2 + b2_pos
                #         pygame.draw.circle(screen, (0, 255, 0), p1.data.numpy().astype(int), 5)
                #         pygame.draw.circle(screen, (0, 0, 255), p2.data.numpy().astype(int), 5)
                #         pygame.draw.line(screen, (0, 255, 0), p1.data.numpy().astype(int),
                #                          (p1.data.numpy() + normal.data.numpy() * 100).astype(int), 3)

                if not recorder:
                    # Don't refresh screen if recording
                    # pygame.display.update(update_list)
                    pygame.display.flip()  # XXX
                else:
                    recorder.record(world.t)

            elapsed_time = time.time() - start_time
            print('\r ', '{} / {}  {}'.format(int(world.t), int(elapsed_time),
                                              1 / animation_dt), end='')
            if not recorder:
                # Adjust frame rate dynamically to keep real time
                wait_time = world.t - elapsed_time
                # time.sleep(0.5)  # XXX
                if wait_time >= 0 and not recorder:
                    wait_time += animation_dt  # XXX
                    time.sleep(max(wait_time - animation_dt, 0))
                #     animation_dt -= 0.005 * wait_time
                # elif wait_time < 0:
                #     animation_dt += 0.005 * -wait_time
                # elapsed_time = time.time() - start_time
