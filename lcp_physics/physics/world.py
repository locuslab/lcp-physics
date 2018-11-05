import time
from functools import lru_cache

import ode
import torch
from torch.autograd import Variable

from . import engines as engines_module
from . import collisions as collisions_module
from .utils import Indices, Params, cross_2d, get_instance, left_orthogonal


X, Y = Indices.X, Indices.Y
DIM = Params.DIM
TOL = 1e-12

Tensor = Params.TENSOR_TYPE


class World:
    def __init__(self, bodies, constraints=[], dt=Params.DEFAULT_DT, engine=Params.DEFAULT_ENGINE,
                 collision_callback=Params.DEFAULT_COLLISION, eps=Params.DEFAULT_EPSILON,
                 fric_dirs=Params.DEFAULT_FRIC_DIRS, post_stab=Params.POST_STABILIZATION):
        # self.collisions_debug = None  # XXX

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
            if not j.static:
                self.static_inverse = False

        M_size = bodies[0].M.size(0)
        self._M = Variable(Tensor(M_size * len(bodies), M_size * len(bodies)).zero_())
        # XXX Better way for diagonal block matrix?
        for i, b in enumerate(bodies):
            self._M[i * M_size:(i + 1) * M_size, i * M_size:(i + 1) * M_size] = b.M

        self.set_v(torch.cat([b.v for b in bodies]))

        self.collisions = None
        self.find_collisions()

    def step(self, fixed_dt=False):
        dt = self.dt
        if fixed_dt:
            end_t = self.t + self.dt
            while self.t < end_t:
                dt = end_t - self.t
                self.step_dt(dt)
        else:
            self.step_dt(dt)

    def step_dt(self, dt):
        start_v = self.v
        start_p = torch.cat([b.p for b in self.bodies])
        start_rot_joints = [(j[0].rot1, j[0].rot2) for j in self.joints]
        start_collisions = self.collisions
        assert all([c[0][3].data[0] <= TOL for c in self.collisions]), \
            'Interpenetration at beginning of step:\n{}'.format(self.collisions)
        new_v = self.engine.solve_dynamics(self, dt)
        self.set_v(new_v)
        while True:
            # try step with current dt
            for body in self.bodies:
                body.move(dt)
            for joint in self.joints:
                joint[0].move(dt)
            self.find_collisions()
            if all([c[0][3].data[0] <= TOL for c in self.collisions]):
                break
            else:
                dt /= 2
                # reset positions to beginning of step
                # XXX Avoid clone?
                self.set_p(start_p.clone())
                for j, c in zip(self.joints, start_rot_joints):
                    j[0].rot1 = c[0].clone()  # XXX Clone necessary?
                    j[0].update_pos()
                self.collisions = start_collisions

        if self.post_stab:
            dp = self.engine.post_stabilization(self).squeeze(0)
            dp /= 2  # XXX Why 1/2 factor?
            # XXX Clean up / Simplify this update?
            self.set_v(dp)
            for body in self.bodies:
                body.move(dt)
            for joint in self.joints:
                joint[0].move(dt)
            self.set_v(new_v)

            self.find_collisions()  # XXX Necessary to recheck collisions?
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

    def restitutions(self):
        restitutions = Variable(Tensor(len(self.collisions)))
        for i, c in enumerate(self.collisions):
            r1 = self.bodies[c[1]].restitution
            r2 = self.bodies[c[2]].restitution
            restitutions[i] = (r1 + r2) / 2
            # restitutions[i] = math.sqrt(r1 * r2)
        return restitutions

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
            dir1 = left_orthogonal(c[0])
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

    def _memoized_mu(self, *collisions):
        # collisions is argument so that cacheing can be implemented at some point
        mu = Variable(Tensor(len(self.collisions)).zero_())
        for i, collision in enumerate(self.collisions):
            i1 = collision[1]
            i2 = collision[2]
            # mu[i] = torch.sqrt(self.bodies[i1].fric_coeff * self.bodies[i2].fric_coeff)
            mu[i] = 0.5 * (self.bodies[i1].fric_coeff + self.bodies[i2].fric_coeff)
        return torch.diag(mu)

    def E(self):
        return self._memoized_E(len(self.collisions))

    def _memoized_E(self, num_collisions):
        n = self.fric_dirs * num_collisions
        E = Tensor(n, num_collisions).zero_()
        for i in range(num_collisions):
            E[i * self.fric_dirs: (i + 1) * self.fric_dirs, i] += 1
        return Variable(E)

    def save_state(self):
        raise NotImplementedError

    def load_state(self, state_dict):
        raise NotImplementedError

    def reset_engine(self):
        raise NotImplementedError


def run_world(world, dt=Params.DEFAULT_DT, run_time=10,
              print_time=True, screen=None, recorder=None):
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

                # Visualize collision points and normal for debug
                # (Uncomment collisions_debug line in collision handler):
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
        if print_time:
            print('\r ', '{} / {}  {} '.format(int(world.t), int(elapsed_time),
                                               1 / animation_dt), end='')
