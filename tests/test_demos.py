import unittest
import math

import torch
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Rect
from lcp_physics.physics.constraints import Joint, TotalConstraint, XConstraint, YConstraint
from lcp_physics.physics.forces import ExternalForce, down_force, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults
from lcp_physics.physics.world import World, run_world


TIME = 20
DT = Defaults.DT


class TestDemos(unittest.TestCase):
    def setUp(self):
        # Run without displaying
        self.screen = None
        # Run with display
        # import pygame
        # pygame.init()
        # width, height = 1000, 600
        # self.screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        # self.screen.set_alpha(None)
        # pygame.display.set_caption('2D Engine')

    def testDemo(self):
        bodies = []
        joints = []
        # Ball hitting object constrained by 1 joint
        for i in range(1, 3):
            c = Circle([150, 150 + 80 * (i - 1)], 20)
            if i == 1:
                c.add_force(ExternalForce(vert_impulse, multiplier=500))
            bodies.append(c)
        joints.append(Joint(bodies[-1], None, [140, 220]))

        # Ball bouncing on body fixed in place
        for i in range(1, 3):
            c = Circle([300 + 1 * (i - 1), 150 + 80 * (i - 1)], 20)
            if i == 1:
                c.add_force(ExternalForce(down_force, multiplier=100))
            bodies.append(c)
        joints.append(TotalConstraint(bodies[-1]))

        # 2 free ball collision angled
        for i in range(1, 3):
            c = Circle([225 - 10 * (i - 1), 300 + 80 * (i - 1)], 20)
            if i == 1:
                c.add_force(ExternalForce(down_force, multiplier=100))
            bodies.append(c)

        # 2 free ball collision straight
        for i in range(1, 3):
            c = Circle([375, 300 + 80 * (i - 1)], 20)
            if i == 1:
                c.add_force(ExternalForce(vert_impulse, multiplier=500))
            bodies.append(c)

        r = Rect([300, 500], [40, 40])
        r.add_force(ExternalForce(down_force, multiplier=-100))
        r.v[0] = -1
        bodies.append(r)

        r = Rect([300, 50], [40, 40])
        r.add_force(ExternalForce(down_force, multiplier=100))
        r.v[0] = -1
        for b in bodies:
            b.add_no_contact(r)
        # bodies.append(r)

        world = World(bodies, joints, dt=DT)
        run_world(world, run_time=10, screen=self.screen)

    def testChain(self):
        bodies = []
        joints = []

        # make chain of rectangles
        r = Rect([300, 50], [20, 60])
        bodies.append(r)
        joints.append(XConstraint(r))
        joints.append(YConstraint(r))
        for i in range(1, 10):
            r = Rect([300, 50 + 50 * i], [20, 60])
            bodies.append(r)
            joints.append(Joint(bodies[-1], bodies[-2], [300, 25 + 50 * i]))
            bodies[-1].add_no_contact(bodies[-2])
        bodies[-1].add_force(ExternalForce(down_force, multiplier=100))

        # make projectile
        c = Circle([50, 500], 20, restitution=1)
        bodies.append(c)
        c.add_force(ExternalForce(hor_impulse, multiplier=1000))

        recorder = None
        # recorder = Recorder(DT, self.screen)
        world = World(bodies, joints, dt=DT)
        run_world(world, run_time=TIME, screen=self.screen, recorder=recorder)

    def testSlide(self):
        bodies = []
        joints = []

        r = Rect([500, 300], [900, 10])
        r.v[0] = math.pi / 32
        r.move(1)
        r.v[0] = 0.
        bodies.append(r)
        joints.append(TotalConstraint(r))

        r = Rect([100, 100], [60, 60])
        r.v[0] = -math.pi / 8 * 0
        r.move(1)
        r.v[0] = 0.
        bodies.append(r)
        r.add_force(ExternalForce(down_force, multiplier=100))
        # r.add_force(ExternalForce(hor_impulse, multiplier=-100))

        # c = Circle([100, 150], 30)
        # bodies.append(c)
        # c.add_force(ExternalForce(gravity, multiplier=100))

        # c = Circle([50, 550], 30)
        # c.add_force(ExternalForce(rot_impulse, multiplier=1000))
        # bodies.append(c)

        # XXX
        # c = Circle([875, 100], 30)
        # bodies.append(c)
        # c.add_force(ExternalForce(gravity, multiplier=100))

        recorder = None
        # recorder = Recorder(DT, self.screen)
        world = World(bodies, joints, dt=DT)
        run_world(world, run_time=TIME, screen=self.screen, recorder=recorder)

    def testFric(self):

        restitution = 0.75
        fric_coeff = 1

        bodies = []
        joints = []

        def timed_force(t):
            if 1 < t < 2:
                return ExternalForce.RIGHT
            else:
                return ExternalForce.ZEROS

        r = Rect([400, 400], [900, 10], restitution=restitution, fric_coeff=fric_coeff)
        bodies.append(r)
        r.add_force(ExternalForce(timed_force, multiplier=100))
        r.add_force(ExternalForce(down_force, multiplier=100))

        c = Circle([200, 364], 30, restitution=restitution, fric_coeff=fric_coeff)
        bodies.append(c)
        c.add_force(ExternalForce(down_force, multiplier=100))

        c = Circle([50, 436], 30, restitution=restitution, fric_coeff=fric_coeff)
        bodies.append(c)
        joints.append(XConstraint(c))
        joints.append(YConstraint(c))
        c = Circle([800, 436], 30, restitution=restitution, fric_coeff=fric_coeff)
        bodies.append(c)
        joints.append(XConstraint(c))
        joints.append(YConstraint(c))

        clock = Circle([975, 575], 20, vel=[1, 0, 0])
        bodies.append(clock)

        recorder = None
        # recorder = Recorder(DT, screen)
        world = World(bodies, joints, dt=DT)
        run_world(world, run_time=10, screen=self.screen, recorder=recorder)

    def testAGrad(self):
        def make_world(learned_force):
            bodies = []
            joints = []

            target = Circle([500, 300], 30)
            bodies.append(target)

            c1 = Circle([250, 210], 30)
            bodies.append(c1)
            c1.add_force(ExternalForce(learned_force))
            c1.add_no_contact(target)

            c2 = Circle([400, 250], 30)
            bodies.append(c2)
            c2.add_no_contact(target)

            world = World(bodies, joints, dt=DT)
            return world, c2, target

        initial_force = torch.DoubleTensor([0, 3, 0])
        initial_force[2] = 0
        initial_force = Variable(initial_force, requires_grad=True)

        # Initial demo
        learned_force = lambda t: initial_force if t < 0.1 else ExternalForce.ZEROS
        # learned_force = gravity
        world, c, target = make_world(learned_force)
        # initial_state = world.save_state()
        # next_fric_coeff = Variable(torch.DoubleTensor([1e-7]), requires_grad=True)
        # c.fric_coeff = next_fric_coeff
        # initial_state = world.save_state()
        run_world(world, run_time=TIME, screen=None)

        learning_rate = 0.001
        max_iter = 100

        dist_hist = []
        last_dist = 1e10
        for i in range(max_iter):
            learned_force = lambda t: initial_force if t < 0.1 else ExternalForce.ZEROS

            world, c, target = make_world(learned_force)
            # world.load_state(initial_state)
            # world.reset_engine()
            # c = world.bodies[0]
            # c.fric_coeff = next_fric_coeff
            run_world(world, run_time=TIME, screen=None)

            dist = (target.pos - c.pos).norm()
            dist.backward()
            grad = initial_force.grad.data
            # grad.clamp_(-10, 10)
            initial_force = Variable(initial_force.data - learning_rate * grad, requires_grad=True)
            # grad = c.fric_coeff.grad.data
            # grad.clamp_(-10, 10)
            # temp = c.fric_coeff.data - learning_rate * grad
            # temp.clamp_(1e-7, 1)
            learning_rate /= 1.1
            # next_fric_coeff = Variable(temp, requires_grad=True)
            # print(next_fric_coeff)
            if abs((last_dist - dist).data[0]) < 1e-5:
                break
            last_dist = dist
            dist_hist.append(dist)

        world = make_world(learned_force)[0]
        # c.fric_coeff = next_fric_coeff
        # world.load_state(initial_state)
        # world.reset_engine()
        run_world(world, run_time=TIME, screen=None, recorder=None)
        dist = (target.pos - c.pos).norm()

    def testInference(self):
        def make_world(forces, mass):
            bodies = []
            joints = []

            # make chain of rectangles
            r = Rect([300, 50], [20, 60])
            bodies.append(r)
            joints.append(Joint(r, None, [300, 30]))
            for i in range(1, 10):
                if i < 9:
                    r = Rect([300, 50 + 50 * i], [20, 60])
                else:
                    r = Rect([300, 50 + 50 * i], [20, 60], mass=mass)
                bodies.append(r)
                joints.append(Joint(bodies[-1], bodies[-2], [300, 25 + 50 * i]))
                bodies[-1].add_no_contact(bodies[-2])
            bodies[-1].add_force(ExternalForce(down_force, multiplier=100))

            # make projectile
            m = 13
            c1 = Circle([50, 500], 20)
            bodies.append(c1)
            for f in forces:
                c1.add_force(ExternalForce(f, multiplier=100 * m))

            world = World(bodies, joints, dt=DT)
            return world, r

        def positions_run_world(world, dt=Defaults.DT, run_time=10,
                                screen=None, recorder=None):
            positions = [torch.cat([b.p for b in world.bodies])]

            while world.t < run_time:
                world.step()
                positions.append(torch.cat([b.p for b in world.bodies]))
            return positions

        MASS_EPS = 1e-7
        forces = [hor_impulse]
        ground_truth_mass = Variable(torch.DoubleTensor([7]))
        world, c = make_world(forces, ground_truth_mass)

        ground_truth_pos = positions_run_world(world, run_time=10, screen=None, recorder=None)
        ground_truth_pos = [p.data for p in ground_truth_pos]
        ground_truth_pos = Variable(torch.cat(ground_truth_pos))

        learning_rate = 0.01
        max_iter = 100

        next_mass = Variable(torch.DoubleTensor([1.3]), requires_grad=True)
        loss_hist = []
        mass_hist = [next_mass]
        last_dist = 1e10
        for i in range(max_iter):
            print(i, end='\r')
            world, c = make_world(forces, next_mass)
            # world.load_state(initial_state)
            # world.reset_engine()
            positions = positions_run_world(world, run_time=10, screen=None)
            positions = torch.cat(positions)
            positions = positions[:len(ground_truth_pos)]
            # temp_ground_truth_pos = ground_truth_pos[:len(positions)]

            loss = torch.nn.MSELoss()(positions, ground_truth_pos)
            loss.backward()
            grad = c.mass.grad.data
            # clip gradient
            grad = torch.max(torch.min(grad, torch.DoubleTensor([100])), torch.DoubleTensor([-100]))
            temp = c.mass.data - learning_rate * grad
            temp = max(MASS_EPS, temp[0])
            next_mass = Variable(torch.DoubleTensor([temp]), requires_grad=True)
            # learning_rate /= 1.1
            if abs((last_dist - loss).data[0]) < 1e-3:
                break
            last_dist = loss
            loss_hist.append(loss)
            mass_hist.append(next_mass)

        assert abs(next_mass.data[0] - ground_truth_mass.data[0]) <= 1e-1


if __name__ == '__main__':
    unittest.main()
