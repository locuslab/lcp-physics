import unittest
import math

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


if __name__ == '__main__':
    unittest.main()
