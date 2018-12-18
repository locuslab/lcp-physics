import unittest

import math

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import TotalConstraint
from lcp_physics.physics.forces import ExternalForce, down_force, hor_impulse, vert_impulse, rot_impulse
from lcp_physics.physics.utils import Defaults, get_tensor
from lcp_physics.physics.world import World, run_world


TIME = 20
DT = Defaults.DT


class TestHull(unittest.TestCase):
    def setUp(self):
        # Run without displaying
        self.screen = None

        # Run with display
        # import pygame
        # pygame.init()
        # width, height = 1000, 600
        # self.screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        # pygame.display.set_caption('2D Engine')
        # self.screen.set_alpha(None)

    def testAngInertia(self):
        r = Rect([300, 300], [20, 20])
        p = Hull([700, 300], [[math.sqrt(2) * 10 + 10, 0 + 10], [0 + 10, math.sqrt(2) * 10 + 10],
                              [math.sqrt(2)*-10+10, 0+10], [0+10, math.sqrt(2)*-10+10]])
        # p = Polygon([700, 300], [[20, 0], [20, 20], [0, 20], [0, 0]])

        # print(p.pos)
        # print(p.verts)
        # print(p.ang_inertia.item(), r.ang_inertia.item())
        assert p.ang_inertia.item() - r.ang_inertia.item() < 1e-12

    def testCentroid(self):
        p = Hull([700, 300], [[math.sqrt(2) * 10 + 10, 0 + 10], [0 + 10, math.sqrt(2) * 10 + 10],
                              [math.sqrt(2)*-10+10, 0+10], [0+10, math.sqrt(2)*-10+10]])
        assert p.pos[0].item() - 700 == 10 and p.pos[1].item() - 300 == 10

    def testRender(self):
        p1 = Hull([700, 200], [[math.sqrt(2) * 10 + 10, 0 + 10], [0 + 10, math.sqrt(2) * 10 + 10],
                               [math.sqrt(2) * -10 + 10, 0 + 10], [0 + 10, math.sqrt(2) * -10 + 10]])
        p2 = Hull([300, 200], [[60, 0], [60, 60], [0, 60]])
        p2.add_force(ExternalForce(hor_impulse, multiplier=1000))
        p2.add_force(ExternalForce(rot_impulse, multiplier=1000))
        p3 = Hull([700, 400], [[60, 0], [60, 60], [0, 0]])
        p4 = Hull([300, 420], [[60, 0], [0, 60], [0, 0]])
        p4.add_force(ExternalForce(hor_impulse, multiplier=1000))
        p4.add_force(ExternalForce(rot_impulse, multiplier=1000))
        p5 = Hull([500, 300], [[50, 0], [30, 60], [-30, 30], [-60, -30], [0, -60]])
        p5.add_force(ExternalForce(hor_impulse, multiplier=1000))
        p5.add_force(ExternalForce(rot_impulse, multiplier=1000))

        # print(p5.verts)
        # print(get_support(p5.verts, wrap_variable([-1, -1])))

        # world = World([p5])
        world = World([p1, p2, p3, p4, p5])
        run_world(world, screen=self.screen, run_time=180)

    def testSlide(self):
        bodies = []
        joints = []
        restitution = 0.5
        fric_coeff = 0.2

        clock = Circle([975, 575], 20, vel=[1, 0, 0])
        bodies.append(clock)

        p1 = Hull([500, 300], [[450, 5], [-450, 5], [-450, -5], [450, -5]],
                  restitution=restitution, fric_coeff=fric_coeff)
        p1.rotate_verts(get_tensor(math.pi / 32))
        bodies.append(p1)
        joints.append(TotalConstraint(p1))

        # Rectangle
        # p2 = Hull([100, 100], [[30, 30], [-30, 30], [-30, -30], [30, -30]],
        #           restitution=restitution, fric_coeff=fric_coeff)
        # Pentagon
        p2 = Hull([100, 100], [[50, 0], [30, 50], [-30, 30], [-50, -30], [0, -50]],
                  restitution=restitution, fric_coeff=fric_coeff)
        # Hexagon
        p2 = Hull([100, 100], [[50, 0], [30, 50], [-30, 30], [-50, -30], [0, -50], [30, -30]],
                  restitution=restitution, fric_coeff=fric_coeff)
        bodies.append(p2)
        p2.add_force(ExternalForce(down_force, multiplier=100))
        # p2.add_force(ExternalForce(hor_impulse, multiplier=-100))

        recorder = None
        # recorder = Recorder(DT, self.screen)
        world = World(bodies, joints, dt=DT)
        run_world(world, run_time=TIME, screen=self.screen, recorder=recorder)


if __name__ == '__main__':
    unittest.main()
