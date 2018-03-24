import unittest

from lcp_physics.physics.bodies import Circle, Rect
from lcp_physics.physics.constraints import TotalConstraint
from lcp_physics.physics.forces import ExternalForce, gravity
from lcp_physics.physics.utils import Params
from lcp_physics.physics.world import BatchWorld, run_world


TIME = 20
DT = Params.DEFAULT_DT


class TestBatch(unittest.TestCase):
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

    def testBatch(self):
        pass  # No batch world for now
        # # World 1
        # bodies1 = []
        # joints1 = []
        #
        # r = Rect([500, 300], [900, 10])
        # bodies1.append(r)
        # joints1.append(TotalConstraint(r))
        #
        # c = Circle([100, 100], 30)
        # bodies1.append(c)
        # c.add_force(ExternalForce(gravity, multiplier=100))
        #
        # # World 2
        # bodies2 = []
        # joints2 = []
        #
        # r = Rect([500, 300], [900, 10])
        # bodies2.append(r)
        # joints2.append(TotalConstraint(r))
        #
        # c = Circle([100, 100], 30)
        # bodies2.append(c)
        # c.add_force(ExternalForce(gravity, multiplier=100))
        #
        # # World 3
        # bodies3 = []
        # joints3 = []
        #
        # r = Rect([500, 300], [900, 10])
        # bodies3.append(r)
        # joints3.append(TotalConstraint(r))
        #
        # c = Circle([25, 200], 30)
        # bodies3.append(c)
        # c.add_force(ExternalForce(gravity, multiplier=100))
        #
        # world = BatchWorld([bodies1, bodies2, bodies3], [joints1, joints2, joints3],
        #                    dt=DT)
        # run_world(world, run_time=TIME, screen=self.screen)


if __name__ == '__main__':
    unittest.main()
