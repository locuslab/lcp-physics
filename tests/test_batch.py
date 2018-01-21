import unittest
import math

import torch
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Rect
from lcp_physics.physics.constraints import Joint
from lcp_physics.physics.forces import ExternalForce, gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Params
from lcp_physics.physics.world import BatchWorld, run_world


TIME = 20
DT = Params.DEFAULT_DT


class TestDemos(unittest.TestCase):
    def setUp(self):
        # Run without displaying
        self.screen = None
        # Run with display
        # pygame.init()
        # width, height = 1000, 600
        # self.screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        # self.screen.set_alpha(None)
        # pygame.display.set_caption('2D Engine')

    def testBatch(self):
        # World 1
        bodies1 = []
        joints1 = []

        r = Rect([500, 300], [900, 10])
        bodies1.append(r)
        joints1.append(Joint(r, None, [100, 260]))
        joints1.append(Joint(r, None, [850, 335]))

        c = Circle([100, 100], 30)
        bodies1.append(c)
        c.add_force(ExternalForce(gravity, multiplier=100))

        # World 2
        bodies2 = []
        joints2 = []

        r = Rect([500, 300], [900, 10])
        bodies2.append(r)
        joints2.append(Joint(r, None, [100, 260]))
        joints2.append(Joint(r, None, [850, 335]))

        c = Circle([100, 100], 30)
        bodies2.append(c)
        c.add_force(ExternalForce(gravity, multiplier=100))

        world = BatchWorld([bodies1, bodies2], [joints1, joints2], dt=DT)
        run_world(world, run_time=TIME, screen=self.screen)


if __name__ == '__main__':
    unittest.main()
