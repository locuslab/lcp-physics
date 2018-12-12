import unittest

import torch

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.forces import Gravity
from lcp_physics.physics.utils import Params,   wrap_tensor


DTYPE = Params.TENSOR_TYPE


class TestBodies(unittest.TestCase):
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

    def testBody(self):
        pass

    def testCircle(self):
        c1 = Circle([0, 0], 1, vel=[0, 0])
        c2 = Circle([0, 0, 0], 1)
        c3 = Circle(torch.tensor([0, 0], dtype=DTYPE), torch.tensor(1, dtype=DTYPE), vel=torch.tensor([0, 0], dtype=DTYPE))
        c4 = Circle(torch.tensor([0, 0, 0], dtype=DTYPE), torch.tensor(1, dtype=DTYPE), vel=torch.tensor([1, 1, 1], dtype=DTYPE))
        c5 = Circle([0, 0, 0], 1, [0, 0, 0], mass=torch.tensor(1, dtype=DTYPE))

        c1.add_no_collision(c2)
        c2.add_force(Gravity())
        c2.apply_forces(1)
        c3.set_p(wrap_tensor([1, 1, 1]))
        c4.move(0.1)

    def testHull(self):
        # test_hull.py
        pass

    def testRect(self):
        r1 = Rect([0, 0], [1, 1], vel=[0, 0])
        r2 = Rect([0, 0, 0], [1, 1])
        r3 = Rect(torch.tensor([0, 0], dtype=DTYPE), [1, 1], vel=torch.tensor([0, 0], dtype=DTYPE))
        r4 = Rect(torch.tensor([0, 0, 0], dtype=DTYPE), [1, 1], vel=torch.tensor([1, 1, 1], dtype=DTYPE))
        r5 = Rect([0, 0, 0], [1, 1], [0, 0, 0], mass=torch.tensor(1, dtype=DTYPE))

        r1.add_no_collision(r2)
        r2.add_force(Gravity())
        r2.apply_forces(1)
        r3.set_p(wrap_tensor([1, 1, 1]))
        r4.move(0.1)


if __name__ == '__main__':
    unittest.main()
