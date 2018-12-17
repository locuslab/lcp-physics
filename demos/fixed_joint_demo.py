import math
import sys

import pygame

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import TotalConstraint, FixedJoint
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Defaults, Recorder
from lcp_physics.physics.world import World, run_world


TIME = 20
DT = Defaults.DT


def fixed_joint_demo(screen):
    bodies = []
    joints = []
    restitution = 0.5
    fric_coeff = 0.15

    r = Rect([120, 100], [60, 60],
             restitution=restitution, fric_coeff=fric_coeff)
    bodies.append(r)
    r.add_force(Gravity(g=100))
    r2 = Rect([160, 100], [60, 60],
              restitution=restitution, fric_coeff=fric_coeff)
    bodies.append(r2)
    joints += [FixedJoint(r, r2)]
    r2.add_no_contact(r)
    r2.add_force(Gravity(g=100))

    inclination = math.pi / 32
    r = Rect([inclination, 500, 500], [900, 10],
             restitution=restitution, fric_coeff=fric_coeff)
    bodies.append(r)
    joints.append(TotalConstraint(r))

    recorder = None
    # recorder = Recorder(DT, screen)
    world = World(bodies, joints, dt=DT)
    run_world(world, run_time=TIME, screen=screen, recorder=recorder)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-nd':
        # Run without displaying
        screen = None
    else:
        pygame.init()
        width, height = 1000, 600
        screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        screen.set_alpha(None)
        pygame.display.set_caption('2D Engine')

    fixed_joint_demo(screen)
