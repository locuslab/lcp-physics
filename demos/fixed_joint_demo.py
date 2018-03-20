import math
import sys

import pygame

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.constraints import Joint, YConstraint, XConstraint, RotConstraint, TotalConstraint
from lcp_physics.physics.forces import ExternalForce, gravity, vert_impulse, hor_impulse
from lcp_physics.physics.utils import Params, Recorder
from lcp_physics.physics.world import World, BatchWorld, run_world


TIME = 20
DT = Params.DEFAULT_DT


def fixed_joint_demo(screen):
    bodies = []
    joints = []
    restitution = Params.DEFAULT_RESTITUTION
    fric_coeff = 0.15

    inclination = math.pi / 32
    r = Rect([inclination, 500, 500], [900, 10],
             restitution=restitution, fric_coeff=fric_coeff)
    bodies.append(r)
    joints.append(TotalConstraint(r))

    r = Rect([100, 100], [60, 60],
             restitution=restitution, fric_coeff=fric_coeff)
    bodies.append(r)
    r.add_force(ExternalForce(gravity, multiplier=100))
    r2 = Rect([140, 100], [60, 60],
              restitution=restitution, fric_coeff=fric_coeff)
    bodies.append(r2)
    joints += [
        Joint(r, r2, [120, 80]),
        Joint(r, r2, [120, 120]),
    ]
    r2.add_no_collision(r)
    r2.add_force(ExternalForce(gravity, multiplier=100))

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
