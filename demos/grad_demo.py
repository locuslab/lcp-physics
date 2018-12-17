import sys
import math

import pygame

import torch
from torch.autograd import Variable

from lcp_physics.physics.bodies import Circle, Rect
from lcp_physics.physics.constraints import Joint
from lcp_physics.physics.forces import ExternalForce, down_force
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.utils import Recorder, plot, Defaults

TIME = 7
DT = Defaults.DT


def grad_demo(screen):
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
    run_world(world, run_time=TIME, screen=screen)

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
        learning_rate *= 0.9
        # next_fric_coeff = Variable(temp, requires_grad=True)
        print(i, '/', max_iter, dist.data[0])
        print(grad)
        # print(next_fric_coeff)
        print(learned_force(0.05))
        print('=======')
        if abs((last_dist - dist).data[0]) < 1e-5:
            break
        last_dist = dist
        dist_hist.append(dist)

    world = make_world(learned_force)[0]
    # c.fric_coeff = next_fric_coeff
    # world.load_state(initial_state)
    # world.reset_engine()
    rec = None
    # rec = Recorder(DT, screen)
    run_world(world, run_time=TIME, screen=screen, recorder=rec)
    dist = (target.pos - c.pos).norm()
    print(dist.data[0])

    # import pickle
    # with open('control_balls_dist_hist.pkl', 'w') as f:
    #     pickle.dump(dist_hist, f)
    plot(dist_hist)


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
    # joints.append(Joint(c2, None, [500, 275]))
    c2.add_no_contact(target)

    world = World(bodies, joints, dt=DT)
    return world, c2, target


# def make_world(learned_force):
#     bodies = []
#     joints = []
#
#     # c = Circle([100, 259], 30)
#     # bodies.append(c)
#     # c.add_force(ExternalForce(learned_force))
#     # c.add_force(ExternalForce(multiplier=10))
#     c = Rect([100, 100], (60, 60))
#     bodies.append(c)
#     c.add_force(ExternalForce(learned_force))
#     # c.add_force(ExternalForce(multiplier=10))
#
#     target = Circle([500, 259], 30)
#     bodies.append(target)
#     target.add_no_collision(c)
#
#     r = Rect([500, 300], [1000, 20])
#     r.v[0] = math.pi / 32
#     r.move(1)
#     r.v[0] = 0.
#     bodies.append(r)
#     joints.append(Joint(r, None, [50, 275]))
#     joints.append(Joint(r, None, [950, 325]))
#     # r = Rect([300, 100], [25, 500], mass=10000)
#     # bodies.append(r)
#
#     world = World(bodies, joints, dt=DT)
#     return world, c, target


# def make_world(learned_force):
#     bodies = []
#     joints = []
#
#     # c = Circle([100, 259], 30)
#     # bodies.append(c)
#     # c.add_force(ExternalForce(learned_force))
#     # c.add_force(ExternalForce(multiplier=10))
#     c = Rect([100, 259], (60, 60))
#     bodies.append(c)
#     c.add_force(ExternalForce(learned_force))
#     c.add_force(ExternalForce(multiplier=1))
#
#     target = Circle([500, 259], 30)
#     bodies.append(target)
#     target.add_no_collision(c)
#
#     r = Rect([500, 300], [1000, 20])
#     bodies.append(r)
#     joints.append(Joint(r, None, [50, 275]))
#     joints.append(Joint(r, None, [950, 325]))
#     # r = Rect([300, 100], [25, 500], mass=10000)
#     # bodies.append(r)
#
#     world = World(bodies, joints, dt=DT)
#     return world, c, target


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-nd':
        # Run without displaying
        screen = None
    else:
        width, height = 1000, 600
        screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        screen.set_alpha(None)
        pygame.display.set_caption('2D Engine')

    grad_demo(screen)
