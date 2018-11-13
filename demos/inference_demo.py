import sys
import time

import pygame

import torch
from torch.nn import MSELoss
from torch.autograd import Variable

from lcp_physics.physics.world import World
from lcp_physics.physics.bodies import Circle, Rect
from lcp_physics.physics.forces import ExternalForce, Gravity, vert_impulse, hor_impulse, rot_impulse
from lcp_physics.physics.constraints import Joint
from lcp_physics.physics.utils import Recorder, plot, Params

TIME = 40
MASS_EPS = 1e-7
DT = Params.DEFAULT_DT
NUM_LINKS = 10


def inference_demo(screen):
    forces = [hor_impulse]
    ground_truth_mass = Variable(torch.DoubleTensor([7]))
    world, c = make_world(forces, ground_truth_mass, num_links=NUM_LINKS)

    rec = None
    # rec = Recorder(DT, screen)
    ground_truth_pos = positions_run_world(world, run_time=10, screen=screen, recorder=rec)
    ground_truth_pos = [p.data for p in ground_truth_pos]
    ground_truth_pos = Variable(torch.cat(ground_truth_pos))

    learning_rate = 0.01
    max_iter = 100

    next_mass = Variable(torch.DoubleTensor([1.3]), requires_grad=True)
    loss_hist = []
    mass_hist = [next_mass]
    last_dist = 1e10
    for i in range(max_iter):
        world, c = make_world(forces, next_mass, num_links=NUM_LINKS)
        # world.load_state(initial_state)
        # world.reset_engine()
        positions = positions_run_world(world, run_time=10, screen=None)
        positions = torch.cat(positions)
        positions = positions[:len(ground_truth_pos)]
        # temp_ground_truth_pos = ground_truth_pos[:len(positions)]

        loss = MSELoss()(positions, ground_truth_pos)
        loss.backward()
        grad = c.mass.grad.data
        # clip gradient
        grad = torch.max(torch.min(grad, torch.DoubleTensor([100])), torch.DoubleTensor([-100]))
        temp = c.mass.data - learning_rate * grad
        temp = max(MASS_EPS, temp[0])
        next_mass = Variable(torch.DoubleTensor([temp]), requires_grad=True)
        # learning_rate /= 1.1
        print(i, '/', max_iter, loss.data[0])
        print(grad)
        print(next_mass)
        # print(learned_force(0.05))
        if abs((last_dist - loss).data[0]) < 1e-3:
            break
        last_dist = loss
        loss_hist.append(loss)
        mass_hist.append(next_mass)

    world = make_world(forces, next_mass, num_links=NUM_LINKS)[0]
    # world.load_state(initial_state)
    # world.reset_engine()
    rec = None
    # rec = Recorder(DT, screen)
    positions_run_world(world, run_time=10, screen=screen, recorder=rec)
    loss = MSELoss()(positions, ground_truth_pos)
    print(loss.data[0])
    print(next_mass)

    plot(loss_hist)
    plot(mass_hist)


def make_world(forces, mass, num_links=10):
    bodies = []
    joints = []

    # make chain of rectangles
    r = Rect([300, 50], [20, 60])
    bodies.append(r)
    joints.append(Joint(r, None, [300, 30]))
    for i in range(1, num_links):
        if i < num_links - 1:
            r = Rect([300, 50 + 50 * i], [20, 60])
        else:
            r = Rect([300, 50 + 50 * i], [20, 60], mass=mass)
        bodies.append(r)
        joints.append(Joint(bodies[-1], bodies[-2], [300, 25 + 50 * i]))
        bodies[-1].add_no_collision(bodies[-2])
    bodies[-1].add_force(Gravity(g=100))

    # make projectile
    m = 13
    c1 = Circle([50, bodies[-1].pos.data[1]], 20)  # same Y as last chain link
    bodies.append(c1)
    for f in forces:
        c1.add_force(ExternalForce(f, multiplier=100 * m))

    world = World(bodies, joints, dt=DT)
    return world, r


def positions_run_world(world, dt=Params.DEFAULT_DT, run_time=10,
                        screen=None, recorder=None):
    positions = [torch.cat([b.p for b in world.bodies])]

    if screen is not None:
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

        animation_dt = dt
        elapsed_time = 0.
        prev_frame_time = -animation_dt
        start_time = time.time()

    while world.t < run_time:
        world.step()
        positions.append(torch.cat([b.p for b in world.bodies]))

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

                # # XXX visualize collision points and normal for debug
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
                else:
                    recorder.record(world.t)

            elapsed_time = time.time() - start_time
            print('\r ', '{} / {}  {} '.format(int(world.t), int(elapsed_time),
                                              1 / animation_dt), end='')
    return positions


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-nd':
        # Run without displaying
        screen = None
    else:
        width, height = 1000, 600
        screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        screen.set_alpha(None)
        pygame.display.set_caption('2D Engine')
    screen = None

    inference_demo(screen)
