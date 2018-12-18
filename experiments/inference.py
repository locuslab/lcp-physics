import sys
import time

import pygame

import torch
from torch.nn import MSELoss

from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.bodies import Circle, Rect
from lcp_physics.physics.forces import ExternalForce, Gravity
from lcp_physics.physics.constraints import Joint
from lcp_physics.physics.utils import Recorder, plot, Defaults


TIME = 40
DT = Defaults.DT
DTYPE = Defaults.DTYPE

STOP_DIFF = 1e-3
MASS_EPS = 1e-7
TOTAL_MASS = 7
NUM_LINKS = 10


def main(screen):
    forces = [hor_impulse]
    ground_truth_mass = torch.tensor([TOTAL_MASS], dtype=DTYPE)
    world, chain = make_world(forces, ground_truth_mass, num_links=NUM_LINKS)

    rec = None
    # rec = Recorder(DT, screen)
    ground_truth_pos = positions_run_world(world, run_time=10, screen=screen, recorder=rec)
    ground_truth_pos = [p.data for p in ground_truth_pos]
    ground_truth_pos = torch.cat(ground_truth_pos)

    learning_rate = 0.5
    max_iter = 100

    next_mass = torch.rand_like(ground_truth_mass, requires_grad=True)
    print('\rInitial mass:', next_mass.item())
    print('-----')

    optim = torch.optim.RMSprop([next_mass], lr=learning_rate)
    loss_hist = []
    mass_hist = [next_mass.item()]
    last_loss = 1e10
    for i in range(max_iter):
        if i % 1 == 0:
            world, chain = make_world(forces, next_mass.clone().detach(), num_links=NUM_LINKS)
            run_world(world, run_time=10, print_time=False, screen=None, recorder=None)

        world, chain = make_world(forces, next_mass, num_links=NUM_LINKS)
        positions = positions_run_world(world, run_time=10, screen=None)
        positions = torch.cat(positions)
        positions = positions[:len(ground_truth_pos)]
        clipped_ground_truth_pos = ground_truth_pos[:len(positions)]

        optim.zero_grad()
        loss = MSELoss()(positions, clipped_ground_truth_pos)
        loss.backward()

        optim.step()

        print('Iteration: {} / {}'.format(i+1, max_iter))
        print('Loss:', loss.item())
        print('Gradient:', next_mass.grad.item())
        print('Next mass:', next_mass.item())
        print('-----')
        if abs((last_loss - loss).item()) < STOP_DIFF:
            print('Loss changed by less than {} between iterations, stopping training.'
                  .format(STOP_DIFF))
            break
        last_loss = loss
        loss_hist.append(loss.item())
        mass_hist.append(next_mass.item())

    world = make_world(forces, next_mass, num_links=NUM_LINKS)[0]
    rec = None
    positions = positions_run_world(world, run_time=10, screen=screen, recorder=rec)
    positions = torch.cat(positions)
    positions = positions[:len(ground_truth_pos)]
    clipped_ground_truth_pos = ground_truth_pos[:len(positions)]
    loss = MSELoss()(positions, clipped_ground_truth_pos)
    print('Final loss:', loss.item())
    print('Final mass:', next_mass.item())

    plot(loss_hist)
    plot(mass_hist)


def hor_impulse(t):
    if t < 0.1:
        return ExternalForce.RIGHT.type(DTYPE)
    else:
        return ExternalForce.ZEROS.type(DTYPE)


def make_world(forces, mass, num_links=10):
    bodies = []
    joints = []

    # make chain of rectangles
    link_mass = mass / num_links
    r = Rect([300, 50], [20, 60], mass=link_mass)
    bodies.append(r)
    joints.append(Joint(r, None, [300, 30]))
    for i in range(1, num_links):
        if i < num_links - 1:
            r = Rect([300, 50 + 50 * i], [20, 60], mass=link_mass)
        else:
            r = Rect([300, 50 + 50 * i], [20, 60], mass=link_mass)
        r.add_force(Gravity(g=100))
        bodies.append(r)
        joints.append(Joint(bodies[-1], bodies[-2], [300, 25 + 50 * i]))
        bodies[-1].add_no_contact(bodies[-2])

    # make projectile
    m = 3
    c_pos = torch.tensor([50, bodies[-1].pos[1]])  # same Y as last chain link
    c = Circle(c_pos, 20, restitution=1.)
    bodies.append(c)
    for f in forces:
        c.add_force(ExternalForce(f, multiplier=500 * m))

    world = World(bodies, joints, dt=DT, post_stab=True)
    return world, r


def positions_run_world(world, dt=Defaults.DT, run_time=10,
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

                if not recorder:
                    # Don't refresh screen if recording
                    pygame.display.update(update_list)
                else:
                    recorder.record(world.t)

            elapsed_time = time.time() - start_time
            # print('\r ', '{} / {}  {} '.format(int(world.t), int(elapsed_time),
            #                                   1 / animation_dt), end='')
    return positions


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '-nd':
        # Run without displaying
        screen = None
    else:
        width, height = 1000, 600
        screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        screen.set_alpha(None)
        pygame.display.set_caption('Chain Mass Inference')

    main(screen)
