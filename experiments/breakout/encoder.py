import time

import pygame
import numpy as np

import torch

from lcp_physics.physics.bodies import Rect, Circle
from lcp_physics.physics.world import World, run_world
from lcp_physics.physics.constraints import TotalConstraint, YConstraint, RotConstraint
from lcp_physics.physics.forces import ExternalForce
from lcp_physics.physics.utils import Defaults, get_tensor

# Utilities
X, Y = 1, 0

# Atari parameters
ATARI_RED = (200, 72, 72)
ATARI_ORANGE = (198, 108, 58)
ATARI_BEIGE = (180, 122, 48)
ATARI_YELLOW = (162, 162, 42)
ATARI_GREEN = (72, 160, 72)
ATARI_BLUE = (66, 72, 200)
ATARI_CYAN = (66, 158, 130)
ATARI_GRAY = (142, 142, 142)
ATARI_BLACK = (0, 0, 0)

SCREEN_HEIGHT = 210
SCREEN_WIDTH = 160

PADDLE_TOP_LINE = 189
PADDLE_BOTTOM_LINE = 192
PADDLE_HEIGHT = 4
PADDLE_WIDTH = 16
PADDLE_COLOR = ATARI_RED

BALL_HEIGHT = 4
BALL_WIDHT = 2
BALL_COLOR = ATARI_RED

BLOCK_HEIGHT = 6
BLOCKS_TOP_LEFT = (57, 8)
BLOCKS_BOTTOM_RIGHT = (92, 151)
BLOCK_COLORS = (ATARI_RED, ATARI_ORANGE, ATARI_BEIGE, ATARI_YELLOW, ATARI_GREEN, ATARI_BLUE)

WALL_WIDTH = 8
WALL_COLOR = ATARI_GRAY
ROOF_LINE = 31
STOPPER_LINE = 189
STOPPER_WIDTH = WALL_WIDTH
STOPPER_HEIGHT = 6
LEFT_STOPPER_COLOR = ATARI_CYAN
RIGHT_STOPPER_COLOR = ATARI_RED

# Physics parameters
DT = 1
FRIC_COEFF = 0
BALL_MASS = 1
STD_RESTITUTION = 1
STOPPER_RESTITUTION = -STD_RESTITUTION


def extract_params(frame):
    frame = np.copy(frame)
    # get paddle position
    # print('========')
    paddle_x = np.argmax(frame[PADDLE_TOP_LINE, :, 0])
    # print(paddle_x)
    if frame[PADDLE_TOP_LINE, paddle_x + BALL_WIDHT + 1, 0] != PADDLE_COLOR[0]:
        # print('Mod to')
        paddle_x = np.argmax(frame[PADDLE_TOP_LINE, paddle_x + BALL_WIDHT + 1:, 0]) + paddle_x + BALL_WIDHT + 1
    paddle_top_left = np.array([PADDLE_TOP_LINE, paddle_x])
    # print(paddle_top_left)

    right = np.argmin(frame[PADDLE_TOP_LINE, paddle_x:, 0] == PADDLE_COLOR[0]) + paddle_x
    # print(right)
    right = right if right != paddle_x else SCREEN_WIDTH
    right = min(right, paddle_x + PADDLE_WIDTH)
    # print(right)
    bottom = PADDLE_TOP_LINE + PADDLE_HEIGHT
    paddle_bottom_right = np.array([bottom, right])
    paddle_corners = [paddle_top_left, paddle_bottom_right]

    # print(paddle_bottom_right)
    # print(PADDLE_TOP_LINE)
    # print('=======')

    # remove paddle from frame
    remove_rect(frame, paddle_top_left, paddle_bottom_right)

    # get blocks positions
    blocks_corners = []
    for i, color in enumerate(BLOCK_COLORS):
        top_line = BLOCKS_TOP_LEFT[Y] + i * BLOCK_HEIGHT
        left = np.argmax(frame[top_line, :, 0] == color[0])
        while left > 0:
            right = np.argmin(frame[top_line, left:, 0] == color[0]) + left
            pos = [np.array([top_line, left]), np.array([top_line + BLOCK_HEIGHT, right])]
            blocks_corners.append(pos)
            remove_rect(frame, pos[0], pos[1] - [1, 1])
            left = np.argmax(frame[top_line, :, 0] == color[0])

    # remove red stopper
    remove_rect(frame, [STOPPER_LINE, SCREEN_WIDTH - STOPPER_WIDTH],
                [STOPPER_LINE + STOPPER_HEIGHT - 1, SCREEN_WIDTH])
    # get ball position and dims
    ball_corners = None
    i = 0
    found_ball = False
    while not found_ball and i < SCREEN_HEIGHT:
        j = 0
        while not found_ball and j < SCREEN_WIDTH:
            if frame[i, j, 0] == BALL_COLOR[0]:
                top_left = np.array([i, j])
                right = np.argmin(frame[i, j:, 0] == BALL_COLOR[0]) + j
                bottom = np.argmin(frame[i:, j, 0] == BALL_COLOR[0]) + i
                ball_corners = ([top_left,
                                 np.array([bottom, right])])
                found_ball = True
            j += 1
        i += 1

    if ball_corners is not None:
        ball_pos, ball_dims = get_pos_dims(*ball_corners)
        ball_params = [ball_pos, min(np.min(ball_dims) / 2, 1)]
    else:
        # ball_params = [[SCREEN_WIDTH / 2, SCREEN_HEIGHT], 1.0]
        # place ball far off when it falls off screen to incur high cost
        ball_params = [[SCREEN_WIDTH / 2, SCREEN_HEIGHT * 100], 1.0]
    paddle_params = get_pos_dims(*paddle_corners)
    blocks_params = [get_pos_dims(*b) for b in blocks_corners]

    return paddle_params, blocks_params, ball_params


def remove_rect(frame, top_left, bottom_right):
    frame[top_left[Y]:bottom_right[Y] + 1, top_left[X]:bottom_right[X] + 1, :] = 0


def get_pos_dims(top_left, bottom_right):
    pos = [(bottom_right[X] + top_left[X]) / 2.0,
           (bottom_right[Y] + top_left[Y]) / 2.0]
    dims = [float(bottom_right[X] - top_left[X]),
            float(bottom_right[Y] - top_left[Y])]
    return pos, dims


def make_world(paddle_params, blocks_params, ball_params, ball_vel):
    bodies = []
    constraints = []

    # Add ball if it exists
    ball_pos, ball_rad = ball_params
    ball_body = Circle(ball_pos, ball_rad, vel=ball_vel, mass=BALL_MASS,
                       restitution=STD_RESTITUTION, fric_coeff=FRIC_COEFF,
                       col=BALL_COLOR, thickness=0)
    bodies.append(ball_body)

    # Add paddle
    paddle_pos, paddle_dims = paddle_params
    paddle_body = Rect(paddle_pos, paddle_dims, restitution=STD_RESTITUTION,
                       fric_coeff=FRIC_COEFF, col=PADDLE_COLOR, thickness=0)
    bodies.append(paddle_body)
    constraints += [
        # YConstraint(paddle_body),
        # RotConstraint(paddle_body)
    ]
    paddle_idx = len(bodies) - 1

    # Add walls
    left_wall = Rect([WALL_WIDTH / 2.0, STOPPER_LINE / 2.0], [WALL_WIDTH, STOPPER_LINE],
                     restitution=STD_RESTITUTION, fric_coeff=FRIC_COEFF,
                     col=WALL_COLOR, thickness=0)
    constraints.append(TotalConstraint(left_wall))
    left_wall.add_no_contact(paddle_body)
    bodies.append(left_wall)
    right_wall = Rect([SCREEN_WIDTH - WALL_WIDTH / 2.0, (STOPPER_LINE + STOPPER_HEIGHT) / 2.0],
                      [WALL_WIDTH, STOPPER_LINE + STOPPER_HEIGHT], restitution=STD_RESTITUTION,
                      fric_coeff=FRIC_COEFF, col=WALL_COLOR, thickness=0)
    constraints.append(TotalConstraint(right_wall))
    paddle_body.add_no_contact(right_wall)
    bodies.append(right_wall)
    roof = Rect([SCREEN_WIDTH / 2.0, ROOF_LINE / 2.0], [SCREEN_WIDTH - 2 * WALL_WIDTH, ROOF_LINE],
                restitution=STD_RESTITUTION, fric_coeff=FRIC_COEFF,
                col=WALL_COLOR, thickness=0)
    roof.add_no_contact(paddle_body)
    roof.add_no_contact(left_wall)
    roof.add_no_contact(right_wall)
    constraints.append(TotalConstraint(roof))
    bodies.append(roof)

    # Add stoppers
    left_stopper = Rect([STOPPER_WIDTH / 2.0, STOPPER_LINE + STOPPER_HEIGHT / 2.0],
                        [STOPPER_WIDTH, STOPPER_HEIGHT], restitution=STOPPER_RESTITUTION,
                        fric_coeff=FRIC_COEFF, col=LEFT_STOPPER_COLOR, thickness=0)
    constraints.append(TotalConstraint(left_stopper))
    bodies.append(left_stopper)
    left_stopper.add_no_contact(left_wall)
    left_stopper.add_no_contact(ball_body)

    right_stopper = Rect([SCREEN_WIDTH + STOPPER_WIDTH / 2.0, STOPPER_LINE + STOPPER_HEIGHT / 2.0],
                         [STOPPER_WIDTH, STOPPER_HEIGHT], restitution=STOPPER_RESTITUTION,
                         fric_coeff=FRIC_COEFF, col=WALL_COLOR, thickness=0)
    constraints.append(TotalConstraint(right_stopper))
    bodies.append(right_stopper)
    right_stopper.add_no_contact(right_wall)
    right_stopper.add_no_contact(ball_body)

    # Add blocks
    blocks_bodies = []
    for block_params in blocks_params:
        block_pos, block_dims = block_params
        import math  # XXX
        level = int((block_pos[1] - block_dims[1] / 2 - BLOCKS_TOP_LEFT[0]) / BLOCK_HEIGHT)
        # TODO Define restitution by level? (different levels have different bounces)
        block_restitution = 1
        block_body = Rect(block_pos, block_dims, restitution=block_restitution,
                          fric_coeff=FRIC_COEFF, col=BLOCK_COLORS[level], thickness=0)
        bodies.append(block_body)
        blocks_bodies.append(block_body)
        constraints.append(TotalConstraint(block_body))
        block_body.add_no_contact(left_wall)
        block_body.add_no_contact(right_wall)
        for other_block in blocks_bodies[:-1]:
            block_body.add_no_contact(other_block)

    return World(bodies, constraints, dt=DT), ball_body, paddle_body, blocks_bodies, paddle_idx


def run_breakout(world, paddle_idx, actions, paddle_vel=None,
                 dt=0.1, run_time=10, screen=None, recorder=None):
    """Helper function to run a simulation forward once a world is created.
    """
    if screen is not None:
        import pygame
        background = pygame.Surface(screen.get_size())
        background = background.convert()
        background.fill((255, 255, 255))

    elapsed_time = 0.
    prev_frame_time = -dt
    start_time = time.time()

    while world.t < run_time:
        a = -0.1
        if len(actions) > 0:
            a = actions[0]
            actions = actions[1:]
        v = world.v.clone()
        # paddle_vel = 0
        # if a == 0:
        #     paddle_vel = paddle_vel
        # elif a == 1:
        #     paddle_vel = -paddle_vel
        v[paddle_idx * 3 + 1] = paddle_vel * a
        world.set_v(get_tensor(v))
        world.step(fixed_dt=True)

        # action = 0
        # if len(actions) > 0:
        #     action = actions[0]
        #     actions = actions[1:]
        # step_breakout(world, paddle_idx, action, paddle_vel)

        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            if elapsed_time - prev_frame_time >= dt or recorder:
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
            if not recorder:
                # Adjust frame rate dynamically to keep real time
                wait_time = dt - (elapsed_time - prev_frame_time)
                if wait_time >= 0 and not recorder:
                    wait_time += dt  # XXX
                    time.sleep(max(wait_time - dt, 0))

        elapsed_time = time.time() - start_time
        # print('\r ', '{} / {}  {} '.format(int(world.t), int(elapsed_time),
        #                                    1 / dt), end='')


def step_breakout(world, paddle_idx, action, paddle_vel_x=None, paddle_vel_y=None):
    v = world.v.detach().clone()
    paddle_act_x = paddle_vel_x * action
    v[paddle_idx * 3 + 1] = paddle_act_x
    paddle_act_y = paddle_vel_y * action
    v[paddle_idx * 3 + 2] = paddle_act_y
    world.set_v(v)

    world.step(fixed_dt=True)


if __name__ == '__main__':
    # Debugging
    import gym
    env = gym.make('BreakoutNoFrameskip-v0')
    env.reset()
    frames = []
    env.step(1)
    # for _ in range(100):
    #     env.step(3)
    # plt.imshow(env.step(0)[0])
    # plt.show()
    frames.append(env.step(3)[0])
    # frame0[87:93, 20:47, :] = 0
    # frame0[87:93, 80:107, :] = 0
    frames.append(env.step(3)[0])
    # frame1[87:93, 20:47, :] = 0
    # frame1[87:93, 80:107, :] = 0
    frames.append(env.step(3)[0])
    frames[-1][50:100, 8:SCREEN_WIDTH-8, :] = 0

    paddle0, blocks0, ball0 = extract_params(frames[0])
    paddle1, blocks1, ball1 = extract_params(frames[-1])
    vel = (np.array(ball1[0]) - np.array(ball0[0])) / DT / (len(frames) - 1)
    print(paddle1)
    print(blocks1)
    print(ball1)
    print(vel)

    pygame.init()
    width, height = SCREEN_WIDTH, SCREEN_HEIGHT
    screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
    screen.set_alpha(None)

    world, _, _, _, paddle_idx = make_world(paddle1, blocks1, ball1, vel)
    actions = [-1] * 3
    run_breakout(world, paddle_idx, actions, paddle_vel=3.5, screen=screen, dt=DT)
