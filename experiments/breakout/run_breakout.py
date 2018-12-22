import os
import time
from datetime import datetime

import gym
import numpy as np

import torch
from torch import nn

from lcp_physics.physics.utils import get_tensor

from encoder import extract_params, make_world, step_breakout, DT

from mpc.mpc import GradMethods, QuadCost, MPC


BALL_VEL = (25, 25)
ACTION_VAL = 1.0
PADDLE_VEL_ADJUST = -1 / DT / ACTION_VAL

LAMBDA_A = 1e-20

CUDA = False


def simulate_breakout(a, world, ball, paddle, blocks, paddle_idx,
                      paddle_vel_x=None, paddle_vel_y=None):
    step_breakout(world, paddle_idx, a, paddle_vel_x=paddle_vel_x, paddle_vel_y=paddle_vel_y)
    ball_params = torch.cat([ball.pos, ball.rad.view(1)])
    paddle_params = torch.cat([paddle.pos, paddle.dims])
    blocks_params = torch.cat([torch.cat([b.pos, b.dims]) for b in blocks])
    sn = torch.cat([paddle_params, blocks_params, ball_params])
    sn = sn.unsqueeze(0)

    new_paddle_params, new_blocks_params, new_ball_params = from_vector(sn.squeeze(0))
    paddle.p = torch.cat([get_tensor([0]), new_paddle_params[0]])
    paddle.dims = new_paddle_params[1]
    ball.p = torch.cat([get_tensor([0]), new_ball_params[0]])
    ball.rad = new_ball_params[1]
    for i in range(len(new_blocks_params)):
        blocks[i].p = torch.cat([get_tensor([0]), new_blocks_params[i][0]])
        blocks[i].dims = new_blocks_params[i][1]
    return sn


def to_vector(paddle_params, blocks_params, ball_params):
    ball = torch.cat([get_tensor(ball_params[0]),
                      get_tensor([ball_params[1]])])
    paddle = torch.cat([get_tensor(p) for p in paddle_params])
    if blocks_params:
        blocks = torch.cat([torch.cat([get_tensor(p) for p in block])
                            for block in blocks_params])
    else:
        # hack: placeholder block outside screen for when there are no blocks
        blocks = get_tensor([1000, 60, 1, 1])
    ret = torch.cat([paddle, blocks, ball])
    if CUDA:
        ret = ret.cuda()
    return ret


def from_vector(v):
    paddle_params = [v[:2], v[2:4]]
    blocks_params = []
    for i in range(4, len(v[:-3]), 4):
        blocks_params.append([v[i:i+2], v[i+2:i+4]])
    ball_params = [v[-3:-1], v[-1]]
    return paddle_params, blocks_params, ball_params


def mpc(s1, a, T, ball_vel=BALL_VEL, lambda_a=1e-3, centering=False,
        paddle_vel=None, vert_cost=False, verbose=1):
    ns = len(s1)
    na = len(a[0])

    class BreakoutDynamics(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, u):
            x_dim = x.ndimension()
            if x_dim == 1:
                x = x.unsqueeze(0)
            params = from_vector(x[0])
            world, ball, paddle, blocks, paddle_idx = make_world(*params, ball_vel)

            next_x = simulate_breakout(u[0], world, ball, paddle, blocks, paddle_idx,
                                       paddle_vel_x=paddle_vel[0], paddle_vel_y=paddle_vel[1])
            return next_x

    dynamics = BreakoutDynamics()
    if CUDA:
        dynamics = dynamics.cuda()
    u_lower, u_upper = -ACTION_VAL, ACTION_VAL
    x_init = s1.clone()
    u_init = get_tensor(a).clone()
    if CUDA:
        u_init = u_init.cuda()

    ball_vel_y = ball_vel[1]
    ball_pos_y = s1[-2]
    paddle_pos_y = s1[1]

    s_Q = torch.zeros(ns)

    Q = torch.cat([s_Q, torch.ones(na) * lambda_a]).type_as(s1).diag().unsqueeze(0).repeat(T, 1, 1)
    if vert_cost:
        Q[:, ns-2, ns-2] = 10
    # Simple X tracking
    if ball_vel_y > 0:
        frames_to_paddle = (paddle_pos_y - ball_pos_y) / ball_vel_y / DT
        for t in range(T):
            if int(frames_to_paddle) + 1 >= t:
                Q[t, ns - 3, ns - 3] = 1
                Q[t, ns - 3, 0] = -1
                Q[t, 0, 0] = 1
                Q[t, 0, ns - 3] = -1
    else:
        if centering and ball_pos_y > 1000:
            Q[:, ns - 3, ns - 3] = 1
            Q[:, ns - 3, 0] = -1
            Q[:, 0, 0] = 1
            Q[:, 0, ns - 3] = -1

    p = torch.zeros(ns + na).type_as(Q)
    p = p.unsqueeze(0).repeat(T, 1)

    Q = Q.unsqueeze(1)
    p = p.unsqueeze(1)

    x_init = x_init.unsqueeze(0)
    solver = MPC(
        ns, na, T=T,
        # x_init=x_init,
        u_init=u_init,
        u_lower=u_lower, u_upper=u_upper,
        verbose=verbose,
        delta_u=10 * ACTION_VAL,
        lqr_iter=1,
        grad_method=GradMethods.AUTO_DIFF,
        n_batch=1,
        max_linesearch_iter=1,
        exit_unconverged=False,
        backprop=False,
    )
    if CUDA:
        solver = solver.cuda()
    cost = QuadCost(Q, p)
    x, u, objs = solver(x_init, cost, dynamics)
    u = u.squeeze(1)
    if CUDA:
        u = u.cpu()
    return u.data.numpy()


def get_action(a, threshold=0.9):
    condition = np.abs(a[0]) >= ACTION_VAL * threshold

    if condition:
        action = int((np.sign(a[0]) + 1) / 2 + 2)
    else:
        action = 0
    return action


def run_breakout(T=2, record=False, threshold=0.99, paddle_vel=None, centering=True,
                 exp_name='', render=True, num_episodes=1, verbose=2):
    exp_name = '_' + exp_name if exp_name else exp_name
    paddle_vel = [v * PADDLE_VEL_ADJUST for v in paddle_vel]

    env = gym.make('BreakoutNoFrameskip-v0')
    env.reset()
    if record:
        timestamp = str(datetime.now()).replace(' ', '_') + exp_name
        env = gym.wrappers.Monitor(env, os.path.join('videos', 'gym', timestamp),
                                   video_callable=lambda episode_id: True)

    na = 1
    output_interval = 10
    maxSteps = 10000000000
    rews = []
    for j in range(num_episodes):
        a = np.array([0.01 * np.random.randn(na) for _ in range(T)])
        env.reset()
        # repeat "start game" action, to make sure its received correctly
        env.step(1)
        env.step(1)
        env.step(1)
        prev_s = to_vector(*extract_params(env.step(1)[0]))
        s = env.step(1)[0]

        rew = 0.
        cur_lives = 5
        for t in range(maxSteps):
            try:
                s = extract_params(s)
                if verbose > 1:
                    if t % output_interval == 0:
                        print('+ Step {}'.format(t))
                    print(s)
                s = to_vector(*s)
                ball_vel = (s[-3:-1] - prev_s[-3:-1]) / DT
                if verbose > 1:
                    print(ball_vel.numpy(), torch.norm(ball_vel))

                a = mpc(s, a, T, ball_vel=ball_vel, lambda_a=LAMBDA_A, paddle_vel=paddle_vel,
                        centering=centering, verbose=verbose-2)
                action = get_action(a, threshold=threshold)
                if verbose > 1:
                    print(action, a[:, 0])
                prev_s = s
                s, rew_t, stop, info = env.step(action)
                if render:
                    env.render()
                if stop:
                    if verbose > 0:
                        print('Episode done.')
                    break
                rew += rew_t
                if verbose > 1:
                    print('Score: {} Lives: {}'.format(rew, info['ale.lives']))
                a = np.concatenate((a[1:], 0.001 * np.random.randn(na)[np.newaxis]))
                if info['ale.lives'] < cur_lives:
                    # dropped the ball
                    cur_lives = info['ale.lives']
                    if verbose > 1:
                        print('===== {} lives remaining ====='. format(cur_lives))
                    a = np.array([0.01 * np.random.randn(na) for _ in range(T)])
                    env.step(1)
                    env.step(1)
                    env.step(1)
                    prev_s = to_vector(*extract_params(env.step(1)[0]))
                    s = env.step(1)[0]
            except KeyboardInterrupt:
                break
        if verbose > 1:
            print(rew, j)
        rews.append(rew)
    if verbose > 1:
        print(rews)
    avg_rew = sum(rews) / num_episodes
    if verbose > 0:
        print('Average reward: {} (std. dev. {})'.format(avg_rew, np.std(rews)))
    return avg_rew


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', '-ne', type=int, default=1)
    parser.add_argument('--T', '-T', type=int, default=2)
    parser.add_argument('--threshold', '-t', type=float, default=0.99)
    parser.add_argument('--paddle-vel', '-pv', type=float, nargs=2, default=[3.74, 0.])
    parser.add_argument('--record', '-r', action='store_true')
    parser.add_argument('--no-display', '-nd', action='store_true')
    parser.add_argument('--exp-name', '-en', type=str, default='')
    parser.add_argument('--no-centering', '-nc', action='store_true')
    args = parser.parse_args()

    run_breakout(T=args.T, record=args.record, threshold=args.threshold, paddle_vel=args.paddle_vel,
                 centering=not args.no_centering, exp_name=args.exp_name, render=not args.no_display,
                 num_episodes=args.num_episodes)
