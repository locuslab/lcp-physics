import gym
import numpy as np

import torch
from torch.nn import MSELoss
from torch.optim import RMSprop

from lcp_physics.physics.utils import plot, get_tensor

from encoder import extract_params, make_world
from run_breakout import simulate_breakout, to_vector, get_action, ACTION_VAL, run_breakout


def infer_params():
    env = gym.make('BreakoutNoFrameskip-v0')
    num_steps = 10
    mini_batch = 1

    learning_rate = 0.03
    max_iter = 1000

    paddle_vel_x = torch.randn(1)
    paddle_vel_y = torch.randn(1)

    infered_params = [
        paddle_vel_x,
        paddle_vel_y,
    ]
    infered_params = [get_tensor([p], requires_grad=True) for p in infered_params]
    optim = RMSprop(infered_params, lr=learning_rate)

    loss_hist = []
    rew_hist = []
    paddle_vel_hist = []
    last_loss = 1e10
    for i in range(max_iter):
        true_states_batch = []
        states_batch = []
        for _ in range(mini_batch):
            s = env.reset()
            initial_state = extract_params(s)
            true_states = [to_vector(*initial_state)[:4]]

            actions = []
            for _ in range(num_steps):
                a = np.sign(np.random.randn()) * ACTION_VAL
                actions.append(a)
                a = get_action([a])
                s = env.step(a)
                true_states.append(to_vector(*extract_params(s[0]))[:4])

            true_states = torch.cat([s.unsqueeze(0) for s in true_states])
            true_states_batch.append(true_states.view(1, -1))

            world, ball, paddle, blocks, paddle_idx = make_world(*initial_state, [0, 0])
            states = [true_states[0].unsqueeze(0)]
            for j in range(num_steps):
                s = simulate_breakout(actions[j], world, ball, paddle, blocks, paddle_idx,
                                      paddle_vel_x=infered_params[0],
                                      paddle_vel_y=infered_params[1],
                                      )
                states.append(s[:, :4])
            states = torch.cat(states)
            states_batch.append(states.view(1, -1))

        optim.zero_grad()
        true_states_batch = torch.cat(true_states_batch)
        states_batch = torch.cat(states_batch)
        loss = MSELoss()(states_batch, true_states_batch)
        loss.backward()
        grad = [p.grad.item() if p.grad is not None else 0 for p in infered_params]

        optim.step()

        print('Iteration:', i, '/', max_iter)
        print('Loss:', loss.item())
        print('Gradient:', grad)
        print('Params:', [p.item() for p in infered_params])
        paddle_vel_hist.append([p.detach().clone() for p in infered_params])

        # XXX Uncomment to test while training
        # num_episodes = 1
        # paddle_vel = [p.item() for p in infered_params]
        # run_breakout(paddle_vel=paddle_vel, exp_name='learn_params_{}'.format(i),
        #              num_episodes=num_episodes, verbose=1)
        print('-----')

        if abs((last_loss - loss).item()) < 1e-15:
            break
        last_loss = loss
        loss_hist.append(loss.item())

    print('paddle vel hist')
    plot(paddle_vel_hist)
    print('loss hist')
    plot(loss_hist)
    print(torch.cat([states, true_states], dim=1))

    return infered_params


if __name__ == '__main__':
    params = infer_params()
    print(params)
