import os
import argparse
import numpy as np
import torch
import gym
from gym import wrappers

from model import GaussianPolicy
from utils import grad_false


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='BipedalWalkerHardcore-v3')
    parser.add_argument('--log_name', type=str, default='sac-seed0-datetime')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    log_dir = os.path.join('logs', args.env_id, args.log_name)

    env = gym.make(args.env_id)
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    policy = GaussianPolicy(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        hidden_units=[512, 256]).to(device)

    checkp = torch.load('./BWH-SAC/policy.pth', map_location='cpu')
    policy.load_state_dict(checkp)
    grad_false(policy)

    def exploit(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, action = policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    reward_sum = 0.0
    num_runs = 50
    for i in range(num_runs):
        state = env.reset()
        episode_reward = 0.
        done = False

        while not done:
            # env.render()
            action = exploit(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        reward_sum += episode_reward
        print('Episode %s: %s' % (i+1, episode_reward))

    print('\nAverage Reward: ', (reward_sum /float(num_runs)))


if __name__ == '__main__':
    run()
