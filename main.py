import os
import argparse
from datetime import datetime
import gym

from agent import SacAgent


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='BipedalWalkerHardcore-v3')
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # You can define configs in the external json or yaml file.
    configs = {
        'num_episodes': args.num_episodes,
        'batch_size': 256,
        'lr': 0.0004,
        'hidden_units': [512, 256],
        'memory_size': 1e6,
        'gamma': 0.99,
        'tau': 0.005,
        'entropy_tuning': True,
        'ent_coef': 0.2,
        'multi_step': 1,
        'per': False,  # prioritized experience replay
        'alpha': 0.6,
        'beta': 0.4,
        'beta_annealing': 0.0001,
        'grad_clip': None,
        'updates_per_step': 1,
        'start_steps': 10000,
        'log_interval': 10,
        'target_update_interval': 1,
        'eval_interval': 20000,
        'cuda': True,
        'seed': args.seed
    }

    env = gym.make(args.env_id)

    log_dir = os.path.join(
        'logs', args.env_id,
        f'sac-seed{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}')

    agent = SacAgent(env=env, log_dir=log_dir, **configs)
    agent.run()


if __name__ == '__main__':
    run()
