from my_scripts.utils import variant_util, load_variant
from envs.env_maker import make_environment
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import os
import numpy as np


class ConstAgent():
    def __init__(self, action):
        self.action = action

    def predict(self, observation, **kwargs):
        return self.action, None


def launch(path, num_episodes, always_action=None):
    env_args, alg_args, learn_args, log_args, variant = variant_util(load_variant(path))

    env_args['render'] = True
    env = make_environment(**env_args)

    if always_action is None:
        agent = PPO.load(os.path.join(path, 'best_model'))
    else:
        agent = ConstAgent(always_action)

    episode_reward = []
    actions = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        cum_reward = 0

        while not done:
            action, _ = agent.predict(observation=obs, deterministic=True)
            actions.append(action)
            obs, rew, done, _, _ = env.step(action)
            steps += 1
            cum_reward += rew

        episode_reward.append(cum_reward)

    episode_reward = np.array(episode_reward)
    print(f'Return {np.mean(episode_reward)} +- {2 * np.std(episode_reward) / np.sqrt(episode_reward.shape[0])}')

    print('Actions Stats:')
    actions = np.array(actions)
    for i in range(len(env.unwrapped.policies)):
        print(f'Policy {i}: {len(actions[actions == i]) / len(actions)}')


def eval_agent(path, num_episodes, parallel, always_action=None):
    env_args, alg_args, learn_args, log_args, variant = variant_util(load_variant(path))
    env_producer = lambda: make_environment(**env_args)
    env = VecMonitor(make_vec_env(env_producer, n_envs=parallel, vec_env_cls=SubprocVecEnv))
    if always_action is None:
        agent = PPO.load(os.path.join(path, 'best_model'))
    else:
        agent = ConstAgent(2)
    print(evaluate_policy(agent, env, n_eval_episodes=num_episodes))


if __name__ == '__main__':
    path = '/home/thomas/Downloads/876767'

    launch(path, num_episodes=20, always_action=1)
    #eval_agent(path, 1, 1, always_action=1)
