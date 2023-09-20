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
    large_reward = []
    fault_penalty = []
    constr_penalty= []
    fault_risk_penalty = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        cumulative_reward = 0

        cumulative_large_reward= 0
        cumulative_fault_penalty = 0
        cumulative_fault_risk_penalty = 0
        cumulative_constr_penalty = 0

        while not done:
            action, _ = agent.predict(observation=obs, deterministic=True)
            actions.append(action)
            obs, rew, done, _, info = env.step(action)
            steps += 1
            cumulative_reward += rew

            cumulative_large_reward += info['reward']['log_large_reward']
            cumulative_fault_penalty += info['reward']['log_fault_penalty']
            cumulative_fault_risk_penalty += info['reward']['log_fault_risk_penalty']
            cumulative_constr_penalty += info['reward']['log_constr_penalty']

        episode_reward.append(cumulative_reward)
        large_reward.append(cumulative_large_reward)
        fault_penalty.append(cumulative_fault_penalty)
        constr_penalty.append(cumulative_constr_penalty)
        fault_risk_penalty.append(cumulative_fault_risk_penalty)

    episode_reward = np.array(episode_reward)
    print(f'Average Reward {np.mean(episode_reward)} +- {2 * np.std(episode_reward) / np.sqrt(episode_reward.shape[0])}')

    print(f'Average Large Reward: {np.mean(episode_reward)}\t {np.mean(episode_reward) / np.mean(episode_reward)}')
    print(f'Average Fault Penalty: {np.mean(fault_penalty)}\t {np.mean(fault_penalty) / np.mean(episode_reward)}')
    print(f'Average Constr Penalty: {np.mean(constr_penalty)}\t {np.mean(constr_penalty) / np.mean(episode_reward)}')
    print(f'Fault Risk Penalty: {np.mean(fault_risk_penalty)}\t {np.mean(fault_risk_penalty) / np.mean(episode_reward)}')

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
    path = '/home/thomas/Downloads/463593'
    path = '/home/thomas/Downloads/299758'
    path = '/home/thomas/Downloads/172658'

    launch(path, num_episodes=1, always_action=None)
    #eval_agent(path, 10, 2, always_action=None)
