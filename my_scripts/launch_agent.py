from my_scripts.utils import variant_util, load_variant
from envs.env_maker import create_producer
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


class RandomAgent():
    def __init__(self, action_space):
        self.ac = action_space

    def predict(self, observation, **kwargs):
        return self.ac.sample(), None


def launch(path, num_episodes, random=False, always_action=None, best=False, store_traj=False, seed=None, custom_env_args=None, action_dict=None):
    env_args, alg_args, learn_args, log_args, variant = variant_util(load_variant(path))

    if custom_env_args is not None:
        for key in custom_env_args:
            env_args[key] = custom_env_args[key]

    env = create_producer(env_args)()

    if always_action is None and not random:
        print(f'Loading {"best" if best else ""} agent at {path}')
        agent = PPO.load(os.path.join(path, 'best_model' if best else 'model'))
    elif random:
        print(f'Using Random Agent')
        agent = RandomAgent(env.action_space)
    else:
        print(f'Always selecting action: {always_action}')
        agent = ConstAgent(always_action)

    if seed is not None:
        np.random.seed(seed)

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

            if action_dict is not None:
                print(f'Action: {action_dict[int(action)]}')

            obs, rew, done, _, info = env.step(action)
            print(rew)
            env.render()
            steps += 1
            cumulative_reward += rew

            #cumulative_large_reward += info['large_reward']
            #cumulative_fault_penalty += info['fault_penalty']
            #cumulative_fault_risk_penalty += info['fault_risk_penalty']
            #cumulative_constr_penalty += info['constr_penalty']

        episode_reward.append(cumulative_reward)
        large_reward.append(cumulative_large_reward)
        fault_penalty.append(cumulative_fault_penalty)
        constr_penalty.append(cumulative_constr_penalty)
        fault_risk_penalty.append(cumulative_fault_risk_penalty)

    episode_reward = np.array(episode_reward)
    fault_penalty = np.array(fault_penalty)
    constr_penalty = np.array(constr_penalty)
    fault_risk_penalty = np.array(fault_risk_penalty)
    print(f'Average Reward {np.mean(episode_reward)} +- {2 * np.std(episode_reward) / np.sqrt(episode_reward.shape[0])}')

    #print(f'Average Large Reward: {np.mean(episode_reward)}\t {np.std(episode_reward) / np.sqrt(episode_reward.shape[0])}')
    #print(f'Average Fault Penalty: {np.mean(fault_penalty)}\t {np.std(fault_penalty) / np.sqrt(fault_penalty.shape[0])}')
    #print(f'Average Constr Penalty: {np.mean(constr_penalty)}\t {np.std(constr_penalty) / np.sqrt(constr_penalty.shape[0])}')
    #rint(f'Fault Risk Penalty: {np.mean(fault_risk_penalty)}\t {np.std(fault_risk_penalty) / np.sqrt(fault_risk_penalty.shape[0])}')

    #print('Actions Stats:')
    #actions = np.array(actions)
    #for i in range(len(env.unwrapped.policies)):
    #    print(f'Policy {i}: {len(actions[actions == i]) / len(actions)}')


def eval_agent(path, num_episodes, parallel, always_action=None, render=False):
    env_args, alg_args, learn_args, log_args, variant = variant_util(load_variant(path))
    env_producer = create_producer(env_args)
    env = VecMonitor(make_vec_env(env_producer, n_envs=parallel, vec_env_cls=SubprocVecEnv))
    if always_action is None:
        agent = PPO.load(os.path.join(path, 'best_model'))
    else:
        agent = ConstAgent(2)
    print(evaluate_policy(agent, env, n_eval_episodes=num_episodes, render=render))


if __name__ == '__main__':
    path = '/home/thomas/Downloads/463593'
    path = '/home/thomas/Downloads/299758'
    path = '/home/thomas/Downloads/172658'
    path = '/home/thomas/Downloads/38050'
    path = '/home/thomas/Downloads/246278'
    path = '/home/thomas/Downloads/127497'
    #path = 'models/ppo/rb_hit+defend_oac+repel_oac+prepare_rb+home_rb/541699'

    custom_env_args = {
    }

    action_dict = {
        0: 'Hit',
        1: 'Defend',
        2: 'Repel',
        3: 'Prepare',
        4: 'Home',
    }

    launch(path,
           num_episodes=10,
           random=False,
           always_action=None,
           best=True,
           store_traj=False,
           seed=666,
           custom_env_args=custom_env_args,
           action_dict=None)
    #eval_agent(path, num_episodes=10, parallel=1, render=True)
