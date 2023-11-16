from envs.env_maker import create_producer
from air_hockey_agent.utils.sb3_variant_util import get_configuration
from my_scripts.experiment2 import alg_dict
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


def launch(path, num_episodes, random=False, always_action=None, best=False, store_traj=False, seed=None, custom_env_args=None, action_dict=None, render=False):
    env_args, alg = get_configuration(path)

    if custom_env_args is not None:
        for key in custom_env_args:
            env_args[key] = custom_env_args[key]

    env = create_producer(env_args)()

    alg_cls = alg_dict[alg]

    if always_action is None and not random:
        print(f'Loading {"best" if best else ""} agent at {path}')
        path = (os.path.join(path, 'best_model' if best else 'model'))
        agent = alg_cls.load(path)
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

    for episode in range(num_episodes):
        print(f'Episode: {episode}/{num_episodes}')
        obs, _ = env.reset()
        done = False
        steps = 0
        cumulative_reward = 0

        while not done:
            action, _ = agent.predict(observation=obs, deterministic=True)
            actions.append(action)

            if action_dict is not None:
                print(f'Action: {action_dict[int(action)]}')

            obs, rew, done, _, info = env.step(action)
            #print(rew)
            if render:
                env.render()
            steps += 1
            cumulative_reward += rew

        episode_reward.append(cumulative_reward)

    episode_reward = np.array(episode_reward)

    print(f'Average Reward {np.mean(episode_reward)} +- {2 * np.std(episode_reward) / np.sqrt(episode_reward.shape[0])}')


if __name__ == '__main__':
    path = '/home/thomas/Downloads/markov/data/'
    subpath = 'oac_7dof-hit/sac/old_env_alpha100/442439'
    #subpath = 'goal/sac/goal_30__ee/290727'
    subpath = 'hit/sac/sde_alpha100/204552'
    #subpath = 'hit/sac/clip_acceleration_6dof/802676'
    subpath = 'hit/sac/clip_acceleration_aim_6dof/461264'
    subpath = 'goal/sac/goal_30__ee__ee_vel/31108'
    #subpath = 'hrl/ppo/without_opponent/82530'

    path = os.path.join(path, subpath)

    #path = 'models/ppo/rb_hit+defend_oac+repel_oac+prepare_rb+home_rb/541699'

    custom_env_args = {

    }

    action_dict = {
        0: 'Repel',
        1: 'Home',
        2: 'Hit',
        3: 'Prepare',
    }

    launch(path,
           num_episodes=20,
           random=False,
           always_action=None,
           best=False,
           store_traj=False,
           seed=None,
           custom_env_args=custom_env_args,
           action_dict=None,
           render=True)
