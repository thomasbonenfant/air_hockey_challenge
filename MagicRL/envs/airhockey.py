from MagicRL.envs import BaseEnv
from envs.airhockeydoublewrapper import AirHockeyDouble
import numpy as np


class AirHockey(BaseEnv):
    def sample_state(self, args: dict = None):
        raise NotImplementedError

    def __init__(self, render=False, **kwargs):
        super().__init__(**kwargs)

        self.env = AirHockeyDouble(opponent_delay=0)
        self.state = None
        self.time = 0
        self.render = render
        self.score = np.array([0, 0])
        self.faults = np.array([0, 0])

        self.reward_coef = 1000
        self.fault_coef = 1000



    def reset(self) -> None:
        self.score = np.array([0, 0])
        self.faults = np.array([0, 0])
        self.time = 0
        self.state = self.env.reset()

    def reward(self, info):
        reward_score = (np.array(info["score"]) - self.score) @ np.array([1, -1]) * self.reward_coef
        reward_fault = -(np.array(info["faults"]) - self.faults) @ np.array([1, 0]) * self.fault_coef

        if reward_score + reward_fault != 0:
            print(f'REWARD: {reward_score + reward_fault}')

        return reward_score + reward_fault

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if self.render:
            self.env.render()

        self.time += 1
        self.state = obs

        rew = self.reward(info)

        # update score and faults count
        self.score = np.array(info['score'])
        self.faults = np.array(info['faults'])

        return obs, rew, done, info

