from envs.air_hockey_option import AirHockeyOption
from envs.utils import Task
from typing import Callable
from copy import deepcopy

from gymnasium.spaces import Dict, flatten_space, flatten

from envs.utils import StateInterface
import numpy as np


class AirHockeyOptionTask(AirHockeyOption, StateInterface):
    def __init__(self, reward_her: Callable = None, **kwargs):
        super().__init__(**kwargs)
        self.task = None

        self.reward_HER = None

        if reward_her is not None:
            self.reward_HER = lambda achieved, desired, info: reward_her(self, achieved, desired, info)
            self.compute_reward = lambda a, d, i: self.reward_HER(a, d, i)[0] # reward_HER returns also a modified info

        self.observation_space = Dict({'observation': flatten_space(self.observation_space)}) # for HER


    def set_task(self, task: Task):
        self.task = task
        self.task.init(self.specs)

        self.observation_space = self.task.update_space(self.observation_space, self.specs)

        self.task.reset()

    def step(self, action):
        s, r, d, t, i = super().step(action)
        d = d or self.task.done_condition(self)

        # for HER use copmute_reward interface and ignore old env reward
        if self.reward_HER is not None:
            r, i = self.reward_HER(s['achieved_goal'], s['desired_goal'], i)
        return s, r, d, t, i

    def reset(self, seed=0, options=None):
        self.task.reset()
        return super().reset(seed, options)

    def process_state(self, state, info):
        state = super().process_state(state, info)

        # Converts state for HER compatibility
        state = {
            'observation': flatten(self.specs.observation_space, state),
             'desired_goal': self.task.get_desired_goal(),
             'achieved_goal': self.task.get_achieved_goal(self)
        }

        return state

    def render(self, render_mode='human'):
        self.task.render(self)
        super().render(render_mode=render_mode)

    def get_state(self):
        res = {
            'ee_pos': self.ee_pos,
            'ee_vel': self.ee_vel,
            'puck_pos': self.puck_pos[:2],
            'puck_vel': self.puck_vel,
            'has_hit': self.has_hit
        }
        return deepcopy(res)
