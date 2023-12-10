from envs.air_hockey_option import AirHockeyOption
from envs.utils import Task
from typing import Callable
from copy import deepcopy

from envs.utils import StateInterface


class AirHockeyOptionTask(AirHockeyOption, StateInterface):
    def __init__(self, reward_fn: Callable, **kwargs):
        super().__init__(**kwargs)
        self.task = None
        self.reward = lambda info, done : reward_fn(self, info, done)

    def set_task(self, task: Task):
        self.task = task
        self.task.init(self.specs)

        self.observation_space = self.task.update_space(self.observation_space, self.specs)

        self.task.reset()

    def step(self, action):
        s, r, d, t, i = super().step(action)
        d = d or self.task.done_condition(self)
        return s, r, d, t, i

    def reset(self, seed=0, options=None):
        self.task.reset()
        return super().reset(seed, options)

    def process_state(self, state, info):
        state = super().process_state(state, info)
        state.update(self.task.get_task_obs())
        return state

    def get_state(self):
        res = {
            'ee_pos': self.ee_pos,
            'ee_vel': self.ee_vel,
            'puck_pos': self.puck_pos,
            'puck_vel': self.puck_vel
        }
        return deepcopy(res)

