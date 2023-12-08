import copy
from abc import ABC, abstractmethod
from gymnasium.spaces import Box, Dict
import numpy as np
from envs.utils.interfaces import StateInterface


class Task(ABC):

    @abstractmethod
    def update_space(self, obs_dict, specs):
        pass

    @abstractmethod
    def init(self, specs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_task_obs(self):
        pass

    @abstractmethod
    def distance(self, env: StateInterface, info, done):
        pass

class DummyTask(Task):

    def update_space(self, obs_dict, specs):
        return obs_dict

    def init(self, specs):
        pass

    def reset(self):
        pass

    def get_task_obs(self):
        return {}

    def distance(self, env: StateInterface, info, done):
        return 0


class PuckDirectionTask(Task):
    def __init__(self):
        self.puck_dir_space = None
        self.task = None

    def init(self, specs):
        self.puck_dir_space = Box(low=np.array([0, -1]), high=np.ones(2, ))
        self.task = None

    def update_space(self, obs_space, specs):
        obs_space['puck_dir'] = self.puck_dir_space

        return Dict(obs_space)

    def reset(self):
        self.task = self.puck_dir_space.sample()

    def distance(self, env, info, done):
        state = env.get_state()
        puck_vel = state['puck_vel'][:2]

        puck_vel = puck_vel / np.linalg.norm(puck_vel)
        curr = np.array([np.cos(puck_vel), np.sin(puck_vel)])

        return np.linalg.norm(self.task - curr)

    def get_task_obs(self):
        return {'puck_dir': self.task}


class PuckPositionVelocity(Task):
    def __init__(self):
        self.space = None
        self.key_pos = 'g_puck_pos'
        self.key_vel = 'g_puck_vel'

        self.puck_pos_space = None
        self.puck_vel_space = None
        self.task = None
        self.scale = lambda x: x # default is no scale

    def init(self, specs):
        if specs.scale_obs:
            min_puck_pos = specs.min_dict['puck_pos']
            puck_pos_range = specs.range_dict['puck_pos']

            min_puck_vel = specs.min_dict['puck_vel']
            puck_vel_range = specs.range_dict['puck_vel']

            self.scale = lambda task: {
                self.key_pos: (task[self.key_pos] - min_puck_pos) / puck_pos_range * 2 - 1,
                self.key_vel: (task[self.key_vel] - min_puck_vel) / puck_vel_range * 2 - 1
            }

    def update_space(self, obs_space: dict, specs):
        self.puck_pos_space = obs_space['puck_pos']
        self.puck_vel_space = obs_space['puck_vel']

        obs_space[self.key_pos] = self.puck_pos_space
        obs_space[self.key_vel] = self.puck_vel_space

        return Dict(obs_space)

    def reset(self):
        self.task = {
            self.key_pos: self.puck_pos_space.sample(),
            self.key_vel: self.puck_vel_space.sample()
        }

        return copy.deepcopy(self.task)

    def get_task_obs(self):
        return copy.deepcopy(self.scale(self.task))

    def distance(self, env: StateInterface, info, done):
        state = env.get_state()
        puck_pos = state['puck_pos'][:2]
        puck_vel = state['puck_vel'][:2]

        distance = np.linalg.norm(np.array([self.task[self.key_pos] - puck_pos, self.task[self.key_vel] - puck_vel]))
        return distance


