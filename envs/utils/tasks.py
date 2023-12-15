import copy
from abc import ABC, abstractmethod
from gymnasium.spaces import Box, Dict
import numpy as np
from envs.utils.interfaces import StateInterface

from air_hockey_challenge.utils.transformations import robot_to_world


def terminate_after_hit(env):
    return env.has_hit


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

    def render(self, env):
        pass

    @abstractmethod
    def distance(self, env: StateInterface, info, done):
        pass

    def done_condition(self, env):
        return False


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

class DummyDefendTask(DummyTask):
    def done_condition(self, env):
        return env.puck_pos[0] >= 1.51 and env.puck_vel[0] > 0

class PuckDirectionTask(Task):
    def __init__(self):
        self.puck_dir_space = None
        self.task = None
        self.reset_flag = True

    def init(self, specs):
        # self.puck_dir_space = Box(low=np.array([0, -1]), high=np.ones((2,)))
        self.puck_dir_space = Box(low=-np.pi / 2, high=np.pi / 2)
        self.task = None

    def update_space(self, obs_space, specs):
        obs_space['puck_dir'] = self.puck_dir_space

        return Dict(obs_space)

    def reset(self):
        self.task = self.puck_dir_space.sample()
        self.reset_flag = True

    def render(self, env):
        cos = np.cos(self.task).squeeze()
        sin = np.sin(self.task).squeeze()

        site = env.env.base_env._data.site("direction")

        puck_pos = env.get_state()['puck_pos'].copy()
        puck_pos[0] -= 1.51
        puck_pos = np.hstack([puck_pos, 0])

        current_rotation_matrix = site.xmat.reshape(3, 3)

        rotation = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        center_rotated = rotation @ np.hstack([0.25, 0, 1])
        center_rotated[2] = 0.03

        site.xmat = np.dot(rotation, current_rotation_matrix).flatten()
        site.xpos = center_rotated
        site.xpos[0] += puck_pos[0]
        site.xpos[1] += puck_pos[1]

    def distance(self, env, info, done):
        state = env.get_state()
        puck_vel = state['puck_vel'][:2]

        theta = np.arctan2(puck_vel[1], puck_vel[0])

        delta = np.abs(self.task.squeeze() - theta)

        if delta > np.pi:
            delta = 2 * np.pi - delta

        return delta

    def get_task_obs(self):
        return {'puck_dir': self.task}

    def done_condition(self, env):
        return terminate_after_hit(env)


class PuckPositionTask(Task):
    def reset(self):
        self.task = self.puck_pos_space.sample()
        self.task[0] /= 2

        assert self.task[0] <= 1.51  # check we do not set an opponent's position as puck goal

    def get_task_obs(self):
        return self.scale({self.key: self.task})

    def distance(self, env: StateInterface, info, done):
        state = env.get_state()

        puck_pos = state['puck_pos'][:2]

        return np.linalg.norm(self.task - puck_pos)

    def render(self, env):
        task_2_render = self.task.copy()
        task_2_render[0] -= 1.51

        env.env.base_env._data.site("goal_vis").xpos = np.hstack([task_2_render, 0])

    def __init__(self):
        self.puck_pos_space = None
        self.task = None
        self.key = 'g_puck_pos'
        self.scale = lambda x: x  # default is no scale

    def init(self, specs):
        if specs.scale_obs:
            min_puck_pos = specs.min_dict['puck_pos']
            puck_pos_range = specs.range_dict['puck_pos']

            self.scale = lambda task: {
                self.key: (task[self.key] - min_puck_pos) / puck_pos_range * 2 - 1,
            }

    def update_space(self, obs_space, specs):
        self.puck_pos_space = obs_space['puck_pos']
        obs_space[self.key] = self.puck_pos_space

        return Dict(obs_space)

    def done_condition(self, env):
        return self.distance(env, {}, False) < 0.2


class PuckPositionVelocity(Task):
    def __init__(self):
        self.space = None
        self.key_pos = 'g_puck_pos'
        self.key_vel = 'g_puck_vel'

        self.puck_pos_space = None
        self.puck_vel_space = None
        self.task = None
        self.scale = lambda x: x  # default is no scale

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


class PuckDirectionDefendTask(PuckDirectionTask):
    def done_condition(self, env):
        state = env.get_state()

        puck_vel = state['puck_vel'][:2]
        puck_pos = state['puck_pos']

        w_puck_pos, _ = robot_to_world(env.specs.robot_frame[0], puck_pos)

        # if hits the border or makes goal
        if w_puck_pos[0] < -env.specs.table_length / 2 + env.specs.mallet_radius:
            return True

        return (np.linalg.norm(puck_vel) < 0.1 and self.distance(env, {}, False) < np.pi / 6) or \
            (puck_vel[0] > 0 and puck_pos[0] >= 1.51)
