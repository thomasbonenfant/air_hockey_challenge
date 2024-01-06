import copy
from abc import ABC, abstractmethod
from gymnasium.spaces import Box, Dict, Discrete, flatten_space, flatten, unflatten
import numpy as np
from envs.utils.interfaces import StateInterface
from collections import defaultdict
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
    def get_desired_goal(self):
        pass

    @abstractmethod
    def get_achieved_goal(self, env: StateInterface):
        pass

    def render(self, env):
        pass

    def unflatten(self, flattened_goal):
        return flattened_goal

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

    def get_desired_goal(self):
        return []

    def get_achieved_goal(self, env):
        return []

    def distance(self, env: StateInterface, info, done):
        return 0


class DummyDefendTask(DummyTask):
    def done_condition(self, env):
        return env.puck_pos[0] >= 1.51 and env.puck_vel[0] > 0


class PuckDirectionTask(Task):
    def __init__(self):
        self.puck_dir_space = None
        self.hit_space = None
        self.goal_space = None
        self.task = None
        self.reset_flag = True

    def init(self, specs):
        self.puck_dir_space = Box(low=-np.pi / 2, high=np.pi / 2)
        self.hit_space = Discrete(2, start=0)
        self.goal_space = Dict({
            'puck_dir': self.puck_dir_space,
            'has_hit': self.hit_space
        })
        self.task = None

    def update_space(self, obs_space, specs):
        obs_space['desired_goal'] = flatten_space(self.goal_space)
        obs_space['achieved_goal'] = obs_space['desired_goal']

        return Dict(obs_space)

    def reset(self):
        self.task = self.goal_space.sample()
        self.task['has_hit'] = True # force task where I have to hit the puck
        self.reset_flag = True

    def render(self, env):
        cos = np.cos(self.task['puck_dir']).squeeze()
        sin = np.sin(self.task['puck_dir']).squeeze()

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

        delta = np.abs(self.task['puck_dir'].squeeze() - theta)

        if delta > np.pi:
            delta = 2 * np.pi - delta

        return delta

    def get_desired_goal(self):
        return flatten(self.goal_space, self.task)

    def get_achieved_goal(self, env):
        state = env.get_state()

        has_hit = state['has_hit']
        puck_vel = state['puck_vel'][:2]
        theta = np.arctan2(puck_vel[1], puck_vel[0])

        achieved_goal = {
            'has_hit': has_hit,
            'puck_dir': theta
        }

        return flatten(self.goal_space, achieved_goal)

    def unflatten(self, flattened_goal):
        return unflatten(self.goal_space, flattened_goal)

    def done_condition(self, env):
        return terminate_after_hit(env)


class PuckPositionTask(Task):
    def __init__(self):
        self.puck_pos_space = None
        self.task = None

    def init(self, specs):
        puck_radius = specs.puck_radius
        table_length = specs.table_length
        table_width = specs.table_width
        low_puck_pos = np.array([-table_length/2 + puck_radius + 1.51, -table_width/2 + puck_radius])
        high_puck_pos = np.array([0 - puck_radius + 1.51, table_width/2 - puck_radius])
        self.puck_pos_space = Box(low_puck_pos, high_puck_pos)

    def reset(self):
        self.task = self.puck_pos_space.sample()

        assert self.task[0] <= 1.51  # check we do not set an opponent's position as puck goal

    def get_desired_goal(self):
        return self.task

    def distance(self, env: StateInterface, info, done):
        state = env.get_state()

        puck_pos = state['puck_pos'][:2]

        return np.linalg.norm(self.task - puck_pos)

    def render(self, env):
        task_2_render = self.task.copy()
        task_2_render[0] -= 1.51

        env.env.base_env._data.site("goal_vis").xpos = np.hstack([task_2_render, 0])

    def update_space(self, obs_space, specs):
        obs_space['achieved_goal'] = self.puck_pos_space
        obs_space['desired_goal'] = self.puck_pos_space

        return Dict(obs_space)

    def done_condition(self, env):
        return self.distance(env, {}, False) < 0.0633

    def get_achieved_goal(self, env: StateInterface):
        return env.get_state().get('puck_pos')


