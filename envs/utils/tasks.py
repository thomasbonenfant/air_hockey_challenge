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
    def __init__(self, include_achieved=True):
        self.include_achieved = include_achieved

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
    def __init__(self, include_hit_flag=False, **kwargs):
        super().__init__(**kwargs)
        self.puck_dir_space = None
        self.hit_space = None
        self.goal_space = None
        self.task = None
        self.reset_flag = True
        self.include_hit_flag = include_hit_flag

    def init(self, specs):
        self.puck_dir_space = Box(low=-np.pi / 2, high=np.pi / 2)
        self.hit_space = Discrete(2, start=0)

        if self.include_hit_flag:
            self.goal_space = Dict({
                'puck_dir': self.puck_dir_space,
                'has_hit': self.hit_space
            })
        else:
            self.goal_space = Dict({
                'puck_dir': self.puck_dir_space
            })

        self.task = None

    def update_space(self, obs_space, specs):
        obs_space['desired_goal'] = flatten_space(self.goal_space)

        if self.include_achieved:
            obs_space['achieved_goal'] = obs_space['desired_goal']

        return Dict(obs_space)

    def reset(self):
        self.task = self.goal_space.sample()
        if self.include_hit_flag:
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
            'puck_dir': theta
        }

        if self.include_hit_flag:
            achieved_goal.update({'has_hit': has_hit})

        return flatten(self.goal_space, achieved_goal)

    def unflatten(self, flattened_goal):
        return unflatten(self.goal_space, flattened_goal)

    def done_condition(self, env):
        return terminate_after_hit(env)


class PuckPositionTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.puck_pos_space = None
        self.task = None
        self.scale_obs = False
        self.min_space = None
        self.range_space = None

    def init(self, specs):
        '''puck_radius = specs.puck_radius
        table_length = specs.table_length
        table_width = specs.table_width
        low_puck_pos = np.array([-table_length/2 + puck_radius + 1.51, -table_width/2 + puck_radius])
        high_puck_pos = np.array([0 - puck_radius + 1.51, table_width/2 - puck_radius])
        self.puck_pos_space = Box(low_puck_pos, high_puck_pos)'''
        self.scale_obs = specs.scale_obs
        self.puck_pos_space = specs.observation_space['puck_pos']
        self.min_space = specs.min_dict.get('puck_pos')
        self.range_space = specs.range_dict.get('puck_pos')


    def reset(self):
        self.task = self.puck_pos_space.sample()

        if self.scale_obs and self.task[0] > 0:
            self.task[0] -= 0.5
        elif not self.scale_obs and self.task[0] > 1.51:
            self.task[0] = -self.task[0] + 2 * 1.51


    def get_desired_goal(self):
        return self.task

    def distance(self, env: StateInterface, info, done):
        state = env.get_state()

        puck_pos = state['puck_pos'][:2]
        task = self.task

        if self.scale_obs:
            task = (self.task + 1) * self.range_space / 2 + self.min_space # unscaled task

        return np.linalg.norm(task - puck_pos)

    def render(self, env):
        task_2_render = self.task.copy()
        if self.scale_obs:
            task_2_render = (self.task + 1) * self.range_space / 2 + self.min_space # unscaled task
        task_2_render[0] -= 1.51

        env.env.base_env._data.site("goal_vis").xpos = np.hstack([task_2_render, 0])

    def update_space(self, obs_space, specs):
        if self.include_achieved:
            obs_space['achieved_goal'] = self.puck_pos_space
        obs_space['desired_goal'] = self.puck_pos_space

        return Dict(obs_space)

    def done_condition(self, env):
        threshold = 0.1
        vel_threshold = 0.1
        if self.scale_obs:
            threshold /= 0.5 * env.specs.table_diag
        return self.distance(env, {}, False) < threshold and np.linalg.norm(env.puck_vel[:2]) < 0.1

    def get_achieved_goal(self, env: StateInterface):
        achieved_goal = env.get_state().get('puck_pos')
        if self.scale_obs:
            achieved_goal = (achieved_goal - self.min_space) / self.range_space * 2 - 1
        return achieved_goal


