import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, flatten_space
from itertools import islice


class Specification(object):
    def __init__(self, env_info, include_joints, include_ee, include_ee_vel, include_opponent, joint_acc_clip, scale_obs,
                 max_path_len, scale_action, remove_last_joint, include_puck, alpha_r, stop_after_hit, include_hit_flag):
        self._robot_model = env_info['robot']['robot_model']
        self._robot_data = env_info['robot']['robot_data']
        self._robot_frame = env_info['robot']['base_frame']
        self._puck_radius = env_info["puck"]["radius"]
        self._mallet_radius = env_info["mallet"]["radius"]
        self._dt = env_info['dt']

        self._table_length = env_info['table']['length']
        self._table_width = env_info['table']['width']
        self._table_diag = np.sqrt(self._table_length ** 2 + self._table_width ** 2)
        self._goal_width = env_info['table']['goal_width']

        self.puck_pos_ids = self.f_puck_pos_ids = env_info['puck_pos_ids']
        self.puck_vel_ids = self.f_puck_vel_ids = env_info['puck_vel_ids']
        self.opponent_ee_ids = self.f_opponent_ee_ids = env_info['opponent_ee_ids']

        self.joint_pos_ids = self.f_joint_pos_ids = env_info["joint_pos_ids"]
        self.joint_vel_ids = self.f_joint_vel_ids =  env_info["joint_vel_ids"]

        obs_space = env_info['rl_info'].observation_space
        low_state = obs_space.low
        high_state = obs_space.high

        low_position = np.array([0.54, -0.5, 0])
        high_position = np.array([1.5, 0.5, 0.3])
        ee_pos_norm = high_position - low_position

        self._max_vel = 5 #obs_space.high[self.puck_vel_ids][0]

        self.include_ee = include_ee
        self.include_ee_vel = include_ee_vel
        self.include_joints = include_joints
        self.include_puck = include_puck
        self.include_opponent = include_opponent
        self.remove_last_joint = remove_last_joint
        self.alpha_r = alpha_r
        self.joint_acc_clip = np.array(joint_acc_clip)
        self.max_path_len = max_path_len
        self.stop_after_hit = stop_after_hit
        self.scale_obs = scale_obs
        self.scale_action = scale_action
        self.include_hit_flag = include_hit_flag

        if self.remove_last_joint:
            self.f_joint_pos_ids = self.joint_pos_ids[:6]
            self.f_joint_vel_ids = self.joint_vel_ids[:6]

        # remove theta and z
        self.f_puck_pos_ids = self.puck_pos_ids[:2]
        self.f_puck_vel_ids = self.puck_vel_ids[:2]
        self.f_opponent_ee_ids = [i for i in islice(self.opponent_ee_ids, 2)]

        if not self.include_opponent:
            self.f_opponent_ee_ids = []

        '''if not self.include_puck:
            self.puck_pos_ids = []
            self.puck_vel_ids = []

        if not self.include_joints:
            self.joint_pos_ids = []
            self.joint_vel_ids = []'''

        if not self.remove_last_joint:
            self.f_joint_pos_ids = [i for i in islice(self.f_joint_pos_ids, 6)]
            self.f_joint_vel_ids = [i for i in islice(self.f_joint_vel_ids, 6)]

        low_joints_pos = obs_space.low[self.f_joint_pos_ids]
        high_joints_pos = obs_space.high[self.f_joint_pos_ids]

        low_joints_vel = obs_space.low[self.f_joint_vel_ids]
        high_joints_vel = obs_space.high[self.f_joint_vel_ids]

        #low_puck_pos = low_state[self.f_puck_pos_ids]
        #high_puck_pos = high_state[self.f_puck_pos_ids]
        low_puck_pos = np.array([-self._table_length / 2 + self._puck_radius + 1.51, -self._table_width / 2 + self._puck_radius])
        high_puck_pos = np.array([0 - self._puck_radius + 1.51, self._table_width / 2 - self._puck_radius])
        low_puck_vel = low_state[self.f_puck_vel_ids]
        high_puck_vel = high_state[self.f_puck_vel_ids]

        low_opponent_ee = low_state[self.f_opponent_ee_ids]
        high_opponent_ee = high_state[self.f_opponent_ee_ids]

        joint_vel_norm = high_joints_vel - low_joints_vel
        joint_pos_norm = high_joints_pos - low_joints_pos

        self._constr_scales = {
            'joint_pos_constr': np.concatenate([joint_pos_norm, joint_pos_norm]),
            'joint_vel_constr': np.concatenate([joint_vel_norm, joint_vel_norm]),
            'ee_constr': np.concatenate([ee_pos_norm[:2], ee_pos_norm[:2], ee_pos_norm[:2]])[:5]
        }

        # Set observation Space and Action Space

        obs_dict = {}
        self.min_dict = {}
        self.range_dict = {}

        if self.include_puck:
            if self.scale_obs:
                self.min_dict['puck_pos'] = low_puck_pos
                self.range_dict['puck_pos'] = high_puck_pos - low_puck_pos

                self.min_dict['puck_vel'] = low_puck_vel
                self.range_dict['puck_vel'] = high_puck_vel - low_puck_vel

                puck_pos_obs = Box(-np.ones(2,), np.ones(2,))
                puck_vel_obs = Box(-np.ones(2,), np.ones(2,))
            else:
                puck_pos_obs = Box(low_puck_pos, high_puck_pos)
                puck_vel_obs = Box(low_puck_vel, high_puck_vel)
            obs_dict['puck_pos'] = puck_pos_obs
            obs_dict['puck_vel'] = puck_vel_obs

        if self.include_joints:
            if self.scale_obs:
                self.min_dict['joint_pos'] = low_joints_pos
                self.range_dict['joint_pos'] = high_joints_pos - low_joints_pos

                self.min_dict['joint_vel'] = low_joints_vel
                self.range_dict['joint_vel'] = high_joints_vel - low_joints_vel

                joint_pos_obs = Box(-np.ones_like(low_joints_pos), np.ones_like(high_joints_pos))
                joint_vel_obs = Box(-np.ones_like(low_joints_vel), np.ones_like(high_joints_pos))
            else:
                joint_pos_obs = Box(low_joints_pos, high_joints_pos)
                joint_vel_obs = Box(low_joints_vel, high_joints_vel)
            obs_dict['joint_pos'] = joint_pos_obs
            obs_dict['joint_vel'] = joint_vel_obs

        if self.include_opponent and len(self.opponent_ee_ids) > 0:
            if self.scale_obs:
                self.min_dict['opponent_ee'] = low_opponent_ee
                self.range_dict['opponent_ee'] = high_opponent_ee - low_opponent_ee
                self.opponent_ee_obs = Box(-np.ones_like(low_opponent_ee), np.ones_like(high_opponent_ee))
            else:
                self.opponent_ee_obs = Box(low_opponent_ee, high_opponent_ee)
            obs_dict['opponent_ee'] = self.opponent_ee_obs

        if self.include_ee:
            if self.scale_obs:
                self._min_obs_ee_pos = low_position[:2]
                self.obs_ee_pos_range = high_position[:2] - low_position[:2]
                ee_obs = Box(-np.ones_like(low_position[:2]), np.ones_like(high_position[:2]))
            else:
                ee_obs = Box(low_position[:2], high_position[:2])
            obs_dict['ee_pos'] = ee_obs

        if self.include_ee_vel:
            if self.scale_obs:
                self._min_obs_ee_vel = -self.max_vel
                self.obs_ee_vel_range = 2 * self.max_vel
                ee_vel = Box(-np.ones((2,)), np.ones((2,)))
            else:
                ee_vel = Box(-self.max_vel * np.ones((2,)), self.max_vel * np.ones((2,)))
            obs_dict['ee_vel'] = ee_vel

        if self.include_hit_flag:
            obs_dict['has_hit'] = Discrete(2, start=0)

        self.observation_space = Dict(obs_dict)

        low_action = env_info['robot']['joint_acc_limit'][0]
        high_action = env_info['robot']['joint_acc_limit'][1]

        if self.joint_acc_clip is not None:
            low_action = np.clip(low_action, -self.joint_acc_clip, 0)
            high_action = np.clip(high_action, 0, self.joint_acc_clip)

            if self.remove_last_joint:
                low_action = low_action[:6]
                high_action = high_action[:6]

        if self.scale_action:
            self._min_ac = low_action
            self.ac_range = high_action - low_action
            self.action_space = Box(low=-np.ones_like(low_action), high=np.ones_like(high_action))
        else:
            self.action_space = Box(low=low_action, high=high_action)

    @property
    def robot_model(self):
        return self._robot_model

    @property
    def robot_data(self):
        return self._robot_data

    @property
    def robot_frame(self):
        return self._robot_frame

    @property
    def puck_radius(self):
        return self._puck_radius

    @property
    def mallet_radius(self):
        return self._mallet_radius

    @property
    def dt(self):
        return self._dt

    @property
    def table_length(self):
        return self._table_length

    @property
    def table_width(self):
        return self._table_width

    @property
    def table_diag(self):
        return self._table_diag

    @property
    def goal_width(self):
        return self._goal_width

    @property
    def max_vel(self):
        return self._max_vel
