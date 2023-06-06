from air_hockey_challenge.framework import AgentBase
import copy
import numpy as np
from numpy.linalg import pinv
from gym import spaces
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
from mepol.src.policy import GaussianPolicy
import torch


class CustomAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)

        self.env_info = env_info

        n_joints = env_info['robot']['n_joints']

        self.world2robot_transf = env_info['robot']['base_frame'][0]
        self.mj_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.mj_data = copy.deepcopy(env_info['robot']['robot_data'])

        puck_pos_ids = env_info['puck_pos_ids']
        puck_vel_ids = env_info['puck_vel_ids']
        joint_pos_ids = env_info['joint_pos_ids']
        joint_vel_ids = env_info['joint_vel_ids']

        table_length = self.env_info['table']['length']
        table_width = self.env_info['table']['width']

        self.joint_min_pos = joint_min_pos = env_info['rl_info'].observation_space.low[joint_pos_ids]
        self.joint_max_pos = joint_max_pos = env_info['rl_info'].observation_space.high[joint_pos_ids]
        self.joint_min_vel = joint_min_vel = env_info['rl_info'].observation_space.low[joint_vel_ids]
        self.joint_max_vel = joint_max_vel = env_info['rl_info'].observation_space.high[joint_vel_ids]

        self.max_ee_pos_action = np.array([0, table_width / 2])
        self.min_ee_pos_action = np.array([-table_length / 2, -table_width / 2])

        # to calculate joint velocity
        self.old_joint_pos = np.zeros(n_joints)

        self.task_space = kwargs['task_space']
        self.task_space_vel = kwargs['task_space_vel']
        self.use_delta_pos = kwargs['use_delta_pos']
        self.delta_dim = kwargs['delta_dim']
        self.use_puck_distance = kwargs['use_puck_distance']
        self.prev_ee_pos = None

        if self.task_space:

            if self.task_space_vel:
                action_space_dim = 4  # x,y, dx, dy

                max_action = np.hstack((self.max_ee_pos_action, [1, 1]))
                min_action = np.hstack((self.min_ee_pos_action, [-1, -1]))
                self.action_space = spaces.Box(low=min_action, high=max_action,
                                               shape=(action_space_dim,), dtype=np.float32)
            else:
                action_space_dim = 2
                if self.use_delta_pos:
                    max_action = np.array([self.delta_dim, self.delta_dim])
                    min_action = -max_action
                else:
                    max_action = self.max_ee_pos_action
                    min_action = self.min_ee_pos_action
                self.action_space = spaces.Box(low=min_action, high=max_action,
                                               shape=(action_space_dim,), dtype=np.float32)
        else:
            action_space_dim = 2 * n_joints
            low_action = np.hstack((joint_min_pos, joint_min_vel))
            high_action = np.hstack((joint_max_pos, joint_max_vel))
            self.action_space = spaces.Box(low=low_action, high=high_action,
                                           shape=(action_space_dim,), dtype=np.float32)

        # observation space specification

        puck_pos_low = np.array([-table_length / 2, -table_width / 2])
        puck_pos_high = np.array([table_length / 2, table_width / 2])
        ee_pos_low = np.array([-table_length / 2, -table_width / 2])
        ee_pos_high = np.array([table_length / 2, table_width / 2])
        puck_vel_low = env_info['rl_info'].observation_space.low[puck_vel_ids][:2]
        puck_vel_high = env_info['rl_info'].observation_space.high[puck_vel_ids][:2]
        ee_vel_low = env_info['rl_info'].observation_space.low[puck_vel_ids][:2]
        ee_vel_high = env_info['rl_info'].observation_space.high[puck_vel_ids][:2]

        ee_puck_dist_low = puck_pos_low - ee_pos_high
        ee_puck_dist_high = puck_pos_high - ee_pos_low

        if self.task_space:
            self.num_features = 14  # puck_x, puck_y, dpuck_x, dpuck_y, joint_pos, joint_vel, ee_x, ee_y, deex, deey
            if self.use_puck_distance:
                obs_low = np.hstack(
                    [puck_pos_low, puck_vel_low, joint_min_pos, joint_min_vel, ee_puck_dist_low, ee_vel_low])
                obs_high = np.hstack(
                    [puck_pos_high, puck_vel_high, joint_max_pos, joint_max_vel, ee_puck_dist_high, ee_vel_high])
            else:
                obs_low = np.hstack(
                    [puck_pos_low, puck_vel_low, joint_min_pos, joint_min_vel, ee_pos_low, ee_vel_low])
                obs_high = np.hstack([puck_pos_high, puck_vel_high, joint_max_vel, ee_pos_high, ee_vel_high])
        else:
            self.num_features = 10
            obs_low = np.hstack([puck_pos_low, puck_vel_low, joint_min_pos, joint_min_vel])
            obs_high = np.hstack([puck_pos_high, puck_vel_high, joint_max_vel, joint_max_vel])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(self.num_features,), dtype=np.float32)

        self.policy = GaussianPolicy(
            hidden_sizes=kwargs['hidden_sizes'],
            num_features=self.num_features,
            action_dim=self.action_space.shape[0],
            activation=torch.nn.ReLU,
            log_std_init=kwargs['log_std_init'],
            use_tanh=kwargs['use_tanh']
        )
        if kwargs['init_policy'] is not None:
            print(f'Loading policy {kwargs["init_policy"]}')
            self.policy.load_state_dict(torch.load(kwargs['init_policy']))

    def reset(self):
        pass

    def draw_action(self, observation):
        # convert observation for policy
        observation = self._convert_obs(observation)

        action = self.policy.predict(observation, deterministic=False)

        action = self._scale_action(action)

        action = self.convert_action(action)
        return action

    def _convert_obs(self, obs):

        self.old_joint_pos = obs[self.env_info['joint_pos_ids']]

        puck_pos, _ = robot_to_world(self.env_info['robot']['base_frame'][0],
                                     obs[self.env_info['puck_pos_ids']])
        puck_pos = puck_pos[:2]

        puck_vel = obs[self.env_info['puck_vel_ids']]
        puck_vel = puck_vel[:2]

        joint_pos = obs[self.env_info['joint_pos_ids']]
        joint_vel = obs[self.env_info['joint_vel_ids']]

        if self.task_space:
            ee_pos, _ = forward_kinematics(self.mj_model, self.mj_data, joint_pos)
            ee_pos, _ = robot_to_world(self.world2robot_transf, ee_pos)
            ee_pos = ee_pos[:2]

            # saves ee_pos for delta pos calculation
            self.prev_ee_pos = ee_pos
            ee_vel = self._apply_forward_velocity_kinematics(joint_pos, joint_vel)[:2]

            if self.use_puck_distance:
                obs = np.hstack((puck_pos, puck_vel, joint_pos, joint_vel, (puck_pos - ee_pos), ee_vel))
            else:
                obs = np.hstack((puck_pos, puck_vel, joint_pos, joint_vel, ee_pos, ee_vel))
        else:
            obs = np.hstack((puck_pos, puck_vel, joint_pos, joint_vel))

        obs = self.to_n1p1(obs)

        return obs

    def _apply_forward_velocity_kinematics(self, joint_pos, joint_vel):

        jac = jacobian(self.mj_model, self.mj_data, joint_pos)[:3]  # last part of the matrix is about rotation.
        ee_vel = jac @ joint_vel

        return ee_vel

    def _apply_inverse_kinematics(self, ee_pos_robot_frame):
        env_info = self.env_info

        position_robot_frame, rotation = world_to_robot(self.world2robot_transf, ee_pos_robot_frame)
        success, action_joints = inverse_kinematics(self.mj_model, self.mj_data,
                                                    position_robot_frame)
        return action_joints

    def to_n1p1(self, state):
        v_min = self.observation_space.low
        v_max = self.observation_space.high
        if any(v_min == - np.inf) or any(v_max == np.inf):
            raise ValueError('unbounded state')
        new_min, new_max = -1, 1
        res = (state - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        res = np.nan_to_num(res, nan=0) # if we want to keep z at zero
        res = np.clip(res, new_min, new_max)
        return res

    def _scale_action(self, action):
        action = action.numpy()
        lb = self.action_space.low
        ub = self.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        return scaled_action

    def convert_action(self, action):
        action = np.array(action)

        if self.task_space:
            if self.task_space_vel:
                ee_pos_action = action[:2]
            else:
                ee_pos_action = action

            # if the action is only a delta position update the resulting ee position action
            if self.use_delta_pos:
                # scale delta action
                ee_pos_action *= self.delta_dim

                ee_pos_action = self.prev_ee_pos + ee_pos_action

            # clip action inside table
            ee_pos_action = np.clip(ee_pos_action, a_min=self.min_ee_pos_action, a_max=self.max_ee_pos_action)

            # make action 3D
            ee_pos_action = np.hstack((ee_pos_action, 0))

            joint_pos_action = self._apply_inverse_kinematics(ee_pos_action)

            if self.task_space_vel:
                ee_vel_action = np.hstack((action[2:], 0))
                jac = jacobian(self.mj_model, self.mj_data, joint_pos_action)[:3]
                joint_vel_action = pinv(jac) @ ee_vel_action

            else:
                joint_vel_action = (joint_pos_action - self.old_joint_pos) / self.env_info['dt']

            # creates (2,3) array
            action = np.vstack((joint_pos_action, joint_vel_action))

        else:
            action = np.reshape(action, (2, 3))

        return action
