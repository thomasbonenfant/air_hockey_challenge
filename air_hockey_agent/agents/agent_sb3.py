from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.utils.sb3_variant_util import variant_util, load_variant
from gymnasium.spaces import Box, Dict, flatten_space, flatten
import numpy as np
from utils.ATACOM_transformation import AtacomTransformation, build_ATACOM_Controller
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
from stable_baselines3 import PPO, SAC
from air_hockey_agent.agents.kalman_filter import PuckTracker
import os


class AgentSB3(AgentBase):
    def __init__(self, env_info, path, acc_ratio=0.1, **kwargs):
        super().__init__(env_info, **kwargs)
        dir_path = os.path.dirname(os.path.abspath(__file__))

        path = os.path.join(dir_path, path)
        agent_path = os.path.join(path, 'best_model')
        self.env_info = env_info

        self.acc_ratio = acc_ratio

        env_args, alg_args, learn_args, log_args, variant = variant_util(load_variant(path))

        if log_args['alg'] == 'sac':
            self.agent = SAC.load(agent_path)
        elif log_args['alg'] == 'ppo':
            self.agent = PPO.load(agent_path)
        else:
            raise NotImplementedError

        del env_args['env']

        for key in env_args:
            setattr(self, key, env_args[key])

        self.robot_model = self.env_info['robot']['robot_model']
        self.robot_data = self.env_info['robot']['robot_data']
        self.puck_radius = self.env_info["puck"]["radius"]
        self.mallet_radius = self.env_info["mallet"]["radius"]
        self.dt = self.env_info['dt']

        self.table_length = self.env_info['table']['length']
        self.table_width = self.env_info['table']['width']
        self.table_diag = np.sqrt(self.table_length ** 2 + self.table_width ** 2)

        obs_space = self.env_info['rl_info'].observation_space
        low_state = obs_space.low
        high_state = obs_space.high

        puck_pos_ids = self.env_info['puck_pos_ids']
        puck_vel_ids = self.env_info['puck_vel_ids']
        opponent_ee_ids = self.env_info['opponent_ee_ids']

        # Constraints Scaling
        low_position = np.array([0.54, -0.5, 0])
        high_position = np.array([1.5, 0.5, 0.3])
        ee_pos_norm = high_position - low_position
        joint_pos_ids = self.env_info["joint_pos_ids"]
        low_joints_pos = self.env_info["rl_info"].observation_space.low[joint_pos_ids]
        high_joints_pos = self.env_info["rl_info"].observation_space.high[joint_pos_ids]
        joint_vel_ids = self.env_info["joint_vel_ids"]
        low_joints_vel = self.env_info["rl_info"].observation_space.low[joint_vel_ids]
        high_joints_vel = self.env_info["rl_info"].observation_space.high[joint_vel_ids]

        if self.remove_last_joint:
            low_joints_pos = low_joints_pos[:6]
            high_joints_pos = high_joints_pos[:6]
            low_joints_vel = low_joints_vel[:6]
            high_joints_vel = high_joints_vel[:6]

        joint_pos_norm = high_joints_pos - low_joints_pos
        joint_vel_norm = high_joints_vel - low_joints_vel
        self.max_vel = self.env_info["rl_info"].observation_space.high[puck_vel_ids][0]

        self._constr_scales = {
            'joint_pos_constr': np.concatenate([joint_pos_norm, joint_pos_norm]),
            'joint_vel_constr': np.concatenate([joint_vel_norm, joint_vel_norm]),
            'ee_constr': np.concatenate([ee_pos_norm[:2], ee_pos_norm[:2], ee_pos_norm[:2]])[:5]
        }

        self.idx_to_delete = np.hstack([puck_pos_ids[2], puck_vel_ids[2], opponent_ee_ids])

        if not self.include_puck:
            self.idx_to_delete = np.hstack([self.idx_to_delete, puck_pos_ids, puck_vel_ids])

        if not self.include_joints:
            self.idx_to_delete = np.hstack([self.idx_to_delete, joint_pos_ids, joint_vel_ids])

        if self.remove_last_joint:
            self.idx_to_delete = np.hstack([self.idx_to_delete, joint_pos_ids[-1], joint_vel_ids[-1]])

        low_state = np.delete(low_state, self.idx_to_delete, axis=0)
        high_state = np.delete(high_state, self.idx_to_delete, axis=0)

        # Set observation Space and Action Space
        self.obs_original_range = high_state - low_state
        self._min_obs_original = low_state

        if self.scale_obs:
            low_state = -1 * np.ones(low_state.shape)
            high_state = np.ones(high_state.shape)

        box_obs = Box(low_state, high_state)
        obs_dict = {'orig_obs': box_obs}

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

        self.dict_observation_space = Dict(obs_dict)
        self.observation_space = flatten_space(self.dict_observation_space)

        low_action = self.acc_ratio * self.env_info['robot']['joint_acc_limit'][0]
        high_action = self.acc_ratio * self.env_info['robot']['joint_acc_limit'][1]

        if self.remove_last_joint:
            low_action = low_action[:6]
            high_action = high_action[:6]

        atacom = build_ATACOM_Controller(self.env_info, slack_type='soft_corner', slack_tol=1e-06, slack_beta=4)
        self.atacom_transformation = AtacomTransformation(self.env_info, False, atacom)

        if self.scale_action:
            self._min_ac = low_action
            self.ac_range = high_action - low_action
            self.action_space = Box(low=-np.ones_like(low_action), high=np.ones_like(high_action))
        else:
            self.action_space = Box(low=low_action, high=high_action)

        self._obs = None
        self.has_hit = False

        self.t = 0
        self.restart = True
        self.puck_tracker = PuckTracker(env_info)
        self.last_joint_pos_action = None
        self.last_joint_vel_action = None

    def reset(self):
        self.restart = True
        self.has_hit = False
        self.atacom_transformation.reset()

    def process_state(self, state):
        obs = np.delete(state, self.idx_to_delete, axis=0)

        obs = {'orig_obs': obs}

        if self.include_ee:
            joint_pos = state[self.env_info['joint_pos_ids']]
            mj_model = self.env_info['robot']['robot_model']
            mj_data = self.env_info['robot']['robot_data']

            obs['ee_pos'] = self.ee_pos[:2]  # do not include z coordinate

        if self.include_ee_vel:
            obs['ee_vel'] = self.ee_vel[:2]

        if self.scale_obs:
            obs = self._scale_obs(obs)

        # flatten observation

        return flatten(self.dict_observation_space, obs)

    def _scale_obs(self, obs):
        obs['orig_obs'] = (obs['orig_obs'] - self._min_obs_original) / self.obs_original_range * 2 - 1

        if self.include_ee:
            obs['ee_pos'] = (obs['ee_pos'] - self._min_obs_ee_pos) / self.obs_ee_pos_range * 2 - 1

        if self.include_ee_vel:
            obs['ee_vel'] = (obs['ee_vel'] - self._min_obs_ee_vel) / self.obs_ee_vel_range * 2 - 1
        return obs

    def _scale_action(self, action):
        action = 0.5 * (action + 1) * self.ac_range + self._min_ac
        action = np.clip(action, self._min_ac, self._min_ac + self.ac_range)
        return action

    def noise_filter(self, observation):
        noisy_puck_pos = self.get_puck_pos(observation)

        if self.restart:
            self.puck_tracker.reset(noisy_puck_pos)

        self.puck_tracker.step(noisy_puck_pos)

        puck_pos = self.puck_tracker.state[[0, 1, 4]].copy()
        puck_vel = self.puck_tracker.state[[2, 3, 5]].copy()

        observation[self.env_info['puck_pos_ids']] = puck_pos

        if not self.restart:
            observation[self.env_info['puck_vel_ids']] = puck_vel
            observation[self.env_info['joint_pos_ids']] = self.last_joint_pos_action.copy()
            observation[self.env_info['joint_vel_ids']] = self.last_joint_vel_action.copy()

        self.restart = False
        return observation

    def draw_action(self, observation):

        # noise filter
        observation = self.noise_filter(observation)

        self._post_simulation(observation)

        obs = self.process_state(observation)

        action, _ = self.agent.predict(obs, deterministic=True)

        action = self._scale_action(action)

        # add final joint
        if self.remove_last_joint:
            action = np.hstack([action, 0])

        action = self.atacom_transformation.draw_action(self._obs, action)

        # check inside boundaries
        assert np.all(action[1] > 0.95 * self.env_info["robot"]["joint_vel_limit"][0]) and \
            np.all(action[1] < 0.95 * self.env_info["robot"]["joint_vel_limit"][1])

        self.last_joint_pos_action = action[0]
        self.last_joint_vel_action = action[1]

        self.t += 1

        return action

    def _post_simulation(self, obs):
        self._obs = obs
        self.puck_pos = self.get_puck_pos(obs)
        self.previous_vel = self.puck_vel if self.t > 0 else None
        self.puck_vel = self.get_puck_vel(obs)
        self.joint_pos = self.get_joint_pos(obs)
        self.joint_vel = self.get_joint_vel(obs)
        self.previous_ee_pos = self.ee_pos if self.t > 0 else None
        self.ee_pos = self.get_ee_pose(obs)

        self.ee_vel = self._apply_forward_velocity_kinematics(self.joint_pos, self.joint_vel)
        if self.previous_vel is not None:
            previous_vel_norm = np.linalg.norm(self.previous_vel[:2])
            current_vel_norm = np.linalg.norm(self.puck_vel[:2])
            distance = np.linalg.norm(self.puck_pos[:2] - self.ee_pos[:2])
            if previous_vel_norm <= current_vel_norm and distance <= (self.puck_radius + self.mallet_radius) * 1.1:
                self.has_hit = True

    def _apply_forward_velocity_kinematics(self, joint_pos, joint_vel):
        robot_model = self.env_info['robot']['robot_model']
        robot_data = self.env_info['robot']['robot_data']
        jac = jacobian(robot_model, robot_data, joint_pos)
        jac = jac[:3]  # last part of the matrix is about rotation. no need for it
        ee_vel = jac @ joint_vel
        return ee_vel

    def get_puck_pos(self, obs):

        return obs[self.env_info['puck_pos_ids']]

    def get_puck_vel(self, obs):

        return obs[self.env_info['puck_vel_ids']]

    def get_joint_pos(self, obs):

        return obs[self.env_info['joint_pos_ids']]

    def get_joint_vel(self, obs):

        return obs[self.env_info['joint_vel_ids']]

    def get_ee_pose(self, obs):

        res = forward_kinematics(self.robot_model, self.robot_data, self.get_joint_pos(obs))
        return res[0]

    def get_opponent_ee_pose(self, obs):

        return obs[self.env_info['opponent_ee_ids']]





