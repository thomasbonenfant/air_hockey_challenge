import gymnasium as gym
from gymnasium.spaces import Box, Dict
from envs.airhockeydoublewrapper import AirHockeyDouble
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
import numpy as np
from utils.ATACOM_transformation import AtacomTransformation, build_ATACOM_Controller
from envs.utils import Specification

PENALTY_POINTS = {"joint_pos_constr": 2, "ee_constr": 3, "joint_vel_constr": 1, "jerk": 1,
                  "computation_time_minor": 0.5,
                  "computation_time_middle": 1, "computation_time_major": 2}


class AirHockeyOption(gym.Env):
    def __init__(self, env, include_joints=False, include_ee=False, include_ee_vel=False,
                 joint_acc_clip=None, scale_obs=True, max_path_len=400, scale_action=True, remove_last_joint=True,
                 include_puck=True, alpha_r=1.0, stop_after_hit=False):
        self.env = env
        #self.env_info = self.env.env_info

        self.specs = Specification(env_info=env.env_info,
                                   include_joints=include_joints,
                                   include_ee=include_ee,
                                   include_ee_vel=include_ee_vel,
                                   include_puck=include_puck,
                                   include_opponent=False, # for options we do not need the opponent
                                   joint_acc_clip=joint_acc_clip,
                                   scale_obs=scale_obs,
                                   scale_action=scale_action,
                                   max_path_len=max_path_len,
                                   remove_last_joint=remove_last_joint,
                                   alpha_r=alpha_r,
                                   stop_after_hit=stop_after_hit)

        self.state = None

        atacom = build_ATACOM_Controller(env.env_info, slack_type='soft_corner', slack_tol=1e-06, slack_beta=4)
        self.atacom_transformation = AtacomTransformation(env.env_info, False, atacom)

        self.observation_space = self.specs.observation_space
        self.action_space = self.specs.action_space

        self._obs = None
        self.has_hit = False
        self.hit_rew_given = False

        self.t = 0

    def _scale_action(self, action):
        action = 0.5 * (action + 1) * self.specs.ac_range + self.specs._min_ac
        return action

    def reward(self, done, info):
        '''dummy reward function'''
        return 0, info

    def step(self, action):
        if self.specs.scale_action:
            action = self._scale_action(action)

        # add final joint
        if self.specs.remove_last_joint:
            action = np.hstack([action, 0])

        action = self.atacom_transformation.draw_action(self._obs, action)

        obs, reward, done, info = self.env.step(action)
        self._obs = obs
        self.t += 1

        self._post_simulation(obs)
        info = self.process_info(info)
        reward, info = self.reward(info, done)

        if self.has_hit and self.specs.stop_after_hit:
            while not done and self.t < self.specs.max_path_len:
                # draws random action
                action = self.action_space.sample()
                if self.specs.remove_last_joint:
                    action = np.hstack([action, 0])
                action = self.atacom_transformation.draw_action(self._obs, action)

                obs, rew, done, info = self.env.step(action)
                self.t += 1
                self.env.render()
                self._post_simulation(obs)
                info = self.process_info(info)
                rew, info = self.reward(info, done)

                reward += rew

        if self.t >= self.specs.max_path_len:
            done = True

        return self.process_state(obs, None), reward, done, False, info

    def reset(self, seed=0, options=None):
        obs = self.env.reset()

        self.t = 0
        self.has_hit = False
        self.hit_rew_given = False

        self._post_simulation(obs)
        self.atacom_transformation.reset()

        return self.process_state(obs, None), None

    def process_info(self, info):

        if 'constraints_value' in info:
            info_constr = info['constraints_value']
            del info['constraints_value']
            for k, v in info_constr.items():
                info[k] = v

        if self.specs.remove_last_joint:
            idx_to_delete = [6, 13]
        else:
            idx_to_delete = []
        for constr in ['joint_pos_constr', 'joint_vel_constr']:
            info[constr] = np.delete(info[constr], idx_to_delete, axis=0)
        return info

    def _reward_constraints(self, info):
        reward_constraints = 0
        penalty_sums = 0
        for constr in ['joint_pos_constr', 'joint_vel_constr', 'ee_constr']:
            slacks = info[constr]
            norms = self.specs._constr_scales[constr]
            slacks[slacks < 0] = 0
            slacks /= norms
            reward_constraints += PENALTY_POINTS[constr] * np.mean(slacks)
            penalty_sums += PENALTY_POINTS[constr]
        reward_constraints = - reward_constraints / penalty_sums
        return reward_constraints

    def process_state(self, state, info):

        obs = {}

        if self.specs.include_puck:
            obs['puck_pos'] = state[self.specs.f_puck_pos_ids]
            obs['puck_vel'] = state[self.specs.f_puck_vel_ids]

        if self.specs.include_joints:
            obs['joint_pos'] = state[self.specs.f_joint_pos_ids]
            obs['joint_vel'] = state[self.specs.f_joint_vel_ids]

        if self.specs.include_opponent:
            obs['opponent_ee'] = state[self.specs.f_opponent_ee_ids]

        if self.specs.include_ee:
            obs['ee_pos'] = self.ee_pos[:2]  # do not include z coordinate

        if self.specs.include_ee_vel:
            obs['ee_vel'] = self.ee_vel[:2]

        if self.specs.scale_obs:
            obs = self._scale_obs(obs)

        return obs

    @staticmethod
    def _scale(x, x_min, x_range):
        return (x - x_min) / x_range * 2 - 1

    def _scale_obs(self, obs):
        for k in self.specs.min_dict:
            obs[k] = self._scale(obs[k], self.specs.min_dict[k], self.specs.range_dict[k])
            #obs[k] = np.clip(obs[k], a_min=self.specs.min_dict[k], a_max=self.specs.min_dict[k] + self.specs.range_dict[k])
        return obs

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
            if previous_vel_norm <= current_vel_norm and distance <= (self.specs.puck_radius + self.specs.mallet_radius) * 1.1:
                self.has_hit = True

    def _apply_forward_velocity_kinematics(self, joint_pos, joint_vel):
        robot_model = self.specs.robot_model
        robot_data = self.specs.robot_data
        jac = jacobian(robot_model, robot_data, joint_pos)
        jac = jac[:3]  # last part of the matrix is about rotation. no need for it
        ee_vel = jac @ joint_vel
        return ee_vel

    def seed(self, seed=None):
        return self.env.seed(seed)

    def render(self, render_mode='human'):
        self.env.render()

    def get_puck_pos(self, obs):

        return obs[self.specs.puck_pos_ids]

    def get_puck_vel(self, obs):

        return obs[self.specs.puck_vel_ids]

    def get_joint_pos(self, obs):

        return obs[self.specs.joint_pos_ids]

    def get_joint_vel(self, obs):

        return obs[self.specs.joint_vel_ids]

    def get_ee_pose(self, obs):

        res = forward_kinematics(self.specs.robot_model, self.specs.robot_data, self.get_joint_pos(obs))
        return res[0]

    def get_opponent_ee_pose(self, obs):

        return obs[self.specs.opponent_ee_obs]
