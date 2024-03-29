import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, flatten, unflatten
from envs.airhockeydoublewrapper import AirHockeyDouble
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
import numpy as np
from utils.ATACOM_transformation import AtacomTransformation, build_ATACOM_Controller

PENALTY_POINTS = {"joint_pos_constr": 2, "ee_constr": 3, "joint_vel_constr": 1, "jerk": 1,
                  "computation_time_minor": 0.5,
                  "computation_time_middle": 1, "computation_time_major": 2}


class AirHockeyGoal(gym.Env):
    def __init__(self, env: AirHockeyDouble, include_joints=False, include_ee=False, include_ee_vel=False,
                 joint_acc_clip=None, scale_obs=True, max_path_len=400, scale_action=True, remove_last_joint=True,
                 include_puck=True, alpha_r=1.0, goal_horizon=30):
        self.env = env
        self.env_info = self.env.env_info
        self.include_joints = include_joints
        self.include_puck = include_puck
        self.include_ee = include_ee
        self.include_ee_vel = include_ee_vel
        self.joint_acc_clip = joint_acc_clip
        self.remove_last_joint = remove_last_joint
        self.scale_obs = scale_obs
        self.scale_action = scale_action
        self.max_path_len = max_path_len

        self.alpha_r = alpha_r
        self.goal_horizon = goal_horizon

        self.robot_model = self.env_info['robot']['robot_model']
        self.robot_data = self.env_info['robot']['robot_data']
        self.puck_radius = self.env_info["puck"]["radius"]
        self.mallet_radius = self.env_info["mallet"]["radius"]
        self.dt = self.env_info['dt']

        self.table_length = self.env_info['table']['length']
        self.table_width = self.env_info['table']['width']
        self.table_diag = np.sqrt(self.table_length ** 2 + self.table_width ** 2)

        obs_space = self.env.env_info['rl_info'].observation_space
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
        joint_vel_ids = self.env_info["joint_vel_ids"]

        low_joints_pos = self.env_info["rl_info"].observation_space.low[joint_pos_ids]
        high_joints_pos = self.env_info["rl_info"].observation_space.high[joint_pos_ids]

        low_joints_vel = self.env_info["rl_info"].observation_space.low[joint_vel_ids]
        high_joints_vel = self.env_info["rl_info"].observation_space.high[joint_vel_ids]

        if self.remove_last_joint:
            low_joints_pos = low_joints_pos[:6]
            high_joints_pos = high_joints_pos[:6]
            low_joints_vel = low_joints_vel[:6]
            high_joints_vel = high_joints_vel[:6]

        joint_vel_norm = high_joints_vel - low_joints_vel
        joint_pos_norm = high_joints_pos - low_joints_pos

        self.max_vel = 5 # self.env_info["rl_info"].observation_space.high[puck_vel_ids][0]

        self._constr_scales = {
            'joint_pos_constr': np.concatenate([joint_pos_norm, joint_pos_norm]),
            'joint_vel_constr': np.concatenate([joint_vel_norm, joint_vel_norm]),
            'ee_constr': np.concatenate([ee_pos_norm[:2], ee_pos_norm[:2], ee_pos_norm[:2]])[:5]
        }

        self.idx_to_delete = np.hstack([puck_pos_ids[2], puck_vel_ids[2], opponent_ee_ids])
        self.idx_to_delete_goal = self.idx_to_delete

        if not self.include_puck:
            self.idx_to_delete = np.hstack([self.idx_to_delete, puck_pos_ids, puck_vel_ids])

        if not self.include_joints:
            self.idx_to_delete = np.hstack([self.idx_to_delete, joint_pos_ids, joint_vel_ids])

        if self.remove_last_joint:
            self.idx_to_delete = np.hstack([self.idx_to_delete, joint_pos_ids[-1], joint_vel_ids[-1]])

        low_state = np.delete(low_state, self.idx_to_delete, axis=0)
        high_state = np.delete(high_state, self.idx_to_delete, axis=0)

        # Set observation Space and Action Space
        self.min_dict = {}
        self.range_dict = {}

        self.min_dict['orig_obs'] = high_state - low_state
        self.range_dict['orig_obs'] = low_state

        if self.scale_obs:
            low_state = -1 * np.ones(low_state.shape)
            high_state = np.ones(high_state.shape)

        box_obs = Box(low_state, high_state)
        obs_dict = {'orig_obs': box_obs}

        if self.include_ee:
            if scale_obs:
                self.min_dict['ee_pos'] = low_position[:2]
                self.range_dict['ee_pos'] = high_position[:2] - low_position[:2]
                ee_obs = Box(-np.ones_like(low_position[:2]), np.ones_like(high_position[:2]))
            else:
                ee_obs = Box(low_position[:2], high_position[:2])
            obs_dict['ee_pos'] = ee_obs

        if self.include_ee_vel:
            if scale_obs:
                self.min_dict['ee_vel'] = -self.max_vel
                self.range_dict['ee_vel'] = 2 * self.max_vel
                ee_vel = Box(-np.ones((2,)), np.ones((2,)))
            else:
                ee_vel = Box(-self.max_vel * np.ones((2,)), self.max_vel * np.ones((2,)))
            obs_dict['ee_vel'] = ee_vel

        goal_dict = {'g_' + k: v for k, v in obs_dict.items() if k in ('ee_pos', 'ee_vel')}

        self.state_space: Dict = Dict(obs_dict)
        self.goal_space = Dict(goal_dict)

        self.observation_space = Dict({**obs_dict, **goal_dict})

        low_action = self.env_info['robot']['joint_acc_limit'][0]
        high_action = self.env_info['robot']['joint_acc_limit'][1]

        if self.joint_acc_clip is not None:
            low_action = np.clip(low_action, -self.joint_acc_clip, 0)
            high_action = np.clip(high_action, 0, self.joint_acc_clip)

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
        self.goal = self.goal_space.sample()
        self.goal_step = 0
        self.prev_obs = None

    def _scale_action(self, action):
        action = 0.5 * (action + 1) * self.ac_range + self._min_ac
        return action

    def relative_goal(self, obs: dict, goal: dict):
        obs = {k: obs[k.split('g_')[1]] for k in self.goal_space.keys()}
        obs_flatten = flatten(self.goal_space, obs)
        goal_flatten = flatten(self.goal_space, goal)

        rel_goal = unflatten(self.goal_space, goal_flatten - obs_flatten)
        return {k: v for k,v in rel_goal.items()}

    def step(self, action):
        if self.scale_action:
            action = self._scale_action(action)

        # add final joint
        if self.remove_last_joint:
            action = np.hstack([action, 0])

        action = self.atacom_transformation.draw_action(self._obs, action)

        obs, reward, done, info = self.env.step(action)
        self._obs = obs
        self.t += 1

        self.goal_step += 1
        if self.goal_step % self.goal_horizon == 0:
            self.goal = self.goal_space.sample()

        self._post_simulation(obs)
        #self.goal = {'ee_pos':self.ee_pos[:2]}
        info = self.process_info(info)

        obs = self.process_state(obs, None) # moved before reward

        reward, info = self.reward(obs, info)

        done = False # ignore environment done

        if self.t >= self.max_path_len:
            done = True

        rel_goal = self.relative_goal(obs, self.goal)

        # check if goal is reached
        if np.linalg.norm(rel_goal['g_ee_pos']) < 0.1 and np.linalg.norm(rel_goal['g_ee_vel']) < 0.1:
            done = True

        obs_and_goal = {**obs, **rel_goal}

        self.prev_obs = obs

        return obs_and_goal, reward, done, False, info

    def reset(self, seed=0, options=None):
        obs = self.env.reset()
        self.t = 0
        self.goal_step = 0
        self.has_hit = False

        self._post_simulation(obs)
        self.atacom_transformation.reset()

        obs = self.process_state(obs, None)
        self.prev_obs = obs

        self.goal = self.goal_space.sample()

        obs_and_goal = {**obs, **self.relative_goal(obs, self.goal)}

        return obs_and_goal, None

    def process_info(self, info):
        if self.remove_last_joint:
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
            norms = self._constr_scales[constr]
            slacks[slacks < 0] = 0
            slacks /= norms
            reward_constraints += PENALTY_POINTS[constr] * np.mean(slacks)
            penalty_sums += PENALTY_POINTS[constr]
        reward_constraints = - reward_constraints / penalty_sums
        return reward_constraints

    def reward(self, obs, info):
        """
        obs is already processed and scaled
        """

        # parameterized reward
        goal_array = flatten(self.goal_space, self.goal)
        obs_array = flatten(self.goal_space, {k: obs[k.split('g_')[1]] for k in self.goal_space.keys()})
        reward = - np.linalg.norm(goal_array - obs_array) # distance from the goal

        '''goal_pos = self.goal['g_ee_pos']
        ee_pos = obs['ee_pos']
        pos_rew = - np.linalg.norm(goal_pos - ee_pos) / (2 * np.sqrt(2))

        rel_goal = self.relative_goal(obs, self.goal)

        vel_rew = 0
        epsilon = 0.01 # if goal velocity is reached the vel reward is 100

        if np.linalg.norm(rel_goal['g_ee_pos']) < 0.1: # If reached the position
            goal_vel = self.goal['g_ee_vel']
            ee_vel = obs['ee_vel']
            vel_rew = 1 / (np.linalg.norm(goal_vel - ee_vel) + epsilon)

        reward = pos_rew + vel_rew'''

        reward_constraint = self.alpha_r * self._reward_constraints(info)
        reward += reward_constraint
        info['constr_reward'] = reward_constraint
        return reward, info

    def process_state(self, state, info):

        obs = np.delete(state, self.idx_to_delete, axis=0)

        obs = {'orig_obs': obs}

        if self.include_ee:
            obs['ee_pos'] = self.ee_pos[:2]  # do not include z coordinate

        if self.include_ee_vel:
            obs['ee_vel'] = self.ee_vel[:2]

        if self.scale_obs:
            obs = self._scale_obs(obs)

        obs = self._clip_obs(obs)

        return obs

    def _clip_obs(self, obs):
        obs_sp = self.state_space

        for k in ('orig_obs', 'ee_pos', 'ee_vel'):
            box_sp: Box = obs_sp[k]
            np.clip(obs[k], a_min=box_sp.low, a_max=box_sp.high)
        return obs

    @staticmethod
    def _scale(x, x_min, x_range):
        return (x - x_min) / x_range * 2 - 1

    def _scale_obs(self, obs):
        '''obs['orig_obs'] = self._scale(obs['orig_obs'], self._min_obs_original, self.obs_original_range)

        if self.include_ee:
            obs['ee_pos'] = self._scale(obs['ee_pos'], self._min_obs_ee_pos, self.obs_ee_pos_range)

        if self.include_ee_vel:
            obs['ee_vel'] = self._scale(obs['ee_vel'], self._min_obs_ee_vel, self.obs_ee_vel_range)'''
        for k in self.state_space:
            obs[k] = self._scale(obs[k], self.min_dict[k], self.range_dict[k])

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
            if previous_vel_norm <= current_vel_norm and distance <= (self.puck_radius + self.mallet_radius) * 1.1:
                self.has_hit = True

    def _apply_forward_velocity_kinematics(self, joint_pos, joint_vel):
        robot_model = self.env_info['robot']['robot_model']
        robot_data = self.env_info['robot']['robot_data']
        jac = jacobian(robot_model, robot_data, joint_pos)
        jac = jac[:3]  # last part of the matrix is about rotation. no need for it
        ee_vel = jac @ joint_vel
        return ee_vel

    def seed(self, seed=None):
        return self.env.seed(seed)

    def render(self, render_mode='human'):
        goal_pos = self.goal['g_ee_pos'].copy()

        if self.scale_obs:
            goal_pos = self.min_dict['ee_pos'] + (goal_pos + 1) * 0.5 * self.range_dict['ee_pos']

        goal_pos[0] -= 1.51

        self.env.base_env._data.site("goal_vis").xpos = np.hstack([goal_pos, 0])
        self.env.render()

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
