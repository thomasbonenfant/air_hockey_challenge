import gymnasium as gym
from gymnasium.spaces import Discrete, Box, MultiDiscrete, Dict
import numpy as np

from envs.airhockeydoublewrapper import AirHockeyDouble

PENALTY_POINTS = {"joint_pos_constr": 2, "ee_constr": 3, "joint_vel_constr": 1, "jerk": 1, "computation_time_minor": 0.5,
                  "computation_time_middle": 1,  "computation_time_major": 2}
#contraints = PENALTY_POINTS.keys()


class HierarchicalEnv(gym.Env):
    def __init__(self, env: AirHockeyDouble, steps_per_action: int, policies: list, policy_state_processors: dict, render_flag: bool,
                 include_timer=False, include_faults=False, large_reward=100, fault_risk_penalty=0.1, fault_penalty=33.33,
                 scale_obs=True, alpha_r=1., use_history=False, include_joints=False, include_ee=False):

        # Set Arguments
        self.env = env
        self.env_info = self.env.env_info
        self.render_flag = render_flag
        self.include_timer = include_timer
        self.include_faults = include_faults
        self.scale_obs = scale_obs
        self.reward = large_reward
        self.fault_penalty = fault_penalty
        self.fault_risk_penalty = fault_risk_penalty
        self.score = np.array([0, 0])
        self.faults = np.array([0, 0])
        self.policies = policies
        self.policy_state_processors = policy_state_processors
        self.steps_per_action = steps_per_action
        self.alpha_r = alpha_r
        self.use_history = use_history # not implemented
        self.include_joints = include_joints

        self.low_level_history = []

        self.action = 0 # need to save to detect a switch in the low level policy and reset the state of the new level policy

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
        low_joints_pos = self.env_info["rl_info"].observation_space.low[joint_pos_ids]
        high_joints_pos = self.env_info["rl_info"].observation_space.high[joint_pos_ids]
        joint_pos_norm = high_joints_pos - low_joints_pos
        joint_vel_ids = self.env_info["joint_vel_ids"]
        low_joints_vel = self.env_info["rl_info"].observation_space.low[joint_vel_ids]
        high_joints_vel = self.env_info["rl_info"].observation_space.high[joint_vel_ids]
        joint_vel_norm = high_joints_vel - low_joints_vel

        self._constr_scales = {
            'joint_pos_constr': np.concatenate([joint_pos_norm, joint_pos_norm]),
            'joint_vel_constr': np.concatenate([joint_vel_norm, joint_vel_norm]),
            'ee_constr': np.concatenate([ee_pos_norm[:2], ee_pos_norm[:2], ee_pos_norm[:2]])[:5]
        }

        self.idx_to_delete = [puck_pos_ids[2], puck_vel_ids[2], opponent_ee_ids[2]] # remove puck's theta and dtheta and ee z

        if not self.include_joints:
            self.idx_to_delete = np.hstack([self.idx_to_delete, joint_pos_ids, joint_vel_ids])

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

        # we don't want to scale the timer between -1 and 1, but between 0 and 1
        if self.include_timer:
            low_timer = 0
            high_timer = 1
            obs_dict['timer'] = Box(low_timer, high_timer)

        if self.include_faults:
            fault_obs = MultiDiscrete([3, 3])
            obs_dict['faults'] = fault_obs

        self.observation_space = Dict(obs_dict)

        self.action_space = Discrete(len(self.policies))

        self.state = None

        # variables to log

        self.log_constr_penalty = 0
        self.log_fault_penalty = 0
        self.log_fault_risk_penalty = 0
        self.log_large_reward = 0

        # make default state processors

        for policy in self.policies:
            if policy not in self.policy_state_processors:
                policy_state_processors[policy] = lambda x: x

    def render(self, mode='human'):
        self.env.render()
        self.render_flag = True

    def reset(self, seed=0, options=None):
        np.random.seed(seed)
        self.state = self.env.reset()
        for policy in self.policies:
            policy.reset()

        return self.process_state(self.state, None), None

    def process_state(self, state, info):
        '''if self.include_timer:
            state = np.hstack([state, np.clip(self.env.base_env.timer / 15, a_min=0, a_max=1)])

        if self.include_faults:
            state = np.hstack([state, self.faults])

        if self.scale_obs:
            state = self._scale_obs(state)'''

        obs = np.delete(state, self.idx_to_delete, axis=0)
        obs = {'orig_obs': obs}

        obs = self._scale_obs(obs)

        if self.include_faults:
            obs['faults'] = self.faults % 3
        if self.include_timer:
            obs['timer'] = np.clip(self.env.base_env.timer / 15, a_min=0, a_max=1)

        return obs

    def _scale_obs(self, obs):
        obs['orig_obs'] = (obs['orig_obs'] - self._min_obs_original) / self.obs_original_range * 2 - 1
        return obs

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

    def step(self, action: int):
        """
        Args:
            action: integer indexing the policy to use
        """

        # assert action in [i for i in range(len(self.policies))]

        policy = self.policies[action]

        if self.action != action:
            self.action = action
            policy.reset()

        steps = 0
        reward = 0
        done = False
        info = None

        # initialize variables to log
        self.log_fault_penalty = 0
        self.log_fault_risk_penalty = 0
        self.log_constr_penalty = 0
        self.log_large_reward = 0

        # print(f"Policy: {action}")

        if self.use_history:
            self.low_level_history = []

        while steps < self.steps_per_action and not done:
            low_level_action = policy.draw_action(self.policy_state_processors[policy](self.state))
            self.state, r, done, info = self.env.step(low_level_action)

            self.low_level_history.append(self.state)

            puck_pos = np.array(self.state[self.env_info["puck_pos_ids"]])

            # penalty at every step if the puck is in our part of the table. Introduced for reducing faults
            if puck_pos[0] + self.env_info["robot"]["base_frame"][0][0, 3] <= 0:
                reward -= self.fault_risk_penalty
                self.log_fault_risk_penalty -= self.fault_risk_penalty

            reward += self.alpha_r * self._reward_constraints(info)
            self.log_constr_penalty += self.alpha_r * self._reward_constraints(info)

            if self.render_flag:
                self.env.render()

            steps += 1

        # fill history with last state if early termination
        if self.use_history and done:
            self.low_level_history.extend([self.state] * (self.steps_per_action - len(self.low_level_history)))
            # TODO: complete history feature implementation

        # large sparse reward for a goal
        reward += (np.array(info["score"]) - self.score) @ np.array([1, -1]) * self.reward
        self.log_large_reward += (np.array(info["score"]) - self.score) @ np.array([1, -1]) * self.reward

        reward -= (np.array(info["faults"]) - self.faults) @ np.array([1, 0]) * self.fault_penalty
        self.log_fault_penalty -= (np.array(info["faults"]) - self.faults) @ np.array([1, 0]) * self.fault_penalty

        self.score = np.array(info['score'])
        self.faults = np.array(info["faults"])

        # add rewards in info
        reward_dict = {'large_reward': self.log_large_reward, 'constr_penalty': self.log_constr_penalty,
                       'fault_penalty': self.log_fault_penalty,
                       'fault_risk_penalty': self.log_fault_risk_penalty}
        info['reward'] = reward_dict

        return self.process_state(self.state, info), reward, done, False, info
