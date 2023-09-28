from air_hockey_challenge.framework import AgentBase
from stable_baselines3 import PPO
from gymnasium.spaces import Box, Dict, MultiDiscrete, flatten
import os.path
import numpy as np

from my_scripts.utils import variant_util, load_variant


class PPOAgent(AgentBase):
    def __init__(self, env_info, path: str, policies: list, state_preprocessors: dict, **kwargs):
        super().__init__(env_info, **kwargs)
        self.policies = policies
        self.state_preprocessors = state_preprocessors

        env_args, alg_args, learn_args, log_args, variant = variant_util(load_variant(path))

        self.agent = PPO.load(os.path.join(path, 'best_model'))
        self.policy = None
        self.counter = 0
        self.steps_per_action = env_args['steps_per_action']
        self.include_joints = env_args['include_joints']
        self.include_timer = env_args['include_timer']
        self.include_faults = env_args['include_faults']
        #self.use_history = env_args['use_history'] # not implemented
        self.scale_obs = env_args['scale_obs']
        self.low_level_history = []

        self.faults = 0
        self.score = 0
        self.prev_side = 1
        self.step = 0

        self.timer = 0
        self.dt = self.env_info['dt']

        for policy in self.policies:
            if policy not in self.state_preprocessors:
                self.state_preprocessors[policy] = lambda x: x

        puck_pos_ids = self.env_info['puck_pos_ids']
        puck_vel_ids = self.env_info['puck_vel_ids']
        opponent_ee_ids = self.env_info['opponent_ee_ids']

        obs_space = self.env_info['rl_info'].observation_space

        low_state = obs_space.low
        high_state = obs_space.high
        joint_pos_ids = self.env_info["joint_pos_ids"]
        joint_vel_ids = self.env_info["joint_vel_ids"]

        self.idx_to_delete = [puck_pos_ids[2], puck_vel_ids[2],
                              opponent_ee_ids[2]]  # remove puck's theta and dtheta and ee z

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

    def process_state(self, state):
        obs = np.delete(state, self.idx_to_delete, axis=0)
        obs = {'orig_obs': obs}

        obs = self._scale_obs(obs)

        if self.include_faults:
            obs['faults'] = self.faults % 3
        if self.include_timer:
            obs['timer'] = np.clip(self.timer / 15, a_min=0, a_max=1)

        return flatten(self.observation_space, obs)

    def _scale_obs(self, obs):
        obs['orig_obs'] = (obs['orig_obs'] - self._min_obs_original) / self.obs_original_range * 2 - 1
        return obs

    def reset(self):
        self.timer = 0

        for policy in self.policies:
            policy.reset()

    def draw_action(self, observation):
        high_level_obs = self.process_state(observation)

        if self.step % self.steps_per_action == 0 or self.policy is None:
            high_level_action, _ = self.agent.predict(high_level_obs, deterministic=True)
            print(f'Step: {self.step}\t Action: {high_level_action}')
            policy = self.policies[high_level_action]
            if self.policy != policy:
                self.policy = policy
                #self.policy.reset()


        action = self.policy.draw_action(self.state_preprocessors[self.policy](observation))

        curr_side = np.sign(self.get_puck_pos(observation)[0])

        # at the end increment timer
        if np.sign(self.get_puck_pos(observation)[0]) == self.prev_side:
            self.timer += self.dt
        else:
            self.prev_side *= curr_side
            self.timer = 0

        self.step += 1
        return action
