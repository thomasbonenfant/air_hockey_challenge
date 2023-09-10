import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np

from envs.airhockeydoublewrapper import AirHockeyDouble


class HierarchicalEnv(gym.Env):
    def __init__(self, env: AirHockeyDouble, steps_per_action: int, policies: list, policy_state_processors: dict, render_flag: bool,
                 include_timer=False):
        self.env = env
        self.env_info = self.env.env_info
        self.render_flag = render_flag
        self.include_timer = include_timer

        obs_space = self.env.env_info['rl_info'].observation_space

        low_state = obs_space.low
        high_state = obs_space.high

        if self.include_timer:
            low_state = np.hstack([low_state, 0])
            high_state = np.hstack([high_state, 1])

        self.observation_space = Box(low_state, high_state)
        self.policies = policies
        self.policy_state_processors = policy_state_processors
        self.steps_per_action = steps_per_action

        self.action_space = Discrete(len(self.policies))

        self.state = None

        self.reward = 1000
        self.fault_risk_penalty = -1
        self.score = np.array([0, 0])
        self.faults = np.array([0, 0])

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
        if self.include_timer:
            state = np.hstack([state, np.clip(self.env.base_env.timer / 15, a_min=0, a_max=1)])
        return state

    def step(self, action: int):
        """
        Args:
            action: integer indexing the policy to use
        """

        # assert action in [i for i in range(len(self.policies))]

        policy = self.policies[action]

        steps = 0
        reward = 0
        done = False
        info = None

        # print(f"Policy: {action}")

        while steps < self.steps_per_action and not done:
            low_level_action = policy.draw_action(self.policy_state_processors[policy](self.state))
            self.state, r, done, info = self.env.step(low_level_action)

            puck_pos = np.array(self.state[self.env_info["puck_pos_ids"]])

            # penalty at every step if the puck is in our part of the table. Introduced for reducing faults
            if puck_pos[0] + self.env_info["robot"]["base_frame"][0][0, 3] <= 0:
                reward += self.fault_risk_penalty

            if self.render_flag:
                self.env.render()

            steps += 1

        # large sparse reward for a goal
        reward += (np.array(info["score"]) - self.score) @ np.array([1, -1]) * self.reward
        self.score = np.array(info['score'])
        #print(reward)

        return self.process_state(self.state, info), reward, done, False, info
