from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from baseline.baseline_agent.baseline_agent import BaselineAgent
import numpy as np


class AirHockeyDouble(AirHockeyChallengeWrapper):
    def __init__(self, interpolation_order=3, **kwargs):
        super().__init__("tournament", interpolation_order=interpolation_order, **kwargs)

        self.second_agent = BaselineAgent(self.env_info, agent_id=2)
        self.second_agent_obs = None

        self.action_idx = (np.arange(self.base_env.action_shape[0][0]),
                           np.arange(self.base_env.action_shape[1][0]))

    def reset(self, state=None):
        obs = super().reset()
        return self.filter_observation(obs)

    def filter_observation(self, obs):
        obs1, obs2 = np.split(obs, 2)
        self.second_agent_obs = obs2

        return obs1

    def step(self, action):
        second_agent_action = self.second_agent.draw_action(self.second_agent_obs)
        dual_action = (action, second_agent_action)

        obs, reward, done, info = super().step(np.array((dual_action[0][self.action_idx[0]],
                                                         dual_action[1][self.action_idx[1]])))

        return self.filter_observation(obs), reward, done, info

    def render(self):
        super().render()
