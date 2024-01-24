from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_agent.delayed_baseline import DelayedBaseline
import numpy as np


class AirHockeyDouble(AirHockeyChallengeWrapper):
    def __init__(self, interpolation_order=3, opponent_delay=500, render=False, **kwargs):
        super().__init__("tournament", interpolation_order=interpolation_order, **kwargs)

        self.second_agent = DelayedBaseline(self.env_info,start_time=opponent_delay, agent_id=2)
        self.second_agent_obs = None

        self.action_idx = (np.arange(self.base_env.action_shape[0][0]),
                           np.arange(self.base_env.action_shape[1][0]))
        self.render_flag = render

    def reset(self, state=None):
        obs = super().reset()
        if self.render_flag:
            self.render()

        self.second_agent.reset()
        return self.filter_observation(obs)

    def filter_observation(self, obs):
        obs1, obs2 = np.split(obs, 2)
        self.second_agent_obs = obs2

        return obs1

    def filter_info(self, info):
        constr_dict = info['constraints_value'][0]
        for constr in constr_dict:
            info[constr] = constr_dict[constr]
        del info['constraints_value']
        return info

    def step(self, action):
        second_agent_action = self.second_agent.draw_action(self.second_agent_obs)
        dual_action = (action, second_agent_action)

        obs, reward, done, info = super().step(np.array((dual_action[0][self.action_idx[0]],
                                                         dual_action[1][self.action_idx[1]])))
        if self.render_flag:
            self.render()

        return self.filter_observation(obs), reward, done, self.filter_info(info)

    def render(self):
        super().render()
