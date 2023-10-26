from air_hockey_challenge.framework import AgentBase
from baseline.baseline_agent.baseline_agent import BaselineAgent
import numpy as np


class DummyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.step = 0
        self.first_position = None

    def reset(self):
        self.step = 0

    def draw_action(self, observation):
        if self.step == 0:
            self.first_position = self.get_joint_pos(observation)

        action = np.vstack([self.first_position, np.zeros((7,))])

        self.step += 1
        return action
