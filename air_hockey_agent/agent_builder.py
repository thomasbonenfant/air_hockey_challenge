import numpy as np
from air_hockey_challenge.framework import AgentBase


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    return DummyAgent(env_info, **kwargs)


class DummyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

        if kwargs['goal'] is not None:
            self.goal = np.array(kwargs['goal'])
        else:
            self.goal = np.array([2,0,0])

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        if self.new_start:
            self.new_start = False
            self.joint_position = self.get_joint_pos(observation)

        action = self.goal
        #print(f'Joints Position: {self.hold_position}')
        #print(f'Action: {action}')

        return action
