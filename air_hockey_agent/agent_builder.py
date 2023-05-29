from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.agents.hit_agent_SAC import HittingAgent
from air_hockey_agent.agents.ATACOM_hit_agent import AtacomHittingAgent
from air_hockey_challenge.utils.transformations import robot_to_world
import pandas as pd

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

    return hit_agent(env_info, **kwargs)
    # agent = DummyAgent(env_info, **kwargs)
    # if 'agent' in kwargs:
    #     # if kwargs['agent'] == 'dummy-agent':
    #     #    agent = DummyAgent(env_info, **kwargs)
    #     # elif kwargs['agent'] == 'defend-agent':
    #     #    agent = SimpleDefendingAgent(env_info, **kwargs)
    #     if kwargs['agent'] == 'hit-agent':
    #         agent = AtacomHittingAgent(env_info, **kwargs)

class hit_agent(AtacomHittingAgent):
    def __init__(self, env_info, **kwargs):
        super(hit_agent, self).__init__(env_info, **kwargs)
        self.dataset = pd.DataFrame()

    def draw_action(self, observation):
        actions = super(hit_agent, self).draw_action(observation)

        self.update_dataset(observation)

        return actions

    def update_dataset(self, observation):
        puck_pos, puck_vel = self.get_puck_state(observation)
        new_data = pd.DataFrame({'puck current pos X': [puck_pos[0]],
                                 'puck current pos Y': [puck_pos[1]],
                                 'puck current vel X': [puck_vel[0]],
                                 'puck current vel Y': [puck_vel[1]]})
        # self.dataset.append(new_data, ignore_index=True)
        self.dataset = pd.concat([self.dataset, new_data], ignore_index=True)

    def __del__(self):
        self.dataset.to_csv('Dataset/data.csv', index=False)
        exit(10)
        super(hit_agent, self).__del__()