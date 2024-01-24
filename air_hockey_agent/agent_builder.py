from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.agents.hierarchical_agent import HierarchicalAgent
from air_hockey_agent.agents.rule_based_agent import PolicyAgent
from air_hockey_agent.agents.dummy_agent import DummyAgent
from air_hockey_agent.agents.agents import DefendAgent, RepelAgent
from air_hockey_agent.agents.agent_sb3 import AgentSB3
#from air_hockey_agent.agents.agent_sb3_old_env import AgentSB3OldEnv

import pickle


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    :param env_info: The environment information
    :return: Either Agent ot Policy
    """
    # with open("env_info_single_agent/env_infos.pkl", "rb") as fp:
    #    env_info_hit, env_info_defend = pickle.load(fp)

    '''if "hit" in env_info["env_name"]:
        return PolicyAgent(env_info, **kwargs, agent_id=1, task="hit")
    if "defend" in env_info["env_name"]:
        return DefendAgent(env_info, **kwargs)
    if "prepare" in env_info["env_name"]:
        return PolicyAgent(env_info, **kwargs, agent_id=1, task="prepare")'''

    # Default return, in case of tournament
    return HierarchicalAgent(env_info, **kwargs)
