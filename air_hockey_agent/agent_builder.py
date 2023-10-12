from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.agents.hierarchichal_agent import HierarchichalAgent
from air_hockey_agent.agents.rule_based_agent import PolicyAgent
from air_hockey_agent.agents.dummy_agent import DummyAgent
from air_hockey_agent.agents.agents import DefendAgent
import pickle


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    :param env_info: The environment information
    :return: Either Agent ot Policy
    """
    #with open("env_info_single_agent/env_infos.pkl", "rb") as fp:
    #    env_info_hit, env_info_defend = pickle.load(fp)

    if "hit" in env_info["env_name"]:
        return PolicyAgent(env_info, **kwargs, agent_id=1, task="hit")
    if "defend" in env_info["env_name"]:
        # return PolicyAgent(env_info, **kwargs, agent_id=1, task="defend")
        return DefendAgent(env_info, **kwargs)
    if "prepare" in env_info["env_name"]:
        return PolicyAgent(env_info, **kwargs, agent_id=1, task="prepare")

    #env_info['opponent_ee_ids'] = []
    # Default return, in case of tournament
    return HierarchichalAgent(env_info, **kwargs)


def build_defend_agent(env_info, **kwargs):
    #with open("env_info_single_agent/env_infos.pkl", "rb") as fp:
    #    env_info_hit, env_info_defend = pickle.load(fp)

    #return DefendAgent(env_info_defend, **kwargs)
    return PolicyAgent(env_info, **kwargs, agent_id=1, task="defend")
