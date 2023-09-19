from air_hockey_agent.agents.agents import HitAgent, DefendAgent, PrepareAgent
from air_hockey_agent.agents.rule_based_agent import PolicyAgent
from air_hockey_agent.agents.dummy_agent import DummyAgent
import pickle
def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    :param env_info: The environment information
    :return: Either Agent ot Policy
    """

    with open("envs/env_info_single_agent/env_infos.pkl", "rb") as fp:
        env_info_hit, env_info_defend = pickle.load(fp)

    if "hit" in env_info["env_name"]:
        return HitAgent(env_info_hit, **kwargs)
    if "defend" in env_info["env_name"]:
        return DefendAgent(env_info, **kwargs)
        # return DummyAgent(env_info, agent_id=1, **kwargs)
    if "prepare" in env_info["env_name"]:
        return PolicyAgent(env_info, **kwargs, agent_id=1, task="prepare")
    if "tournament" in env_info["env_name"]:
        if kwargs['policy'] == 'hit_oac':
            return HitAgent(env_info_hit, **kwargs)
        elif kwargs['policy'] == 'hit_rb':
            return PolicyAgent(env_info, **kwargs, agent_id=1, task="hit")
        elif kwargs['policy'] == 'defend':
            agent =  DefendAgent(env_info, **kwargs)
            return agent
