from air_hockey_challenge.framework.agent_base import AgentBase
from baseline.baseline_agent.tactics import *
from air_hockey_challenge.utils.kinematics import forward_kinematics
from baseline.baseline_agent.baseline_agent import BaselineAgent

def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    :param env_info: The environment information
    :return: Either Agent ot Policy
    """
    if "hit" in env_info["env_name"]:
        return BaselineAgent(env_info, **kwargs, agent_id=1, only_tactic="hit")
    if "defend" in env_info["env_name"]:
        return BaselineAgent(env_info, **kwargs, agent_id=1, only_tactic="defend")
    if "prepare" in env_info["env_name"]:
        return BaselineAgent(env_info, **kwargs, agent_id=1, only_tactic="prepare")

    return BaselineAgent(env_info, **kwargs)