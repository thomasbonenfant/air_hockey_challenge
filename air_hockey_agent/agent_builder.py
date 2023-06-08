from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.agents.rule_based_agent import PolicyAgent


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

    # TODO might insert thetas here and pass them to the agent builder

    agent = PolicyAgent(env_info, **kwargs)

    return agent
