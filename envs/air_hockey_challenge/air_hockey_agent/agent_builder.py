from air_hockey_agent.agents.agents import HitAgent, DefendAgent, PrepareAgent

def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    :param env_info: The environment information
    :return: Either Agent ot Policy
    """
    if "hit" in env_info["env_name"]:
        return HitAgent(env_info, **kwargs)
    if "defend" in env_info["env_name"]:
        return DefendAgent(env_info, **kwargs)
        # return DummyAgent(env_info, agent_id=1, **kwargs)
    if "prepare" in env_info["env_name"]:
        return PrepareAgent(env_info, **kwargs)
