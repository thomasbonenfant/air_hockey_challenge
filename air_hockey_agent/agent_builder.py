from air_hockey_agent.agents import DummyAgent, SimpleDefendingAgent


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

    agent = DummyAgent(env_info, **kwargs)

    if 'agent' in kwargs:
        if kwargs['agent'] == 'dummy-agent':
            agent = DummyAgent(env_info, **kwargs)
        elif kwargs['agent'] == 'defend-agent':
            agent = SimpleDefendingAgent(env_info, **kwargs)


    return agent
