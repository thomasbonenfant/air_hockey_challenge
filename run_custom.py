from air_hockey_challenge.framework.evaluate_tournament import run_tournament
from air_hockey_agent.agent_builder import build_agent, build_defend_agent

from pathlib import Path
import yaml


def run_custom():
    # path for configuration of the first agent
    agent_config_1_path = Path(__file__).parent.joinpath("air_hockey_agent/agent_config.yml")

    # use same configuration for both agents, agent plays against itself
    agent_config_2_path = agent_config_1_path

    # Load configuration files
    with open(agent_config_1_path) as stream:
        agent_config_1 = yaml.safe_load(stream)

    with open(agent_config_2_path) as stream:
        agent_config_2 = yaml.safe_load(stream)

    run_tournament(build_agent_1=build_agent, build_agent_2=build_agent, agent_2_kwargs=agent_config_2, **agent_config_1)


if __name__ == "__main__":
    run_custom()
