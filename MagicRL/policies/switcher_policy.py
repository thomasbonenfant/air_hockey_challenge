from MagicRL.policies import BasePolicy
from air_hockey_agent.agents.hierarchical_agent import HierarchicalAgent


class SwitcherPolicy(BasePolicy):
    """
    Adapter Class For Hierarchical Agent
    """

    def compute_score(self, state, action):
        pass

    def reduce_exploration(self):
        pass

    def __init__(self, env_info):
        super().__init__()
        self.wrapped_pol = HierarchicalAgent(env_info, agent_id=1)
        self.tot_params = self.wrapped_pol.tot_params

    def draw_action(self, state):
        return self.wrapped_pol.draw_action(observation=state)

    def set_parameters(self, thetas):
        self.wrapped_pol.set_parameters(thetas)

    def reset(self):
        self.wrapped_pol.reset()
