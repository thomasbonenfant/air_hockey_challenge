from MagicRL.policies.base_policy import BasePolicy
import numpy as np
import copy

class RandomPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

        self.tot_params = 1
        self.thetas = np.array([0])
    def draw_action(self, state):
        return np.random.random((2,7))

    def set_parameters(self, thetas):
        self.thetas = copy.deepcopy(thetas)

    def compute_score(self, state, action):
        pass

    def reduce_exploration(self):
        pass