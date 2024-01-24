import copy

from MagicRL.policies.switcher_policy import SwitcherPolicy
from MagicRL.envs.airhockey import AirHockey, AirHockeyDouble
from MagicRL.algorithms.samplers import TrajectorySampler
from MagicRL.data_processors import IdentityDataProcessor
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import sys
from pympler.asizeof import asizeof, asized

EPISODES=1
STEPS=100
HORIZON=200

wrapped_env = AirHockeyDouble(opponent_delay=0)
env_info = wrapped_env.env_info
env = AirHockey(render=False, horizon=500, gamma=0.997)


class PolicyMaker:
    def __init__(self, env_info):
        self.thetas = None
        self.env_info = env_info

    def set_parameters(self, thetas):
        self.thetas = thetas

    def __call__(self, *args):
        pol = SwitcherPolicy(env_info=self.env_info)
        if self.thetas is not None:
            pol.set_parameters(self.thetas)
        pol.reset()
        return pol


pol = SwitcherPolicy(env_info=env.env.env_info)
hp = np.array([0.3, 4, 0.2])

pol.reset()

print(asizeof(pol))

#env.step(np.random.random((2,7)))



# for i in tqdm(range(EPISODES)):
#     pol.reset()
#     env.reset()
#
#     t = 0
#     done = False
#
#     while t < HORIZON and not done:
#         state = env.state
#         env.step(pol.draw_action(state))
#
# sampler = TrajectorySampler(env, pol, IdentityDataProcessor())
# print(sampler.collect_trajectory(hp))


