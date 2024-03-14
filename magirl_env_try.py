import copy

from MagicRL.policies.switcher_policy import SwitcherPolicy
from MagicRL.envs.airhockey import AirHockey, AirHockeyDouble

from tqdm import tqdm
import numpy as np


EPISODES=1
GAMMA=0.999
wrapped_env = AirHockeyDouble(opponent_delay=0)
env_info = wrapped_env.env_info
env = AirHockey(render=True, horizon=5000, gamma=GAMMA)


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

pol.set_parameters(np.array([0.3, 0.2]))

pol.reset()

for i in tqdm(range(EPISODES)):
    pol.reset()
    env.reset()

    t = 0
    done = False

    while t < env.horizon and not done:
        state = env.state
        obs, rew, done, _ = env.step(pol.draw_action(state))
        if rew != 0:
            print(f'STEP: {t}\tRETURN: {rew * GAMMA ** t}')
        t += 1





