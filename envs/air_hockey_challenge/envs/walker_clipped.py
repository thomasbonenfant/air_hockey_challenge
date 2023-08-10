import numpy as np
from gym.envs.mujoco import Walker2dEnv
from gym import spaces


class WolkerClipped(Walker2dEnv):

    def __init__(self):
        self.bounds = [-120, 120]
        super().__init__()
        s = self.reset()
        low = np.ones_like(s) * self.bounds[0]
        high = np.ones_like(s) * self.bounds[1]
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        






