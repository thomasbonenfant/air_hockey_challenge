import numpy as np
import gym


class Ant(gym.Wrapper):

    def __init__(self):
        ant_env = gym.make('Ant-v4',
                           exclude_current_positions_from_observation=False)
        super().__init__(ant_env)

        self.num_features = self.state_vector().shape[0]

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        return self.state_vector(), reward, done, truncated, info

    def reset(self):
        super().reset()
        return self.state_vector()
