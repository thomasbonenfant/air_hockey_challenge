import numpy as np
import gym
from gym import spaces
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper

class GymAirHockey(gym.Env):
    def __init__(self, *args, **kwargs):
        self.challenge_env = AirHockeyChallengeWrapper('3dof-hit')
        n_joints = self.challenge_env.env_info['robot']['n_joints']

        self.action_space = spaces.Box(low=-3, high=3, shape=(2*n_joints,), dtype=np.float32)
        self.num_features = 12
        self.observation_space = spaces.Box(low = -10, high=10, shape=(self.num_features,), dtype=np.float32)

    def reset(self):
        return self.challenge_env.reset()
    
    def step(self, action):
        action = np.reshape(action, (2,3))
        obs, reward, done, info = self.challenge_env.step(action)
        return obs, reward, done, False, info

    def render(self, mode="human"):
        self.challenge_env.render()
    
    def close(self):
        pass


    


    
    



        