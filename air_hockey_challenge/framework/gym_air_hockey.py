import numpy as np
import gym
from gym import spaces
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.utils.transformations import robot_to_world
from air_hockey_challenge.utils.kinematics import forward_kinematics

class GymAirHockey(gym.Env):
    def __init__(self, use_puck_distance = True):
        '''
        Args:
            use_puck_distance: whether add end effector's distance from the puck in the observation
        '''
        self.challenge_env = AirHockeyChallengeWrapper('3dof-hit')
        n_joints = self.challenge_env.env_info['robot']['n_joints']

        self.action_space = spaces.Box(low=-3, high=3, shape=(2*n_joints,), dtype=np.float32)
        self.num_features = 10

        self.use_puck_distance = use_puck_distance

        if self.use_puck_distance:
            self.num_features += 2

        self.observation_space = spaces.Box(low = -10, high=10, shape=(self.num_features,), dtype=np.float32)

    def convert_obs(self, obs):
        # change coordinates to world's reference frame coordinates
        puck_pos, _ = robot_to_world(self.challenge_env.env_info['robot']['base_frame'][0], obs[:3])
        puck_pos = puck_pos[:2]

        puck_vel = obs[self.challenge_env.env_info['joint_vel_ids']]
        joint_pos = obs[self.challenge_env.env_info['joint_pos_ids']]
        joint_vel = obs[self.challenge_env.env_info['joint_vel_ids']]

        obs = np.hstack((puck_pos[:2], puck_vel[:2], joint_vel, joint_pos))

        if self.use_puck_distance:
            mj_model = self.challenge_env.env_info['robot']['robot_model']
            mj_data = self.challenge_env.env_info['robot']['robot_data']
            ee_pos, _ = forward_kinematics(mj_model,mj_data, joint_pos)
            ee_pos = ee_pos[0]

            puck_distance = puck_pos - ee_pos
            obs = np.hstack((obs, puck_distance))


        return obs



    def reset(self):
        return self.convert_obs(self.challenge_env.reset())
    
    def step(self, action):
        action = np.reshape(action, (2,3))
        obs, reward, done, info = self.challenge_env.step(action)

        return self.convert_obs(obs), reward, done, False, info

    def render(self, mode="human"):
        self.challenge_env.render()
    
    def close(self):
        pass


    


    
    



        