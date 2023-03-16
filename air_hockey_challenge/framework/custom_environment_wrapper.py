from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.utils.kinematics import inverse_kinematics

import numpy as np

class CustomEnvironmentWrapper(AirHockeyChallengeWrapper):
    #
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        '''
        param action: size (3,) ndarray describing end effector position
        
        '''
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']


        success, action = inverse_kinematics(mj_model, mj_data, action)

        joint_velocities = np.zeros((1,self.env_info['robot']['n_joints'])) #for now sets joint velocities as 0

        action = np.vstack([action, joint_velocities])

        return super().step(action) #calls original step function