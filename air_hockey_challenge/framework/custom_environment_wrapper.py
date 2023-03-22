from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.utils.kinematics import inverse_kinematics
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot

import numpy as np

class CustomEnvironmentWrapper(AirHockeyChallengeWrapper):
    ''' This wrapper extends the original AirHockeyChallengeWrapper in order to use the end effector position
    in the world's frame as action.

    It needs to be instanciated in air_hockey_challenge/framework/evaluate_agent.py in place of AirHockeyChallenge
    '''

    def __init__(self, env):
        super().__init__(env)

        self.world2robot_transf = self.env_info['robot']['base_frame'][0]
            

    def step(self, action):
        '''
        param action: size (3,) ndarray describing end effector position
        
        '''
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']

        position_robot_frame, rotation = world_to_robot(self.world2robot_transf, action)

        success, action_joints = inverse_kinematics(mj_model, mj_data, position_robot_frame) # inverse_kinematics uses robot's frame for coordinates

        joint_velocities = np.zeros((1,self.env_info['robot']['n_joints'])) # for now sets joint velocities as 0

        action = np.vstack([action_joints, joint_velocities])

        return super().step(action) # calls original step function