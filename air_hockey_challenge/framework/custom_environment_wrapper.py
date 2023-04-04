from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.utils.kinematics import inverse_kinematics, forward_kinematics, jacobian  
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot

import numpy as np
import pickle
import datetime

class CustomEnvironmentWrapper(AirHockeyChallengeWrapper):
    ''' This wrapper extends the original AirHockeyChallengeWrapper in order to use the end effector position
    in the world's frame as action.

    It needs to be instanciated in air_hockey_challenge/framework/evaluate_agent.py in place of AirHockeyChallenge
    '''

    def __init__(self, env):
        super().__init__(env)

        self.world2robot_transf = self.env_info['robot']['base_frame'][0]
        self.old_joint_pos = None

        self.log_data = False

        self.steps = 0

        if self.log_data:
            self.joint_action = []
            self.ee_action = []

            self.joint_pos = []
            self.joint_vel = []
            self.joint_jerk = []
            self.joint_acc = []

            self.ee_pos = []
            self.ee_vel = []
            self.ee_acc = []

        #add new observation info 
        #new_obs_idx = 12
        self.env_info['ee_pos_ids'] = [0, 1, 2]
        self.env_info['ee_vel_ids'] = [3, 4, 5]
        self.env_info['new_puck_pos_ids'] = [6, 7, 8]
        self.env_info['new_puck_vel_ids'] = [9, 10, 11]

            
    def step(self, ee_pos_action):
        '''
        param action: size (3,) ndarray describing end effector position
        '''

        self.steps += 1
        action_joints = self._apply_inverse_kinematics(ee_pos_action)

        joint_velocities = (action_joints - self.old_joint_pos) / self.env_info['dt']
       
        self.old_joint_pos = action_joints #update stored joint position for future velocity computation

        action = np.vstack([action_joints, joint_velocities])

        obs, reward, done, info = super().step(action) # calls original step function


        if self.log_data:
            self.joint_action.append(action)
            self.ee_action.append(ee_pos_action)

            self.joint_pos.append(obs[self.env_info['joint_pos_ids']])
            self.joint_vel.append(obs[self.env_info['joint_vel_ids']])

            j_acc = (np.array(self.joint_vel[-1]) - np.array(self.joint_vel[-2])) / self.env_info['dt']
            self.joint_acc.append(j_acc.tolist())

            self.joint_jerk.append(info['jerk'])

            self.ee_pos.append(self._apply_forward_kinematics(self.joint_pos[-1]))
            vel = (np.array(self.ee_pos[-1]) - np.array(self.ee_pos[-2])) / self.env_info['dt']

            self.ee_vel.append(vel.tolist())

            acc = (np.array(self.ee_vel[-1]) - np.array(self.ee_vel[-2])) / self.env_info['dt']

            self.ee_acc.append(acc.tolist())

            if done or self.steps == self.info.horizon:
                log_result = (self.joint_action, self.ee_action, self.joint_pos, self.joint_vel, self.joint_acc, self.ee_pos, self.ee_vel, self.ee_acc, self.joint_jerk)
                curr_time = datetime.datetime.now()
                with open(f'logs/custom_log_{curr_time.strftime("%Y-%m-%d_%H-%M-%S")}.pkl','wb') as f:
                    pickle.dump(log_result, f)


        obs = self.convert_obs(obs) #updates obs to have (ee_pos, ee_vel, puck_pos, puck_vel)
        return obs, reward, done, info

    def convert_obs(self, obs):
        '''
        changes the obs array to have only ee_pos, ee_vel, puck_pos, puck_vel
        
        '''
        joint_pos = obs[self.env_info['joint_pos_ids']]
        joint_vel = obs[self.env_info['joint_vel_ids']]
        puck_pos = obs[self.env_info['puck_pos_ids']]
        puck_vel = obs[self.env_info['puck_vel_ids']]
    
        ee_position = self._apply_forward_kinematics(joint_pos)
        ee_velocity = self._apply_forward_velocity_kinematics(joint_pos, joint_vel)

        #obs = np.concatenate((obs, ee_position, ee_velocity))
        obs = np.concatenate((ee_position, ee_velocity, puck_pos, puck_vel))

        return obs
        
    def _apply_forward_velocity_kinematics(self, joint_pos, joint_vel):
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']

        jac = jacobian(mj_model, mj_data, joint_pos)[:3] #last part of the matrix is about rotation. no need for it
        ee_vel = jac @ joint_vel

        return ee_vel

    def _apply_inverse_kinematics(self, ee_pos_robot_frame):
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']
        position_robot_frame, rotation = world_to_robot(self.world2robot_transf, ee_pos_robot_frame)
        success, action_joints = inverse_kinematics(mj_model, mj_data, position_robot_frame) # inverse_kinematics uses robot's frame for coordinates
        return action_joints

    def _apply_forward_kinematics(self, joint_pos):
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']
        ee_pos_robot_frame, rotation = forward_kinematics(mj_model, mj_data, joint_pos)
        ee_pos_world_frame, rotation = robot_to_world(self.world2robot_transf, ee_pos_robot_frame)
        return ee_pos_world_frame


    def reset(self, state=None):
        obs = super().reset()

        self.steps = 0

        self.old_joint_pos = obs[self.env_info['joint_pos_ids']]

        if self.log_data:
            self.joint_pos.append(self.old_joint_pos)
            self.joint_vel.append(np.zeros(self.env_info['robot']['n_joints']))
            self.joint_jerk.append(np.zeros(self.env_info['robot']['n_joints']))
            self.joint_acc.append(np.zeros(self.env_info['robot']['n_joints']))

            self.ee_pos.append(self._apply_forward_kinematics(self.old_joint_pos))
            self.ee_vel.append([0 for i in range(self.env_info['robot']['n_joints'])])
            self.ee_acc.append([0 for i in range(self.env_info['robot']['n_joints'])])

        obs = self.convert_obs(obs)
        return obs