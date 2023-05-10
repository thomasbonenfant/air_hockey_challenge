import numpy as np
import copy
from numpy.linalg import pinv
import gym
from gym import spaces
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian


class GymAirHockey(gym.Env):
    def __init__(self, action_type = 'position-velocity', interpolation_order=3, custom_reward_function=None,
                 task_space=True, task_space_vel=True, use_delta_pos=True):
        '''
        Args:
            task_space: if True changes the action space to x,y
        '''
        self.challenge_env = AirHockeyChallengeWrapper('3dof-hit', action_type=action_type,
                                                       interpolation_order=interpolation_order,
                                                       custom_reward_function=custom_reward_function)

        self.env_info = env_info = self.challenge_env.env_info

        n_joints = self.challenge_env.env_info['robot']['n_joints']
        self.gamma = env_info['rl_info'].gamma



        self.world2robot_transf = self.challenge_env.env_info['robot']['base_frame'][0]
        self.mj_model = copy.deepcopy(self.challenge_env.env_info['robot']['robot_model'])
        self.mj_data = copy.deepcopy(self.challenge_env.env_info['robot']['robot_data'])

        # to calculate joint velocity
        self.old_joint_pos = np.zeros(n_joints)

        self.task_space = task_space
        self.task_space_vel = task_space_vel
        self.use_delta_pos = use_delta_pos
        self.prev_ee_pos = None

        if self.task_space:
            if self.task_space_vel:
                action_space_dim = 4    # x,y, dx, dy

                max_action = np.array([0.974, 0.519, 10,10])
                self.action_space = spaces.Box(low=-max_action, high=max_action,
                                               shape=(action_space_dim,), dtype=np.float32)
            else:
                action_space_dim = 2
                if self.use_delta_pos:
                    max_action = np.array([0.1, 0.1])
                else:
                    max_action = np.array([0.974, 0.519])
                self.action_space = spaces.Box(low=-max_action, high=max_action,
                                               shape=(action_space_dim,), dtype=np.float32)
        else:
            action_space_dim = 2 * n_joints
            self.action_space = spaces.Box(low=-3, high=3,
                                           shape=(action_space_dim,), dtype=np.float32)

        if self.task_space:
            self.num_features = 14 # puck_x, puck_y, dpuck_x, dpuck_y, ee_x, ee_y, deex, deey
        else:
            self.num_features = 10

        self.observation_space = spaces.Box(low = -10, high=10, shape=(self.num_features,), dtype=np.float32)

    def convert_obs(self, obs):
        # change coordinates to world's reference frame coordinates
        puck_pos, _ = robot_to_world(self.challenge_env.env_info['robot']['base_frame'][0], obs[self.challenge_env.env_info['puck_pos_ids']])
        puck_pos = puck_pos[:2]

        puck_vel = obs[self.challenge_env.env_info['puck_vel_ids']]

        joint_pos = obs[self.challenge_env.env_info['joint_pos_ids']]
        joint_vel = obs[self.challenge_env.env_info['joint_vel_ids']]

        obs = np.hstack((puck_pos[:2], puck_vel[:2], joint_pos, joint_vel))

        if self.task_space:
            ee_pos, _ = forward_kinematics(self.mj_model, self.mj_data, joint_pos)
            ee_pos, _ = robot_to_world(self.world2robot_transf, ee_pos)
            self.prev_ee_pos = ee_pos
            ee_vel = self._apply_forward_velocity_kinematics(joint_pos, joint_vel)[:2]

            obs = np.hstack((obs, ee_pos[:2], ee_vel))

        return obs



    def reset(self):
        obs =  self.challenge_env.reset()

        self.old_joint_pos = obs[self.challenge_env.env_info['joint_pos_ids']]

        return self.convert_obs(obs)
    
    def step(self, action):

        action = self._convert_action(action)

        obs, reward, done, info = self.challenge_env.step(action)

        self.old_joint_pos = obs[self.challenge_env.env_info['joint_pos_ids']]

        return self.convert_obs(obs), reward, done, False, info

    def render(self, mode="human"):
        self.challenge_env.render()
    
    def close(self):
        pass

    def _apply_inverse_kinematics(self, ee_pos_robot_frame):
        env_info = self.challenge_env.env_info

        position_robot_frame, rotation = world_to_robot(self.world2robot_transf, ee_pos_robot_frame)
        success, action_joints = inverse_kinematics(self.mj_model, self.mj_data,
                                                    position_robot_frame)
        return action_joints

    def _apply_forward_velocity_kinematics(self, joint_pos, joint_vel):

        jac = jacobian(self.mj_model, self.mj_data, joint_pos)[:3]  # last part of the matrix is about rotation.
        ee_vel = jac @ joint_vel

        return ee_vel

    def _convert_action(self, action):
        if self.task_space:
            if self.task_space_vel:
                ee_pos_action = np.hstack((action[:2], 1))
            else:
                ee_pos_action = np.hstack((action, 0))

            # if the action is only a delta position update the resulting ee position action
            if self.use_delta_pos:
                ee_pos_action = self.prev_ee_pos + ee_pos_action
            joint_pos_action = self._apply_inverse_kinematics(ee_pos_action)

            if self.task_space_vel:
                ee_vel_action = np.hstack((action[2:], 0))
                jac = jacobian(self.mj_model, self.mj_data, joint_pos_action)[:3]
                joint_vel_action = pinv(jac) @ ee_vel_action

            else:
                joint_vel_action = (joint_pos_action - self.old_joint_pos) / self.challenge_env.env_info['dt']

            # creates (2,3) array
            action = np.vstack((joint_pos_action, joint_vel_action))

        else:
            action = np.reshape(action, (2, 3))

        return action

    


    
    



        