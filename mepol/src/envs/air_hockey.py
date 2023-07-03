import numpy as np
import copy
from numpy.linalg import pinv
import gym
from gym import spaces
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian


class GymAirHockey(gym.Env):
    def __init__(self, env_name, interpolation_order=3, custom_reward_function=None,
                 task_space=True, task_space_vel=True, use_delta_pos=True, delta_dim=0.1, use_puck_distance=True, **kwargs):
        """
        Args:
            task_space: if True changes the action space to x,y
        """

        self.challenge_env = AirHockeyChallengeWrapper(env_name,
                                                       interpolation_order=interpolation_order,
                                                       custom_reward_function=custom_reward_function)

        self.env_info = env_info = self.challenge_env.env_info

        n_joints = env_info['robot']['n_joints']
        self.gamma = env_info['rl_info'].gamma
        self.large_reward = 1000

        self.ee_desired_height = env_info['robot']['ee_desired_height'] - 0.1
        self.world2robot_transf = env_info['robot']['base_frame'][0]
        self.mj_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.mj_data = copy.deepcopy(env_info['robot']['robot_data'])

        puck_pos_ids = env_info['puck_pos_ids']
        puck_vel_ids = env_info['puck_vel_ids']
        joint_pos_ids = env_info['joint_pos_ids']
        joint_vel_ids = env_info['joint_vel_ids']

        table_length = self.env_info['table']['length']
        table_width = self.env_info['table']['width']

        self.joint_min_pos = joint_min_pos = env_info['rl_info'].observation_space.low[joint_pos_ids]
        self.joint_max_pos = joint_max_pos = env_info['rl_info'].observation_space.high[joint_pos_ids]
        self.joint_min_vel = joint_min_vel = env_info['rl_info'].observation_space.low[joint_vel_ids]
        self.joint_max_vel = joint_max_vel = env_info['rl_info'].observation_space.high[joint_vel_ids]

        mallet_radius = env_info['mallet']['radius']
        puck_radius = env_info['puck']['radius']

        self.max_ee_pos_action = np.array([0, (table_width / 2) - mallet_radius])
        self.min_ee_pos_action = np.array([(-table_length / 2) + mallet_radius, (-table_width / 2) + mallet_radius])

        # to calculate joint velocity
        self.old_joint_pos = np.zeros(n_joints)

        self.old_puck_vel = np.zeros(2)  # to check if puck was hit

        self.task_space = task_space
        self.task_space_vel = task_space_vel
        self.use_delta_pos = use_delta_pos
        self.delta_dim = delta_dim
        self.use_puck_distance = use_puck_distance
        self.prev_ee_pos = None

        if self.task_space:

            if self.task_space_vel:
                action_space_dim = 4    # x,y, dx, dy

                max_action = np.hstack((self.max_ee_pos_action, [1, 1]))
                min_action = np.hstack((self.min_ee_pos_action, [-1, -1]))
                self.action_space = spaces.Box(low=min_action, high=max_action,
                                               shape=(action_space_dim,), dtype=np.float32)
            else:
                action_space_dim = 2
                if self.use_delta_pos:
                    max_action = np.array([self.delta_dim, self.delta_dim])
                    min_action = -max_action
                else:
                    max_action = self.max_ee_pos_action
                    min_action = self.min_ee_pos_action
                self.action_space = spaces.Box(low=min_action, high=max_action,
                                               shape=(action_space_dim,), dtype=np.float32)
        else:
            action_space_dim = 2 * n_joints
            low_action = np.hstack((joint_min_pos, joint_min_vel))
            high_action = np.hstack((joint_max_pos, joint_max_vel))
            self.action_space = spaces.Box(low=low_action, high=high_action,
                                           shape=(action_space_dim,), dtype=np.float32)

        # observation space specification

        puck_pos_low = np.array([-table_length / 2 + puck_radius, -table_width / 2 + puck_radius])
        puck_pos_high = np.array([table_length / 2 - puck_radius, table_width / 2 - puck_radius])
        ee_pos_low = np.array([-table_length / 2 + mallet_radius, -table_width / 2 + mallet_radius])
        ee_pos_high = np.array([table_length / 2 - mallet_radius, table_width / 2 - mallet_radius])
        puck_vel_low = env_info['rl_info'].observation_space.low[puck_vel_ids][:2]
        puck_vel_high = env_info['rl_info'].observation_space.high[puck_vel_ids][:2]
        ee_vel_low = env_info['rl_info'].observation_space.low[puck_vel_ids][:2]
        ee_vel_high = env_info['rl_info'].observation_space.high[puck_vel_ids][:2]

        ee_puck_dist_low = puck_pos_low - ee_pos_high
        ee_puck_dist_high = puck_pos_high - ee_pos_low

        if self.task_space:
            self.num_features = 2*2 + 2*n_joints + 2*2 # puck_x, puck_y, dpuck_x, dpuck_y, joint_pos, joint_vel, ee_x, ee_y, deex, deey
            if self.use_puck_distance:
                obs_low = np.hstack(
                    [puck_pos_low, puck_vel_low, joint_min_pos, joint_min_vel, ee_puck_dist_low, ee_vel_low])
                obs_high = np.hstack(
                    [puck_pos_high, puck_vel_high, joint_max_pos, joint_max_vel, ee_puck_dist_high, ee_vel_high])
            else:
                obs_low = np.hstack(
                    [puck_pos_low, puck_vel_low, joint_min_pos, joint_min_vel, ee_pos_low, ee_vel_low])
                obs_high = np.hstack([puck_pos_high, puck_vel_high, joint_max_vel, ee_pos_high, ee_vel_high])
        else:
            self.num_features = 2*2 + 2*n_joints
            obs_low = np.hstack([puck_pos_low, puck_vel_low, joint_min_pos, joint_min_vel])
            obs_high = np.hstack([puck_pos_high, puck_vel_high, joint_max_vel, joint_max_vel])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(self.num_features,), dtype=np.float32)
        self.obs_mean = 0.5 * (obs_low + obs_high)

    def convert_obs(self, obs):
        # change coordinates to world's reference frame coordinates
        puck_pos, _ = robot_to_world(self.challenge_env.env_info['robot']['base_frame'][0], obs[self.challenge_env.env_info['puck_pos_ids']])
        puck_pos = puck_pos[:2]

        puck_vel = obs[self.challenge_env.env_info['puck_vel_ids']]
        puck_vel = puck_vel[:2]

        joint_pos = obs[self.challenge_env.env_info['joint_pos_ids']]
        joint_vel = obs[self.challenge_env.env_info['joint_vel_ids']]

        if self.task_space:
            ee_pos, _ = forward_kinematics(self.mj_model, self.mj_data, joint_pos)
            ee_pos, _ = robot_to_world(self.world2robot_transf, ee_pos)
            ee_pos = ee_pos[:2]

            #saves ee_pos for delta pos calculation
            self.prev_ee_pos = ee_pos
            ee_vel = self._apply_forward_velocity_kinematics(joint_pos, joint_vel)[:2]

            if self.use_puck_distance:
                obs = np.hstack((puck_pos, puck_vel, joint_pos, joint_vel, (puck_pos - ee_pos), ee_vel))
            else:
                obs = np.hstack((puck_pos, puck_vel, joint_pos, joint_vel, ee_pos, ee_vel))
        else:
            obs = np.hstack((puck_pos, puck_vel, joint_pos, joint_vel))

        return obs

    def reset(self, seed=None, options=None):
        obs = self.challenge_env.reset()

        self.old_joint_pos = obs[self.env_info['joint_pos_ids']]

        obs = self.convert_obs(obs)

        self.old_puck_vel = obs[[2, 3]]
        #assert np.array_equal(self.old_puck_vel, [0, 0])

        return obs

    def step(self, action):

        action = self._convert_action(action)

        obs, reward, done, info = self.challenge_env.step(action)

        # calculate reward
        if info['success']:
            reward = self.large_reward

        self.old_joint_pos = obs[self.challenge_env.env_info['joint_pos_ids']]

        obs = self.convert_obs(obs)

        puck_vel = obs[[2, 3]]

        if np.array_equal(self.old_puck_vel, [0, 0]) and not np.array_equal(puck_vel, [0, 0]):
            info['hit'] = True
        else:
            info['hit'] = False

        #check if the puck has hit the board
        info['hit_board'] = self._check_hit_board(obs)

        # update old_puck_vel
        self.old_puck_vel = puck_vel

        return obs, reward, done, False, info

    def _check_hit_board(self, obs):
        puck_pos = obs[[0, 1]]
        env_info = self.env_info
        tolerance = 5e-3

        puck_radius = env_info['puck']['radius']
        table_length = env_info['table']['length']
        table_width = env_info['table']['width']
        goal_width = env_info['table']['goal_width']

        x_constraint = np.abs(puck_pos[0]) - (table_length / 2 - puck_radius) + tolerance
        y_constraint = np.abs(puck_pos[1]) - (table_width / 2 - puck_radius) + tolerance
        goal_constraint = np.abs(puck_pos[1]) - (goal_width / 2 )


        return (x_constraint >= 0 and goal_constraint > 0) or y_constraint >= 0

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

    def solve_QP(self):
        pass

    def _apply_forward_velocity_kinematics(self, joint_pos, joint_vel):

        jac = jacobian(self.mj_model, self.mj_data, joint_pos)[:3]  # last part of the matrix is about rotation.
        ee_vel = jac @ joint_vel

        return ee_vel

    def _convert_action(self, action):

        action = np.array(action)

        if self.task_space:
            if self.task_space_vel:
                ee_pos_action = action[:2]
            else:
                ee_pos_action = action

            # if the action is only a delta position update the resulting ee position action
            if self.use_delta_pos:
                # scale delta action
                ee_pos_action *= self.delta_dim

                ee_pos_action = self.prev_ee_pos + ee_pos_action

            # following flag is used to know if joint accelerations need to be set to 0 if we are going on border
            has_to_clip = (ee_pos_action > self.min_ee_pos_action).all() and (ee_pos_action < self.max_ee_pos_action).all()

            # clip action into table
            ee_pos_action = np.clip(ee_pos_action, a_min=self.min_ee_pos_action, a_max=self.max_ee_pos_action)

            # make action 3D
            ee_pos_action = np.hstack((ee_pos_action, self.ee_desired_height))

            joint_pos_action = self._apply_inverse_kinematics(ee_pos_action)

            if self.task_space_vel:
                ee_vel_action = np.hstack((action[2:], 0))
                jac = jacobian(self.mj_model, self.mj_data, joint_pos_action)[:3]
                joint_vel_action = pinv(jac) @ ee_vel_action

            else:
                if not has_to_clip:
                    joint_vel_action = (joint_pos_action - self.old_joint_pos) / self.challenge_env.env_info['dt']
                else:
                    joint_vel_action = np.zeros((self.env_info['robot']['n_joints']))

            # creates (2,3) array
            action = np.vstack((joint_pos_action, joint_vel_action))

        else:
            action = np.reshape(action, (2, self.env_info['robot']['n_joints']))

        return action

class GymAirHockeyAcceleration(gym.Env):
    def __init__(self, env_name, interpolation_order=3, custom_reward_function=None, max_acceleration=1, **kwargs):
        self.wrapped_env = GymAirHockey(env_name,
                                        interpolation_order=interpolation_order,
                                        custom_reward_function=custom_reward_function,
                                        delta_dim=1,
                                        use_puck_distance=True,
                                        task_space=True,
                                        task_space_vel=False,
                                        use_delta_pos=True)

        self.max_acceleration = max_acceleration
        self.curr_ee_vel = np.array([0,0])
        self.dt = self.wrapped_env.env_info['dt']

        low_acc = np.array([0,0])
        high_acc = np.array([max_acceleration, 2*np.pi])
        self.action_space = spaces.Box(low=low_acc, high=high_acc)
        self.observation_space = self.wrapped_env.observation_space
        self.num_features = self.wrapped_env.num_features

    def reset(self, seed=None, options=None):
        obs = self.wrapped_env.reset()

        self.curr_ee_vel = obs[-2:]

        return obs

    def step(self, polar_action):

        x_cart = polar_action[0]*np.cos(polar_action[1])
        y_cart = polar_action[0]*np.sin(polar_action[1])
        cartesian_acc = np.array([x_cart, y_cart])

        delta_ee_pos = self.curr_ee_vel * self.dt + 0.5 * cartesian_acc * (self.dt ** 2)
        next_ee_vel = self.curr_ee_vel + cartesian_acc * self.dt

        # action = np.hstack((delta_ee_pos, next_ee_vel))
        action = delta_ee_pos

        obs, reward, done, truncated, info = self.wrapped_env.step(action)

        self.curr_ee_vel = obs[-2:]

        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        self.wrapped_env.render(mode)
