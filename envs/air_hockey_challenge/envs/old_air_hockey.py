import copy
import io
import json
import os
import errno
from datetime import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from matplotlib.patches import Ellipse

from envs.air_hockey_challenge.air_hockey_challenge.framework.air_hockey_challenge_wrapper import \
    AirHockeyChallengeWrapper
from envs.air_hockey_challenge.air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics
from envs.air_hockey_challenge.baseline.baseline_agent.system_state import SystemState

PENALTY_POINTS = {"joint_pos_constr": 2, "ee_constr": 3, "joint_vel_constr": 1, "jerk": 1,
                  "computation_time_minor": 0.5,
                  "computation_time_middle": 1, "computation_time_major": 2}
contraints = PENALTY_POINTS.keys()


# Plot the trajectories
def plot_trajectories(origin_name=None, save_dir=None, episode_id=-1):
    """
    Args:
        origin_name: name of the json to read in which we expect to have all the info. The json must have for each
        episode:
            ee_x list
            ee_y list
            puck_x list
            puck_y list
            thetas
        save_dir: name of the directory where to save the image
        episode_id: the episode we are interested in, if -1 all are plotted

    Returns: None
    """
    # Open the json
    with io.open(origin_name, 'rb') as json_file:
        data = json.load(json_file)

    if episode_id == -1:
        n_episodes = len(data)
        ep_keys = list(data.keys())
    else:
        n_episodes = 1
        ep_keys = [str(episode_id)]

    # fixed dimensions
    table_l = 1.948
    table_w = 1.038
    goal_w = 0.25
    p_r = 0.03165
    ee_r = 0.04815

    # create and adjust the plot
    plt.clf()
    plt.title("Trajectories")
    plt.grid()
    ds = 0.01
    plt.xlim(-table_l / 2 - ds, table_l / 2 + ds)
    plt.ylim(-table_w / 2 - ds, table_w / 2 + ds)
    plt.gca().set_aspect('equal', adjustable='box')

    # plot the table
    x = [-table_l / 2, table_l / 2]
    plt.plot(x, np.ones(len(x)) * table_w / 2, "black")
    plt.plot(x, -np.ones(len(x)) * table_w / 2, "black")

    y = [-table_w / 2, table_w / 2]
    plt.plot(np.ones(len(y)) * table_l / 2, y, "black")
    plt.plot(-np.ones(len(y)) * table_l / 2, y, "black")
    plt.plot(np.zeros(len(y)), y, "k--")

    y_goal = [-goal_w / 2, goal_w / 2]
    plt.plot(np.ones(len(y_goal)) * table_l / 2, y_goal, "orange", linewidth=3)
    plt.plot(-np.ones(len(y_goal)) * table_l / 2, y_goal, "orange", linewidth=3)

    # plot the trajectory
    for ep in ep_keys:
        # plot the initial pos of the ee and the puck
        ee_pos = [data[ep]["ee_x"][0], data[ep]["ee_y"][0]]
        p_pos = [data[ep]["puck_x"][0], data[ep]["puck_y"][0]]
        plt.gca().add_artist(Ellipse((ee_pos[0], ee_pos[1]), ee_r * 2, ee_r * 2, color="r"))
        plt.gca().add_artist(Ellipse((p_pos[0], p_pos[1]), p_r * 2, p_r * 2, color="b"))

        # plot the actual trajectory
        ee_x = data[ep]["ee_x"]
        ee_y = data[ep]["ee_y"]
        p_x = data[ep]["puck_x"]
        p_y = data[ep]["puck_y"]
        ll = str(ep) + ": " + str(data[ep]["thetas"])
        plt.plot(ee_x + p_x, ee_y + p_y, "-", label=ll)

    # legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # check if the directory exists
    if not os.path.exists(os.path.dirname(save_dir+"/")):
        try:
            os.makedirs(os.path.dirname(save_dir+"/"))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # save
    name = save_dir + "/" + "trajectory_plot_" + datetime.now().strftime('_%Y%m%d__%H_%M_%S') + ".pdf"
    plt.savefig(name, format="pdf", bbox_inches="tight")

    return


class AirHockeyEnv(gym.Env):
    def __init__(self, env, action_type="position-velocity", interpolation_order=3, custom_reward_function=None,
                 simple_reward=False, high_level_action=True, agent_id=1, delta_action=False, delta_ratio=0.1,
                 jerk_only=False, include_joints=False, **kwargs):
        self._env = AirHockeyChallengeWrapper(env=env, action_type=action_type, interpolation_order=interpolation_order,
                                              custom_reward_function=custom_reward_function, **kwargs)
        self.env_info = env_info = self._env.env_info
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        ac_space = self._env.env_info["rl_info"].action_space
        obs_space = self._env.env_info["rl_info"].observation_space
        self.gamma = self._env.env_info["rl_info"].gamma
        self.simple_reward = simple_reward
        self.jerk_only = jerk_only
        self.high_level_action = high_level_action
        self.delta_action = delta_action
        self.delta_ratio = delta_ratio
        self.include_joints = include_joints
        self.desired_height = self._env.env_info['robot']['ee_desired_height']
        self.low_position = np.array([0.54, -0.5, 0])
        self.high_position = np.array([1.5, 0.5, 0.3])
        if self.high_level_action:
            if self._env.env_info['robot']['n_joints'] == 3:
                joint_anchor_pos = np.array([-1.15570723, 1.30024401, 1.44280414])
                x_init = np.array([0.65, 0., 0.1])
                x_home = np.array([0.65, 0., 0.1])
                max_hit_velocity = 1.0
            else:
                joint_anchor_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
                x_init = np.array([0.65, 0., self._env.env_info['robot']['ee_desired_height'] + 0.2])
                x_home = np.array([0.65, 0., self._env.env_info['robot']['ee_desired_height']])
                max_hit_velocity = 1.2

            self.agent_params = {'switch_tactics_min_steps': 15,
                                 'max_prediction_time': 1.0,
                                 'max_plan_steps': 5,
                                 'static_vel_threshold': 0.15,
                                 'transversal_vel_threshold': 0.1,
                                 'joint_anchor_pos': joint_anchor_pos,
                                 'default_linear_vel': 0.6,
                                 'x_init': x_init,
                                 'x_home': x_home,
                                 'hit_range': [0.8, 1.3],
                                 'max_hit_velocity': max_hit_velocity,
                                 'defend_range': [0.8, 1.0],
                                 'defend_width': 0.45,
                                 'prepare_range': [0.8, 1.3]}

            self._system_state = SystemState(self._env.env_info, agent_id, agent_params=self.agent_params)
            self.puck_pos_ids = puck_pos_ids = self._env.env_info["puck_pos_ids"]
            self.puck_vel_ids = puck_vel_ids = self._env.env_info["puck_vel_ids"]
            low_position = self._env.env_info["rl_info"].observation_space.low[puck_pos_ids]
            low_velocity = self._env.env_info["rl_info"].observation_space.low[puck_vel_ids]
            high_position = self._env.env_info["rl_info"].observation_space.high[puck_pos_ids]
            high_velocity = self._env.env_info["rl_info"].observation_space.high[puck_vel_ids]
            low_action = low_position[:2]
            low_action[0] = 1.52
            low_action = np.array([0.54, -0.5])
            high_action = np.array([1.5, 0.5])
            low_position = np.array([0.54, -0.5, 0])
            high_position = np.array([1.5, 0.5, 0.3])
            if self.include_joints:
                joint_pos_ids = self._env.env_info["joint_pos_ids"]
                low_joints_pos = self._env.env_info["rl_info"].observation_space.low[joint_pos_ids]
                high_joints_pos = self._env.env_info["rl_info"].observation_space.high[joint_pos_ids]
                joint_vel_ids = self._env.env_info["joint_vel_ids"]
                low_joints_vel = self._env.env_info["rl_info"].observation_space.low[joint_vel_ids]
                high_joints_vel = self._env.env_info["rl_info"].observation_space.high[joint_vel_ids]
                low_state = np.concatenate([low_position, low_velocity, low_position[:2], low_joints_pos,
                                            low_joints_vel])
                high_state = np.concatenate([high_position, high_velocity, high_position[:2], high_joints_pos,
                                             high_joints_vel])
            else:
                low_state = np.concatenate([low_position, low_velocity, low_position[:2]])
                high_state = np.concatenate([high_position, high_velocity, high_position[:2]])
        else:
            low_action = np.concatenate([ac_space.low, ac_space.low])
            high_action = np.concatenate([ac_space.high, ac_space.high])
            low_state = obs_space.low
            high_state = obs_space.high
            low_position = np.array([0.54, -0.5])
            high_position = np.array([1.5, 0.5])
            low_state = np.concatenate([low_state, low_position])
            high_state = np.concatenate([high_state, high_position])
        if self.delta_action:
            range_action = np.abs(high_action - low_action) * delta_ratio
            low_action = - range_action
            high_action = range_action
        self.action_space = spaces.Box(low=low_action, high=high_action)
        self.observation_space = spaces.Box(low=low_state, high=high_state)
        self.t = 0
        self.state = self.reset()

    def get_puck_pos(self, obs):
        """
        Get the Puck's position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            Puck's position of the robot

        """
        return obs[self.env_info['puck_pos_ids']]

    def get_puck_vel(self, obs):
        """
        Get the Puck's velocity from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            Puck's velocity of the robot

        """
        return obs[self.env_info['puck_vel_ids']]

    def get_joint_pos(self, obs):
        """
        Get the joint position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint position of the robot

        """
        return obs[self.env_info['joint_pos_ids']]

    def get_joint_vel(self, obs):
        """
        Get the joint velocity from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint velocity of the robot

        """
        return obs[self.env_info['joint_vel_ids']]

    def get_ee_pose(self, obs):
        """
        Get the Opponent's End-Effector's Position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            opponent's end-effector's position

        """
        return forward_kinematics(self.robot_model, self.robot_data, self.get_joint_pos(obs))[0]

    def _action_transform(self, action):
        obs = self._state
        joint_pos = self.get_joint_pos(obs)
        joint_vel = self.get_joint_vel(obs)
        self._system_state.update_observation(joint_pos, joint_vel, self.get_puck_pos(obs))
        command = np.concatenate([action, np.array([self.desired_height])])
        success, new_joint_pos = inverse_kinematics(self.robot_model, self.robot_data, command)
        joint_velocities = (new_joint_pos - self._old_joint_pos) / self.env_info['dt']

        if not success:
            self._fail_count += 1
            # new_joint_pos = joint_pos

        action = np.vstack([new_joint_pos, joint_velocities])
        return action

    def step(self, action):
        if self.delta_action:
            if self.high_level_action:
                ee_pos = self.get_ee_pose(self._state)
                action = ee_pos[:2] + action[:2]
            else:
                joint_pos = self.get_joint_pos(self._state)
                joint_vel = self.get_joint_vel(self._state)
                action = np.concatenate([joint_pos, joint_vel]) + action
        if self.high_level_action:
            action = np.clip(action, a_min=self.low_position[:2], a_max=self.high_position[:2])
            action = self._action_transform(action[:2])
        else:
            action = np.reshape(action, (2, -1))
        obs, reward, done, info = self._env.step(action)
        joint_pos = self.get_joint_pos(obs)
        self._old_joint_pos = joint_pos
        info["joint_pos_constr"] = info["constraints_value"]["joint_pos_constr"]
        info["joint_vel_constr"] = info["constraints_value"]["joint_vel_constr"]
        info["ee_constr"] = info["constraints_value"]["ee_constr"]
        info.pop('constraints_value', None)
        if self.simple_reward:
            r = info["success"]
            if r:
                if not done:
                    print("Whaaat")
        else:
            r = 0
            if not self.jerk_only:
                for constr in ['joint_pos_constr', 'joint_vel_constr', 'ee_constr']:
                    if np.any(np.array(info[constr] > 0)):
                        r -= PENALTY_POINTS[constr]
            if np.any(np.array(info["jerk"] > 10000)):
                # r -= PENALTY_POINTS["jerk"]
                r -= PENALTY_POINTS["jerk"] + np.clip(np.array(info["jerk"]), 0, 100000).max() / 100000
            r += info["success"] * 800
            if done and not info["success"] and self.t < self._env._mdp_info.horizon:
                r -= (PENALTY_POINTS["jerk"] + 1) * 400
        self._state = obs
        obs = self._get_state(obs)
        self.state = obs
        self.t += 1
        return obs, r, done, info

    # pock pos, puck_vel, ee_poss
    def _get_state(self, obs):
        ee_pos = self.get_ee_pose(obs)
        if self.high_level_action:
            puck_pos = self.get_puck_pos(obs)
            puck_vel = self.get_puck_vel(obs)
            if self.include_joints:
                joint_pos = self.get_joint_pos(obs)
                joint_vel = self.get_joint_vel(obs)
                return np.concatenate([puck_pos, puck_vel, ee_pos[:2], joint_pos, joint_vel])
            return np.concatenate([puck_pos, puck_vel, ee_pos[:2]])
        else:
            return np.concatenate([obs, ee_pos[:2]])

    def reset(self):
        self.t = 0
        self._fail_count = 0
        st = self._env.reset()
        self._state = st
        st = self._get_state(st)
        self.state = st
        self._old_joint_pos = self.get_joint_pos(self._state)
        return st

    def seed(self, seed=None):
        return self._env.seed(seed)

    @property
    def np_random(self):
        return np.random

    def render(self, mode='human'):
        self._env.render()
