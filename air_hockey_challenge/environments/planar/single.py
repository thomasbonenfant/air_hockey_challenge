import numpy as np
import mujoco
import os
import yaml

from air_hockey_challenge.environments.planar import AirHockeyBase


class AirHockeySingle(AirHockeyBase):
    """
    Base class for single agent air hockey tasks that introduces
    noises in the observations.

    """
    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):

        """
        Constructor.

        """
        self.init_state = np.array([-1.15570723,  1.30024401,  1.44280414])
        super().__init__(gamma=gamma, horizon=horizon, n_agents=1, viewer_params=viewer_params)

        self.gamma = gamma
        self.filter_ratio = 0.274
        self.q_pos_prev = np.zeros(self.env_info["robot"]["n_joints"])
        self.q_vel_prev = np.zeros(self.env_info["robot"]["n_joints"])

        # previous position/velocity of the joints, used if one of them is broken or if we loose the tracking
        self.last_puck_pos = None
        self.last_puck_vel = None

        # Dictionary containing the noise options
        path = 'air_hockey_agent/environment_config.yml'
        if os.path.exists(path):
            with open(path, 'r') as stream:
                config = yaml.safe_load(stream)

            # load options from configuration file
            self.cov = config['cov_obs']  # std dev of the noise in the observations
            self.var_env = dict()
            self.var_env['is_track_lost'] = config['is_track_lost']     # True if we loose the track, does not affect the robot that much
            self.var_env['is_obs_noisy'] = config['is_obs_noisy']       # True if there is noise in the measurements, effectively reduces performances
            self.var_env['track_loss_prob'] = config['track_loss_prob'] # probability of loosing the tracking at each observation

            #self.var_env['noise_value'] = np.random.normal(0, self.sigma, 6)  # white noise vector for puck's pos and vel
            self.var_env['noise_value'] = np.random.multivariate_normal(np.zeros(len(self.cov)), self.cov)

        else:
            # load options from configuration file
            self.cov = None  # std dev of the noise in the observations
            self.var_env = dict()
            self.var_env['is_track_lost'] = None
            self.var_env['is_obs_noisy'] = None
            self.var_env['track_loss_prob'] = None
            #self.var_env['noise_value'] = np.random.multivariate_normal(np.zeros(2), self.cov)


    def get_ee(self):
        """
        Getting the ee properties from the current internal state. Can also be obtained via forward kinematics
        on the current joint position, this function exists to avoid redundant computations.

        Returns:
            ([pos_x, pos_y, pos_z], [ang_vel_x, ang_vel_y, ang_vel_z, lin_vel_x, lin_vel_y, lin_vel_z])
        """
        ee_pos = self._read_data("robot_1/ee_pos")

        ee_vel = self._read_data("robot_1/ee_vel")

        return ee_pos, ee_vel

    def get_joints(self, obs):
        """
        Get joint position and velocity of the robot
        """
        q_pos = np.zeros(3)
        q_vel = np.zeros(3)
        for i in range(3):
            q_pos[i] = self.obs_helper.get_from_obs(obs, "robot_1/joint_" + str(i+1) + "_pos")[0]
            q_vel[i] = self.obs_helper.get_from_obs(obs, "robot_1/joint_" + str(i+1) + "_vel")[0]

        return q_pos, q_vel

    def get_state(self, obs):
        """
        Get joint position, velocities; puck position and velocities, opponent's end effector position and velocities
        if present.
        It retrieves info by calling already existing functions, it is a summary to retrieve all info together

        Returns
        -------
            (puck_position, puck_velocities, joint_position, joint_velocities, ee_position, ee_velocities, has_hit)
        """

        joint_pos, joint_vel = self.get_joints(obs)
        puck_pos, puck_vel = self.get_puck(obs)
        ee_pos, ee_vel = self.get_ee()

        # FIXME it his just a patch
        # FIXME since the puck starts still it will be hit when its speed will be different from 0
        has_hit = False
        if puck_vel[0] != 0:
            has_hit = True
        #has_hit = self._check_collision("puck", "robot_1/ee")  # FIXME dovrebbe andare nello step

        return puck_pos, puck_vel, joint_pos, joint_vel, ee_pos, ee_vel, has_hit

    def _modify_observation(self, obs):
        new_obs = obs.copy()
        puck_pos, puck_vel = self.get_puck(obs)

        puck_pos = self._puck_2d_in_robot_frame(puck_pos, self.env_info['robot']['base_frame'][0])

        puck_vel = self._puck_2d_in_robot_frame(puck_vel, self.env_info['robot']['base_frame'][0], type='vel')

        # Loss of tracking
        if self.var_env["is_track_lost"]:
            if np.random.uniform(0,1) < self.var_env['track_loss_prob'] or (self.last_puck_pos is None and self.last_puck_vel is None):
                self.last_puck_pos = puck_pos
                self.last_puck_vel = puck_vel

            puck_pos = self.last_puck_pos
            puck_vel = self.last_puck_vel

        # Observation Noise 
        if self.var_env["is_obs_noisy"]:

            #self.var_env['noise_value'] = np.random.normal(0, self.sigma, 6) # resample to add a different white noise at each observation
            self.var_env['noise_value'] = np.random.multivariate_normal(np.zeros(len(self.cov)), self.cov, size=3) # resample to add a different white noise at each observation

            #print('\n', self.var_env['noise_value'], '\n')
        
            self.obs_helper.get_from_obs(new_obs, "puck_x_pos")[:] = puck_pos[0] + self.var_env['noise_value'][0][0]
            self.obs_helper.get_from_obs(new_obs, "puck_y_pos")[:] = puck_pos[1] + self.var_env['noise_value'][1][0]
            self.obs_helper.get_from_obs(new_obs, "puck_yaw_pos")[:] = puck_pos[2] + self.var_env['noise_value'][2][0]

            self.obs_helper.get_from_obs(new_obs, "puck_x_vel")[:] = puck_vel[0] + self.var_env['noise_value'][0][1]
            self.obs_helper.get_from_obs(new_obs, "puck_y_vel")[:] = puck_vel[1] + self.var_env['noise_value'][1][1]
            self.obs_helper.get_from_obs(new_obs, "puck_yaw_vel")[:] = puck_vel[2] + self.var_env['noise_value'][2][1]
            
        else:
            self.obs_helper.get_from_obs(new_obs, "puck_x_pos")[:] = puck_pos[0]
            self.obs_helper.get_from_obs(new_obs, "puck_y_pos")[:] = puck_pos[1]
            self.obs_helper.get_from_obs(new_obs, "puck_yaw_pos")[:] = puck_pos[2]

            self.obs_helper.get_from_obs(new_obs, "puck_x_vel")[:] = puck_vel[0]
            self.obs_helper.get_from_obs(new_obs, "puck_y_vel")[:] = puck_vel[1]
            self.obs_helper.get_from_obs(new_obs, "puck_yaw_vel")[:] = puck_vel[2]

        return new_obs

    def setup(self, state=None):
        for i in range(3):
            self._data.joint("planar_robot_1/joint_" + str(i+1)).qpos = self.init_state[i]
            self.q_pos_prev[i] = self.init_state[i]
            self.q_vel_prev[i] = self._data.joint("planar_robot_1/joint_" + str(i + 1)).qvel

        mujoco.mj_fwdPosition(self._model, self._data)
        super().setup(state)

    def _create_observation(self, obs):
        # Filter the joint velocity
        q_pos, q_vel = self.get_joints(obs)
        q_vel_filter = self.filter_ratio * q_vel + (1 - self.filter_ratio) * self.q_vel_prev
        self.q_pos_prev = q_pos
        self.q_vel_prev = q_vel_filter

        for i in range(3):
            self.obs_helper.get_from_obs(obs, "robot_1/joint_" + str(i+1) + "_vel")[:] = q_vel_filter[i]

        yaw_angle = self.obs_helper.get_from_obs(obs, "puck_yaw_pos")
        self.obs_helper.get_from_obs(obs, "puck_yaw_pos")[:] = (yaw_angle + np.pi) % (2 * np.pi) - np.pi
        return obs
