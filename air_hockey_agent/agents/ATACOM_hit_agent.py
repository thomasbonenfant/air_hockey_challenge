import os
import matplotlib.pyplot as plt
import numpy as np
import math
from air_hockey_agent.agents.hit_agent_SAC import HittingAgent
from air_hockey_challenge.utils import forward_kinematics
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
from air_hockey_agent.agents.atacom.utils.null_space_coordinate import rref, pinv_null
from air_hockey_agent.agents.atacom.constraints import ViabilityConstraint, ConstraintsSet
from mushroom_rl.utils.spaces import Box
import time

import mujoco
from air_hockey_challenge.utils.kinematics import link_to_xml_name, inverse_kinematics


class AtacomHittingAgent(HittingAgent):

    def __init__(self, env, **kwargs):
        """
        Constructor
        Args:
            base_env (mushroomrl.Core.Environment): The base environment inherited from
            dim_q (int): [int] dimension of the directly controllable variable
            vel_max (array, float): the maximum velocity of the directly controllable variable
            acc_max (array, float): the maximum acceleration of the directly controllable variable
            f (ViabilityConstraint, ConstraintsSet): the equality constraint f(q) = 0
            g (ViabilityConstraint, ConstraintsSet): the inequality constraint g(q) = 0
            Kc (array, float): the scaling factor for error correction
            Kq (array, float): the scaling factor for the viability acceleration bound
            time_step (float): the step size for time discretization
        """

        env['rl_info'].action_space = Box(env['robot']['joint_acc_limit'][0], env['robot']['joint_acc_limit'][1])
        env['rl_info'].observation_space = Box(np.append(env['rl_info'].observation_space.low, 0), np.append(env['rl_info'].observation_space.high, 1))
        super().__init__(env, **kwargs)

        dim_q = self.env_info['robot']['n_joints']
        Kc = 100.
        timestamp = 1/50.
        # why dim out of 5?
        # cart_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=5, fun=self.cart_pos_g, J=self.cart_pos_J_g, b=self.cart_pos_b_g, K=0.5)
        ee_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=3, fun=self.ee_pos_g, J=self.ee_pos_J_g, b=self.ee_pos_b_g, K=0.5)

        # TODO: still need to check validity of fun, J, b
        joint_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=dim_q, fun=self.joint_pos_g, J=self.joint_pos_J_g,
                                          b=self.joint_pos_b_g, K=1.0)
        f = None
        g = ConstraintsSet(dim_q)
        g.add_constraint(ee_pos_g)
        g.add_constraint(joint_pos_g)
        #g.add_constraint(joint_vel_g)

        # TODO: if we want to include the jerk we may want to create a accel constraint and put the value here
        acc_max = 10
        vel_max = self.env_info['constraints'].get('joint_vel_constr').joint_limits[1:] # takes only positive constraints
        vel_max = np.squeeze(vel_max)
        vel_max = np.ones(dim_q) * 2.35619449

        Kq = 2 * acc_max / vel_max

        self.env = env
        self.dims = {'q': dim_q, 'f': 0, 'g': 0}
        self.f = f
        self.g = g
        self.time_step = timestamp

        self._logger = None

        if self.f is not None:
            assert self.dims['q'] == self.f.dim_q, "Input dimension is different in f"
            self.dims['f'] = self.f.dim_out
        if self.g is not None:
            assert self.dims['q'] == self.g.dim_q, "Input dimension is different in g"
            self.dims['g'] = self.g.dim_out
            self.mu = np.zeros(self.dims['g'])

        self.dims['null'] = self.dims['q'] - self.dims['f']
        self.dims['c'] = self.dims['f'] + self.dims['g']

        if np.isscalar(Kc):
            self.K_c = np.ones(self.dims['c']) * Kc
        else:
            self.K_c = Kc

        """
        Initial point of the position was zeros. it is not right since the initial pos of the joints
        is not at zeros. this is not such a big deal but it may cause some issues. the numbers hard coded are the 
        initial pos of the joints. initial velocity is zeros which is right.
        """
        # TODO for next phase, we have to have initial pos of the joints of the new robot
        self.q = np.array([-1.156, 1.300, 1.443])
        self.dq = np.zeros(self.dims['q'])

        self._mdp_info = self.env['rl_info'].copy()

        #self._mdp_info.action_space = Box(low=-np.ones(self.dims['null']), high=np.ones(self.dims['null']))

        if np.isscalar(vel_max):
            self.vel_max = np.ones(self.dims['q']) * vel_max
        else:
            self.vel_max = vel_max
            assert np.shape(self.vel_max)[0] == self.dims['q']

        if np.isscalar(acc_max):
            self.acc_max = np.ones(self.dims['q']) * acc_max
        else:
            self.acc_max = acc_max
            assert np.shape(self.acc_max)[0] == self.dims['q']

        if np.isscalar(Kq):
            self.K_q = np.ones(self.dims['q']) * Kq
        else:
            self.K_q = Kq
            assert np.shape(self.K_q)[0] == self.dims['q']

        # self.alpha_max = np.ones(self.dims['null']) * self.acc_max.max()
        self.alpha_max = np.ones(self.dims['null']) * self.acc_max.max()

        self._act_a = None
        self._act_b = None
        self._act_err = None

        self.constr_logs = list()
        # shouldn't be needed
        # self.env.step_action_function = self.step_action_function
        self.mu = 0
        self.last_actions = []
        self.tmp = 0
        self.add_preprocessor(self.add_observation_preprocessors)
        self.was_hit = 0

        self.i = 0


    def episode_start(self):
        #print("ATACOM episode start")
        self.was_hit = 0
        self.reset()
        #super().episode_start()

    def add_observation_preprocessors(self, state):
        """mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']
        id_ee = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_to_xml_name(mj_model, 'ee'))
        print(mujoco.mj_collision(mj_model[id_ee], mj_data[id_ee]), mujoco.contact())
        #id_puck = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_to_xml_name(mj_model, 'puck'))
        acc = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, id_ee, acc)
        print(acc)
        print(id_ee)
        #print(id_puck)
        # to use  mujoco.mjtObj.mjOBJ_SITE the value should be 6
        print(mujoco.mj_collision(mj_model, mj_data))"""
        # TODO: implement better contact mechanism
        if self.was_hit == 0:
            self.was_hit = 0 if np.sum(self.get_puck_vel(state)[:2]) == 0 else 1
            #if self.was_hit == 1:
            #    print("HIT")

        return np.append(state, self.was_hit)

    def fit(self, dataset, **info):
        # change action to acceleration
        for i in range(len(dataset)):
            # convert tuple to list
            dataset[i] = list(dataset[i])
            # change action (space, velocity) to action (acceleration)
            dataset[i][1] = np.array(self.last_actions[i])

            # convert list back to tuple
            dataset[i] = tuple(dataset[i])
        self.last_actions = []

        super().fit(dataset, **info)


    def generate_circle(self, center_x, center_y, radius, num_points):
        # Generate equally spaced angles
        angles = [2 * math.pi * i / num_points + math.pi for i in range(num_points)]

        # Generate data points on the circle
        x_values = [center_x + radius * math.cos(angle) for angle in angles]
        y_values = [center_y + radius * math.sin(angle) for angle in angles]

        return x_values, y_values

    def draw_action(self, observation):
        """
        Draw an action from the agent's policy.
        :param observation: The current observation of the environment.
        """
        # Sample policy action αk ∼ π(·|sk).
        self.q = self.get_joint_pos(observation)
        self.dq = self.get_joint_vel(observation)

        n_points = 100
        self.i += 1
        self.i = self.i % n_points

        # alpha = super().draw_action(observation)
        # self.last_actions.append(alpha)
        #
        ee_pos = self.get_ee_pose(observation)[0]
        ee_pos_world = robot_to_world(self.env_info["robot"]["base_frame"][0], ee_pos)[0]
        x_values, y_values = self.generate_circle(-0.63, 0, 0.23, n_points)
        new_EE_pos = np.array([x_values[self.i], y_values[self.i], 0.0])
        if self.i == 1:
            colors = np.linspace(0, 1, n_points)
            plt.scatter(x_values, y_values, c=colors, cmap='coolwarm')
            plt.colorbar(label='Color')
            plt.show()
        ee_in_robot = world_to_robot(self.env_info["robot"]["base_frame"][0], new_EE_pos)[0]
        success, joint_pos_des = inverse_kinematics(self.robot_model, self.robot_data, ee_in_robot)
        joint_vel = (joint_pos_des - self.q) / self.time_step
        time.sleep(0.1)
        joint_acc = (joint_vel - self.dq) / self.time_step

        alpha = joint_acc
        self.last_actions.append(alpha)

        # return np.concatenate((joint_pos_des, joint_vel)).reshape(2, 3)
        """if self.was_hit == 0:
            ee_pos = forward_kinematics(self.robot_model, self.robot_data, self.get_joint_pos(observation))[0]
            puck_pos = self.get_puck_pos(observation)
            diff = (puck_pos - ee_pos) / np.linalg.norm(puck_pos - ee_pos) * 0.05 + ee_pos
            success, joint_pos_des = inverse_kinematics(self.robot_model, self.robot_data, diff)
            joint_vel = (joint_pos_des - self.q) / self.time_step
        else:
            joint_vel = np.zeros(self.dims['q'])

        alpha = (joint_vel - self.dq) / self.time_step"""

        #alpha = np.clip(alpha, self.env_info['robot']['joint_acc_limit'][0], self.env_info['robot']['joint_acc_limit'][1])
        #alpha = alpha * self.alpha_max # TODO: why alpha max breaks everything?
        # Observe the qk, q˙k from sk

        # Compute slack variable mu
        # self._compute_slack_variables()

        # Compute Jc, k = Jc(qk, µk), ψk = ψ(qk, q˙k), ck = c(qk, q˙k, µk)
        Jc, psi = self._construct_Jc_psi(self.q, self.mu, self.dq)
        Jc_inv, Nc = pinv_null(Jc)
        # Compute the RCEF of tangent space basis of NcR
        # Nc = rref(Nc[:, :self.dims['null']], row_vectors=False, tol=0.05)
        Nc = rref(Nc[:, :self.dims['null']], row_vectors=False, tol=0.05)

        # Compute the tangent space acceleration [q¨k µ˙ k].T ← −J^†_c,k [K_cck + ψ_k] + N^R_c α_k
        self._act_a = -Jc_inv @ psi
        self._act_b = Nc @ alpha
        self._act_err = self._compute_error_correction(self.q, self.dq, self.mu, Jc_inv)
        ddq_ds = self._act_a + self._act_b + self._act_err

        self.mu += ddq_ds[self.dims['q']:(self.dims['q'] + self.dims['g'])] * self.time_step
        # Clip the joint acceleration q¨k ← clip(q¨k, al, au)
        ddq = self.acc_truncation(self.dq, ddq_ds[:self.dims['q']])
        # Integrate the slack variable µk+1 = µk + µ˙ k∆T
        # pokh = forward_kinematics_acc(self.robot_model, self.robot_data, ddq)

        ctrl_action = self.acc_to_ctrl_action(ddq)#.reshape((6,))

        return ctrl_action

    def acc_to_ctrl_action(self, ddq):
        # integrate acceleration because we do control the robot with a PD Controller
        # next_dq = self.dq + ddq * self.time_step
        # next_q = self.q + self.dq * self.time_step + 0.5 * ddq * (self.time_step ** 2)
        M = 4
        dt = self.time_step / M
        # Runge-Kutta integration
        pos = self.q
        vel = self.dq
        for _ in range(M):

            k1_qvel = dt * ddq
            k1_qpos = dt * vel
            k2_qvel = dt * ddq  # Use updated acceleration if available
            k2_qpos = dt * (vel + 0.5 * k1_qvel)
            k3_qvel = dt * ddq  # Use updated acceleration if available
            k3_qpos = dt * (vel + 0.5 * k2_qvel)
            k4_qvel = dt * ddq  # Use updated acceleration if available
            k4_qpos = dt * (vel + k3_qvel)

            # Update joint positions and velocities
            pos += (1 / 6) * (k1_qpos + 2 * k2_qpos + 2 * k3_qpos + k4_qpos)
            vel += (1 / 6) * (k1_qvel + 2 * k2_qvel + 2 * k3_qvel + k4_qvel)

        return np.concatenate((pos, vel)).reshape(2,3)

        # return self.env.client.calculateInverseDynamics(self.env._model_map['planar_robot_1'], q, dq, ddq)

    def acc_truncation(self, dq, ddq):
        # TODO: test on correctness of this value
        acc_u = np.maximum(np.minimum(self.acc_max, -self.K_q * (dq - self.vel_max)), -self.acc_max)
        acc_l = np.minimum(np.maximum(-self.acc_max, -self.K_q * (dq + self.vel_max)), self.acc_max)
        ddq = np.clip(ddq, acc_l, acc_u)
        return ddq

    """ OK """
    def _compute_slack_variables(self):
        self.mu = None
        if self.dims['g'] > 0:
            # TODO: check if this is correct
            mu_squared = np.maximum(-2 * self.g.fun(self.q, self.dq, origin_constr=False), 0)
            self.mu = np.sqrt(mu_squared)
    """ OK """
    def _construct_Jc_psi(self, q, s, dq):
        Jc = np.zeros((self.dims['f'] + self.dims['g'], self.dims['q'] + self.dims['g']))
        psi = np.zeros(self.dims['c'])
        if self.dims['f'] > 0:
            idx_0 = 0
            idx_1 = self.dims['f']
            Jc[idx_0:idx_1, :self.dims['q']] = self.f.K_J(q)
            psi[idx_0:idx_1] = self.f.b(q, dq)
        if self.dims['g'] > 0:
            idx_0 = self.dims['f']
            idx_1 = self.dims['f'] + self.dims['g']
            Jc[idx_0:idx_1, :self.dims['q']] = self.g.K_J(q)
            Jc[idx_0:idx_1, self.dims['q']:(self.dims['q'] + self.dims['g'])] = np.diag(s)
            psi[idx_0:idx_1] = self.g.b(q, dq)
        return Jc, psi

    def _compute_c(self, q, dq, s, origin_constr=False):
        c = np.zeros(self.dims['f'] + self.dims['g'])
        if self.dims['f'] > 0:
            idx_0 = 0
            idx_1 = self.dims['f']
            c[idx_0:idx_1] = self.f.fun(q, dq, origin_constr)
        if self.dims['g'] > 0:
            idx_0 = self.dims['f']
            idx_1 = self.dims['f'] + self.dims['g']
            if origin_constr:
                c[idx_0:idx_1] = self.g.fun(q, dq, origin_constr)
            else:
                c[idx_0:idx_1] = self.g.fun(q, dq, origin_constr) + 1 / 2 * s ** 2
        return c

    def _update_constraint_stats(self, q, dq):
        c_i = self._compute_c(q, dq, 0., origin_constr=True)
        c_i[:self.dims['f']] = np.abs(c_i[:self.dims['f']])
        c_dq_i = np.abs(dq) - self.vel_max
        self.constr_logs.append([np.max(c_i), np.max(c_dq_i)])

    def _compute_error_correction(self, q, dq, mu, Jc_inv, act_null=None):
        q_tmp = q.copy()
        dq_tmp = dq.copy()
        mu_tmp = None

        if self.dims['g'] > 0:
            mu_tmp = mu.copy()

        if act_null is not None:
            q_tmp += dq_tmp * self.time_step + act_null[:self.dims['q']] * self.time_step ** 2 / 2
            dq_tmp += act_null[:self.dims['q']] * self.time_step
            if self.dims['g'] > 0:
                mu_tmp += act_null[self.dims['q']:self.dims['q'] + self.dims['g']] * self.time_step

        return -Jc_inv @ (self.K_c * self._compute_c(q_tmp, dq_tmp, mu_tmp, origin_constr=False))

    def ee_pos_g(self, q):
        """ Compute the constraint function g(q) = 0 position of the end-effector"""
        ee_pos = forward_kinematics(self.robot_model, self.robot_data, q)[0]
        ee_pos_world = robot_to_world(self.env_info["robot"]["base_frame"][0], ee_pos)[0]
        # ee_pos_world = ee_pos + self.env.agents[0]['frame'][:2, 3]
        g_1 = - ee_pos_world[0] - (self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius'])
        g_2 = - ee_pos_world[1] - (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])
        g_3 = ee_pos_world[1] - (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])
        """
        I changed the code to be like the original, but this code and yours are identical.
        """

        return np.array([g_1, g_2, g_3])
        # g = self.env_info['constraints'].get('ee_constr')
        # return g.fun(q, None)[:3] # dq is not used, pick only the first 3 elements (do not need height constraint)

    def ee_pos_J_g(self, q):
        """ Compute the constraint function g'(q) = 0 derivative of the end-effector """
        #ee_jac = self.env_info['constraints'].get('ee_constraint').jacobian(q)
        #J_c = np.array([[-1., 0.], [0., -1.], [0., 1.]])
        #return J_c @ ee_jac
        
        g = self.env_info['constraints'].get('ee_constr')

        return g.jacobian(q, None)[:3, :3] # dq is not used

    def ee_pos_b_g(self, q, dq):
        """ TODO: understand what this is: it should be dJ/dt * dq/dt  """
        #ee_pos = forward_kinematics(self.robot_model, self.robot_data, q)
        #pino.getFrameClassicalAcceleration(self.pino_model, self.pino_data, self.pino_model.nframes - 1,pino.LOCAL_WORLD_ALIGNED).vector

        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']
        #acc = mujoco.mj_objectAcceleration(mj_model,mj_data, mujoco.mjtObj.mjOBJ_SITE, link_to_xml_name(mj_model, 'ee'))[3:] #last 3 elements are linear acceleration
        acc = np.zeros(6)
        id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_to_xml_name(mj_model, 'ee'))

        # to use  mujoco.mjtObj.mjOBJ_SITE the value should be 6
        # TODO: check if this works
        mujoco.mj_objectAcceleration(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, id, acc, 0)  # last 3 elements are linear acceleration
        acc = acc[3:]

        J_c = np.array([[-1., 0.], [0., -1.], [0., 1.]])

        return J_c @ acc[:2] # Do not understand why acc is used. acc is in theory J * ddq + dJ * dq (it's like we omit the first term)

    def joint_pos_g(self, q):
        """Compute the constraint function g(q) = 0 position of the joints"""
        g = self.env_info['constraints'].get('joint_pos_constr')
        # here the function of g is returning q - qu. we must find qu.
        q_minus_qu = g.fun(q, None)[:3]
        qu = q - q_minus_qu
        """
        Here is the main bug that I found. if you consider only 3 element of the g.fun(), based on this link:
        https://air-hockey-challenges-docs.readthedocs.io/en/latest/constraints.html#constraints
        you are ignoring the lower bound on the joint pos. if you consider all of them, which are 6, the code will break
        because they used something which was 3 originally, so for solving it, their method is better, since having 
        q ** 2 - qu ** 2 < 0 => 
        q ** 2 < qu ** 2 => 
        |q| < |qu| => 
        -qu < q < qu
        this is correct for us since the lower bound of the joint pos is actually the negative of the upper bound   
        """
        # return g.fun(q, None)[:3]
        return np.array(q ** 2 - qu ** 2)

    def joint_pos_J_g(self, q):
        """Compute constraint jacobian function J_g(q)"""
        g = self.env_info['constraints'].get('joint_pos_constr')
        # return g.jacobian(q, None)[:3, :3]
        return 2 * np.diag(q)

    def joint_pos_b_g(self, q, dq):
        """Compute constraint b(q,dq) function. It should be equal to dJ/dt * dq/dt"""
        # return 2 * np.diag(q)
        # g = self.env_info['constraint'].get('joint_pos_constr')
        # return np.zeros(g.output_dim) # this constraint is linear so second order derivative goes to zero

        return 2 * dq ** 2

    #next constraint is not used
    #def joint_vel_g(self, q, dq):
    #    #return np.array([dq ** 2 - self.pino_model.velocityLimit ** 2])

    #def joint_vel_A_g(self, q, dq):
    #    return 2 * np.diag(dq)

    #def joint_vel_b_g(self, q, dq):
    #    return np.zeros(3)

    def reset(self):
        # RESET the initial pos and vel t
        """
        we have to reset also the initialization of the robot pos and velocity
        """
        self.q = np.array([-1.156, 1.300, 1.443])
        self.dq = np.zeros(self.dims['q'])
        self.i = 0
        super().reset()
        self._compute_slack_variables()
def _mujoco_fk(acc, name, model, data):
    data.qacc[:len(acc)] = acc
    mujoco.mj_inverse(model, data)
    f = data.qfrc_inverse
    data.ctrl[:len(f)] = f
    mujoco.mj_kinematics(model, data)
    return data.body(name).xpos.copy(), data.body(name).xmat.reshape(3, 3).copy()

def link_to_xml_name(mj_model, link):
    try:
        mj_model.body('iiwa_1/base')
        link_to_frame_idx = {
            "1": "iiwa_1/link_1",
            "2": "iiwa_1/link_2",
            "3": "iiwa_1/link_3",
            "4": "iiwa_1/link_4",
            "5": "iiwa_1/link_5",
            "6": "iiwa_1/link_6",
            "7": "iiwa_1/link_7",
            "ee": "iiwa_1/striker_joint_link",
        }
    except:
        link_to_frame_idx = {
            "1": "planar_robot_1/body_1",
            "2": "planar_robot_1/body_2",
            "3": "planar_robot_1/body_3",
            "ee": "planar_robot_1/body_ee",
        }
    return link_to_frame_idx[link]



def forward_kinematics_acc(mj_model, mj_data, q, link="ee"):
    """
    Compute the forward kinematics of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is the
        case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        q (np.array):
            joint configuration for which the forward kinematics are computed
        link (string, "ee"):
            Link for which the forward kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"]. When using planar the choices are ["1", "2", "3", "ee"]

    Returns
    -------
    position: numpy.ndarray, (3,)
        Position of the link in robot's base frame
    orientation: numpy.ndarray, (3, 3)
        Orientation of the link in robot's base frame
    """

    return _mujoco_fk(q, link_to_xml_name(mj_model, link), mj_model, mj_data)