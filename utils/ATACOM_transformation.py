import numpy as np
from utils.null_space_coordinate import rref, pinv_null
from utils.constraints import ViabilityConstraint, ConstraintsSet
from mushroom_rl.utils.spaces import Box
import mujoco
from envs.air_hockey_challenge.air_hockey_challenge.utils.kinematics import link_to_xml_name


class AtacomTransformation:

    def __init__(self, env_info, get_joint_pos, get_joint_vel, Kc=200., timestamp=1/50):
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
        self.env_info = env_info

        self.dim_q = dim_q = self.env_info['robot']['n_joints']
        self.get_joint_pos = get_joint_pos
        self.get_joint_vel = get_joint_vel

        ee_pos_g2 = ViabilityConstraint(dim_q=dim_q, dim_out=2, fun=self.ee_pose_f, J=self.ee_pose_J_f,
                                        b=self.ee_pos_b_f, K=1)
        ee_pos_f = ViabilityConstraint(dim_q=dim_q, dim_out=1, fun=self.ee_pose_f, J=self.ee_pose_J_f,
                                       b=self.ee_pos_b_f, K=1)

        self.ee_pos_dim_out = 5
        # ee_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=5, fun=self.ee_pos_g, J=self.ee_pos_J_g, b=self.ee_pos_b_g, K=0.5)
        ee_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=self.ee_pos_dim_out, fun=self.ee_pos_g, J=self.ee_pos_J_g,
                                       b=self.ee_pos_b_g, K=0.5)

        joint_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=dim_q, fun=self.joint_pos_g, J=self.joint_pos_J_g,
                                          b=self.joint_pos_b_g, K=1.0)

        joint_vel_g = ViabilityConstraint(dim_q=dim_q, dim_out=3, fun=self.joint_vel_g, J=self.joint_vel_J_g,
                                          b=self.joint_vel_b_g, K=1.0)

        f = ConstraintsSet(dim_q)
        g = ConstraintsSet(dim_q)
        f.add_constraint(ee_pos_f)
        g.add_constraint(ee_pos_g)
        g.add_constraint(joint_pos_g)

        # TODO: if we want to include the jerk we may want to create a accel constraint and put the value here
        acc_max = self.env_info['robot']['joint_acc_limit'][1]
        vel_max = self.env_info['constraints'].get('joint_vel_constr').joint_limits[
            1]  # takes only positive constraints
        vel_max = np.squeeze(vel_max)

        Kq = 2 * acc_max / vel_max
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

        self.q = np.zeros(self.dims['q'])
        self.dq = np.zeros(self.dims['q'])

        self._mdp_info = self.env_info['rl_info'].copy()
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
        self.alpha_max = np.ones(self.dims['null']) * self.acc_max.max()

        self._act_a = None
        self._act_b = None
        self._act_err = None

        self.constr_logs = list()
        # shouldn't be needed
        # self.env.step_action_function = self.step_action_function
        self.mu = 0
        self.tmp = 0
        #self.add_preprocessor(self.add_observation_preprocessors)
        self.was_hit = 0

    def reset(self):
        # TODO: why is this function not executed?
        # in episode start this method is not called but the child method is called (HittingAgent)
        #print("resetting ATACOM")
        self.first_step = True
        self.was_hit = 0

    def episode_start(self):
        self.was_hit = 0
        self.reset()

    def draw_action(self, alpha, joint_pos, joint_vel):
        """
        Draw an action from the agent's policy.
        :param observation: The current observation of the environment.
        """
        # Sample policy action αk ∼ π(·|sk).
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
        #alpha = alpha * self.alpha_max
        # Observe the qk, q˙k from sk
        self.q = joint_pos
        self.dq = joint_vel

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
        ctrl_action = self.acc_to_ctrl_action(ddq)

        return ctrl_action

    def acc_to_ctrl_action(self, ddq):
        # integrate acceleration because we do control the robot with a PD Controller
        next_dq = self.dq + ddq * self.time_step
        next_q = self.q + self.dq * self.time_step + 0.5 * ddq * (self.time_step ** 2)
        return np.concatenate((next_q, next_dq)).reshape(2, self.dims['q'])

        # return self.env.client.calculateInverseDynamics(self.env._model_map['planar_robot_1'], q, dq, ddq)

    def acc_truncation(self, dq, ddq):
        # TODO: test on correctness of this value
        acc_u = np.maximum(np.minimum(self.acc_max, -self.K_q * (dq - self.vel_max)), -self.acc_max)
        acc_l = np.minimum(np.maximum(-self.acc_max, -self.K_q * (dq + self.vel_max)), self.acc_max)
        ddq = np.clip(ddq, acc_l, acc_u)
        return ddq

    def _compute_slack_variables(self):
        self.mu = None
        if self.dims['g'] > 0:
            # TODO: check if this is correct
            mu_squared = np.maximum(-2 * self.g.fun(self.q, self.dq, origin_constr=False), 0)
            self.mu = np.sqrt(mu_squared)

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

    def ee_pose_f(self, q):
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']
        ee_pos_z = forward_kinematics(mj_model, mj_data, q)[0][2]
        return np.atleast_1d(ee_pos_z)

    def ee_pose_J_f(self, q):
        """q = self._get_pino_value(q)
        pino.framesForwardKinematics(self.pino_model, self.pino_data, q)
        ee_jac = pino.computeFrameJacobian(self.pino_model, self.pino_data, q,
                                           self.frame_idx, pino.LOCAL_WORLD_ALIGNED)[:, :self.env.n_ctrl_joints]
        J_pos = ee_jac[2]
        return np.atleast_2d(J_pos)"""
        ee_jac = self.env_info['constraints'].get('ee_constr').jacobian(q, None)
        J_pos = ee_jac[2]
        return np.atleast_2d(J_pos[:self.dims['q']])
        # return np.atleast_2d(np.array([-J_pos[:self.dims['q']], J_pos[:self.dims['q']]]))

    def ee_pos_b_f(self, q, dq):
        acc = np.zeros(6)
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']
        id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_to_xml_name(mj_data, 'ee'))
        # to use  mujoco.mjtObj.mjOBJ_SITE the value should be 6
        mujoco.mj_objectAcceleration(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, id, acc,
                                     0)  # last 3 elements are linear acceleration
        acc = acc[3:]
        return np.atleast_1d(acc[2])

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
        g = self.env_info['constraints'].get('ee_constr')
        return g.fun(q, None)[:self.ee_pos_dim_out] # dq is not used, pick only the first 3 elements (do not need height constraint)

    def ee_pos_J_g(self, q):
        """ Compute the constraint function g'(q) = 0 derivative of the end-effector """
        g = self.env_info['constraints'].get('ee_constr')
        jacobian = g.jacobian(q, None)
        return jacobian[:self.ee_pos_dim_out, :self.dims['q']] # dq is not used

    def ee_pos_b_g(self, q, dq):
        """ TODO: understand what this is: it should be dJ/dt * dq/dt  """
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']
        #acc = mujoco.mj_objectAcceleration(mj_model,mj_data, mujoco.mjtObj.mjOBJ_SITE, link_to_xml_name(mj_model, 'ee'))[3:] #last 3 elements are linear acceleration
        acc = np.zeros(6)
        id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_to_xml_name(mj_model, 'ee'))

        # to use  mujoco.mjtObj.mjOBJ_SITE the value should be 6
        # TODO: check if this works
        mujoco.mj_objectAcceleration(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, id, acc,
                                     0)  # last 3 elements are linear acceleration
        acc = acc[3:]
        J_c = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 1., 0.], [0., 0., -1.], [0., 0., 1.]])
        return J_c @ acc  # Do not understand why acc is used. acc is in theory J * ddq + dJ * dq (it's like we omit the first term)

    def joint_pos_g(self, q):
        """Compute the constraint function g(q) = 0 position of the joints"""
        # return np.array(q ** 2 - self.pino_model.upperPositionLimit ** 2)
        g = self.env_info['constraints'].get('joint_pos_constr')
        return np.array(q ** 2 - self.env_info['constraints'].get('joint_pos_constr').joint_limits[1] ** 2)

    def joint_pos_J_g(self, q):
        """Compute constraint jacobian function J_g(q)"""
        # g = self.env_info['constraints'].get('joint_pos_constr')
        # # return g.jacobian(q, None)[:3, :3]
        # jacobian = g.jacobian(q, None)
        # return jacobian[:self.dim_q, :self.dim_q]
        return 2 * np.diag(q)

    def joint_pos_b_g(self, q, dq):
        """Compute constraint b(q,dq) function. It should be equal to dJ/dt * dq/dt"""
        # return 2 * np.diag(q)
        # g = self.env_info['constraint'].get('joint_pos_constr')
        # return np.zeros(g.output_dim) # this constraint is linear so second order derivative goes to zero

        return 2 * dq ** 2

    def joint_vel_g(self, dq):
        # g = self.env_info['constraints'].get('joint_vel_constr')
        return np.array([dq ** 2 - self.env_info['constraints'].get('joint_vel_constr').joint_limits[1] ** 2])
        # return g.fun(None, dq)

    def joint_vel_J_g(self, dq):
        # g = self.env_info['constraints'].get('joint_vel_constr')
        # return g.jacobian(None, dq)
        return 2 * np.diag(dq)

    def joint_vel_b_g(self, q, dq):
        return np.zeros(3)

    def reset(self):
        self._compute_slack_variables()