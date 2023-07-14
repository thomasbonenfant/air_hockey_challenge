import numpy as np
from mushroom_rl.utils.spaces import Box

from air_hockey_agent.agents.agent_SAC import AgentAirhockeySAC
from air_hockey_agent.agents.atacom.utils.null_space_coordinate import rref, pinv_null
from air_hockey_agent.agents.atacom.constraints import ViabilityConstraint, ConstraintsSet

import mujoco
from air_hockey_challenge.utils.kinematics import link_to_xml_name, forward_kinematics, jacobian


class ATACOMAgent(AgentAirhockeySAC):

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
        self.n_non_controllable_joints = 0
        self.n_equality_constraints = 1
        env['rl_info'].action_space = Box(low=-1, high=1, shape=(7-self.n_equality_constraints,))
        super().__init__(env, **kwargs)
        dim_q = self.env_info['robot']['n_joints'] #- self.n_non_controllable_joints

        Kc = 50.
        timestamp = 1 / 50.

        ee_pos_f = ViabilityConstraint(dim_q=dim_q, dim_out=1, fun=self.ee_pose_f, J=self.ee_pose_J_f,
                                       b=self.ee_pos_b_f, K=1)

        self.ee_pos_dim_out = 5
        # ee_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=5, fun=self.ee_pos_g, J=self.ee_pos_J_g, b=self.ee_pos_b_g, K=0.5)
        ee_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=self.ee_pos_dim_out, fun=self.ee_pos_g, J=self.ee_pos_J_g,
                                       b=self.ee_pos_b_g, K=1.)

        joint_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=dim_q, fun=self.joint_pos_g, J=self.joint_pos_J_g,
                                          b=self.joint_pos_b_g, K=1.0)

        joint_vel_g = ViabilityConstraint(dim_q=dim_q, dim_out=3, fun=self.joint_vel_g, J=self.joint_vel_J_g,
                                          b=self.joint_vel_b_g, K=1.0)

        f = ConstraintsSet(dim_q)
        g = ConstraintsSet(dim_q)
        if self.n_equality_constraints > 0:
            f.add_constraint(ee_pos_f)
        g.add_constraint(ee_pos_g)
        g.add_constraint(joint_pos_g)
        #g.add_constraint(joint_vel_g)

        # TODO: if we want to include the jerk we may want to create a accel constraint and put the value here
        acc_max = self.env_info['robot']['joint_acc_limit'][1][:dim_q]
        vel_max = self.env_info['constraints'].get('joint_vel_constr').joint_limits[1]  # takes only positive constraints
        vel_max = np.squeeze(vel_max)[:dim_q]

        self.robot_model = self.env_info['robot']['robot_model']
        self.robot_data = self.env_info['robot']['robot_data']
        self.joint_pos_constraint = self.env_info['constraints'].get('joint_pos_constr').joint_limits[1][:dim_q]
        self.joint_vel_constraint = self.env_info['constraints'].get('joint_vel_constr').joint_limits[1][:dim_q]

        Kq = 4 * acc_max / vel_max

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

        self.q = np.zeros(self.dims['q'])
        self.dq = np.zeros(self.dims['q'])

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
        self.mu = 0
        self.first_step = True

    def reset(self):
        self.first_step = True
        super().reset()

    def mid_point(self, a, b):
        return (a + b) / 2

    def draw_action(self, observation):
        action = super().draw_action(observation)
        alpha = action * self.alpha_max

        self.q = self.get_joint_pos(observation)[:self.dims['q']]
        self.dq = self.get_joint_vel(observation)[:self.dims['q']]

        if self.first_step:
            self.first_step = False
            self._compute_slack_variables()

        # Compute Jc, k = Jc(qk, µk), ψk = ψ(qk, q˙k), ck = c(qk, q˙k, µk)
        Jc, psi = self._construct_Jc_psi(self.q, self.mu, self.dq)
        Jc_inv, Nc = pinv_null(Jc)
        # Compute the RCEF of tangent space basis of NcR
        #Nc = np.array(sympy.Matrix(Nc[:, :self.dims['null']]).rref()[0], dtype=np.float64)
        Nc = rref(Nc[:, :self.dims['null']], row_vectors=False, tol=0.05)
        # Compute the tangent space acceleration [q¨k µ˙ k].T ← −J^†_c,k [K_cck + ψ_k] + N^R_c α_k
        self._act_a = -Jc_inv @ psi
        self._act_b = Nc @ alpha
        self._act_err = self._compute_error_correction(self.q, self.dq, self.mu, Jc_inv)
        ddq_ds = self._act_a + self._act_b + self._act_err

        # Integrate the slack variable µk+1 = µk + µ˙ k∆T
        self.mu += ddq_ds[self.dims['q']:(self.dims['q'] + self.dims['g'])] * self.time_step
        # Clip the joint acceleration q¨k ← clip(q¨k, al, au)
        ddq = self.acc_truncation(self.dq, ddq_ds[:self.dims['q']])

        ctrl_action = self.acc_to_ctrl_action(ddq)  # .reshape((6,))
        #ctrl_action[0, 6] = self._compute_joint_7(ctrl_action[0])
        return ctrl_action

    def acc_to_ctrl_action(self, ddq):
        # integrate acceleration because we do control the robot with a PD Controller

        next_dq = self.dq + ddq * self.time_step
        next_q = self.q + self.dq * self.time_step + 0.5 * ddq * (self.time_step ** 2)
        return np.concatenate((next_q, next_dq)).reshape(2, self.dims['q'])

    def acc_truncation(self, dq, ddq):
        acc_u = np.maximum(np.minimum(self.acc_max, -self.K_q * (dq - self.vel_max)), -self.acc_max)
        acc_l = np.minimum(np.maximum(-self.acc_max, -self.K_q * (dq + self.vel_max)), self.acc_max)
        ddq = np.clip(ddq, acc_l, acc_u)
        return ddq

    def _compute_slack_variables(self):
        self.mu = None
        if self.dims['g'] > 0:
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

    def _compute_c(self, q, dq, mu, origin_constr=False):
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
                c[idx_0:idx_1] = self.g.fun(q, dq, origin_constr) + 1 / 2 * mu ** 2
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

    def ee_pose_f(self, q):
        ee_pos_z = forward_kinematics(self.robot_model, self.robot_data, q)[0][2]
        # return np.atleast_1d(ee_pos_z - self.env.env_spec['universal_height'])
        return np.atleast_1d(ee_pos_z - 0.0645)
        #return np.atleast_1d(np.array([-ee_pos_z, ee_pos_z]))

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
        #return np.atleast_2d(np.array([-J_pos[:self.dims['q']], J_pos[:self.dims['q']]]))

    def ee_pos_b_f(self, q, dq):
        """q = self._get_pino_value(q)
        dq = self._get_pino_value(dq)
        pino.forwardKinematics(self.pino_model, self.pino_data, q, dq)
        acc = pino.getFrameClassicalAcceleration(self.pino_model, self.pino_data, self.frame_idx,
                                                 pino.LOCAL_WORLD_ALIGNED).vector
        b_pos = acc[2]
        return np.atleast_1d(b_pos)"""
        acc = np.zeros(6)
        id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_to_xml_name(self.robot_model, 'ee'))

        # to use  mujoco.mjtObj.mjOBJ_SITE the value should be 6
        mujoco.mj_objectAcceleration(self.robot_model, self.robot_data, mujoco.mjtObj.mjOBJ_BODY, id, acc, 0)  # last 3 elements are linear acceleration
        acc = acc[:3]
        return np.atleast_1d(acc[2])
        #return np.atleast_1d(np.array([-acc[2],acc[2]]))

    def ee_pos_g(self, q):
        """ Compute the constraint function g(q) = 0 position of the end-effector"""
        # ee_pos = forward_kinematics(self.robot_model, self.robot_data, q)
        # ee_pos_world = ee_pos + self.env.agents[0]['frame'][:2, 3]
        # g_1 = - ee_pos_world[0] - (self.env.env_spec['table']['length'] / 2 - self.env.env_spec['mallet']['radius'])
        # g_2 = - ee_pos_world[1] - (self.env.env_spec['table']['width'] / 2 - self.env.env_spec['mallet']['radius'])
        # g_3 = ee_pos_world[1] - (self.env.env_spec['table']['width'] / 2 - self.env.env_spec['mallet']['radius'])
        # return np.array([g_1, g_2, g_3])

        ee_constr = self.env_info['constraints'].get('ee_constr')

        return ee_constr.fun(q, None)[:self.ee_pos_dim_out]  # dq is not used, pick only the first 3 elements (do not need height constraint)

    def ee_pos_J_g(self, q):
        """ Compute the constraint function g'(q) = 0 derivative of the end-effector """
        # ee_jac = self.env_info['constraints'].get('ee_constraint').jacobian(q)
        # J_c = np.array([[-1., 0.], [0., -1.], [0., 1.]])
        # return J_c @ ee_jac
        jac_ee = jacobian(self.robot_model, self.robot_data, q, 'ee')
        jac_4 = jacobian(self.robot_model, self.robot_data, q, '4')
        jac_7 = jacobian(self.robot_model, self.robot_data, q, '7')

        return np.vstack([-jac_ee[0], -jac_ee[1], jac_ee[1], -jac_4[2], -jac_7[2]])
        #ee_constr = self.env_info['constraints'].get('ee_constr')
        #return ee_constr.jacobian(q, None)[:self.ee_pos_dim_out, :self.dims['q']]  # dq is not used

    def ee_pos_b_g(self, q, dq):
        """ TODO: understand what this is: it should be dJ/dt * dq/dt  """
        # ee_pos = forward_kinematics(self.robot_model, self.robot_data, q)
        # pino.getFrameClassicalAcceleration(self.pino_model, self.pino_data, self.pino_model.nframes - 1,pino.LOCAL_WORLD_ALIGNED).vector

        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']
        # acc = mujoco.mj_objectAcceleration(mj_model,mj_data, mujoco.mjtObj.mjOBJ_SITE, link_to_xml_name(mj_model, 'ee'))[3:] #last 3 elements are linear acceleration
        acc = np.zeros(6)
        acc4 = np.zeros(6)
        acc7 = np.zeros(6)
        id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_to_xml_name(mj_model, 'ee'))
        id4 = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_to_xml_name(self.robot_model, '4'))
        id7 = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_to_xml_name(self.robot_model, '7'))

        # to use  mujoco.mjtObj.mjOBJ_SITE the value should be 6
        # TODO: check if this works
        mujoco.mj_objectAcceleration(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, id, acc, 0)  # last 3 elements are linear acceleration
        mujoco.mj_objectAcceleration(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, id4, acc4, 0)
        mujoco.mj_objectAcceleration(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, id7, acc7, 0)
        acc = acc[3:]

        J_c = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 1., 0.], [0., 0., -1.], [0., 0., 1.]])[:self.ee_pos_dim_out]
        #return J_c @ acc  # Do not understand why acc is used. acc is in theory J * ddq + dJ * dq (it's like we omit the first term)
        return np.array([-acc[0], -acc[1], acc[1], -acc4[2], -acc7[2]])

    def joint_pos_g(self, q):
        """Compute the constraint function g(q) = 0 position of the joints"""
        # return np.array(q ** 2 - self.pino_model.upperPositionLimit ** 2)
        # g = self.env_info['constraints'].get('joint_pos_constr')
        # return g.fun(q, None)[:3]
        return np.array(q ** 2 - self.joint_pos_constraint ** 2)

    def joint_pos_J_g(self, q):
        """Compute constraint jacobian function J_g(q)"""
        # g = self.env_info['constraints'].get('joint_pos_constr')
        # return g.jacobian(q, None)[:3, :3]
        return 2 * np.diag(q)

    def joint_pos_b_g(self, q, dq):
        """Compute constraint b(q,dq) function. It should be equal to dJ/dt * dq/dt"""
        # return 2 * np.diag(q)
        # g = self.env_info['constraint'].get('joint_pos_constr')
        return 2 * dq ** 2

    def joint_vel_g(self, dq):
        # g = self.env_info['constraints'].get('joint_vel_constr')
        return np.array([dq ** 2 - self.joint_vel_constraint ** 2])
        # return g.fun(None, dq)

    def joint_vel_J_g(self, dq):
        # g = self.env_info['constraints'].get('joint_vel_constr')
        # return g.jacobian(None, dq)
        return 2 * np.diag(dq)

    def joint_vel_b_g(self, q, dq):
        return np.zeros(3)

    def _compute_joint_7(self, joint_state):
        q_cur = joint_state.copy()
        q_cur_7 = q_cur[6]
        q_cur[6] = 0.
        rotation = 1

        position = 0
        f_cur = forward_kinematics(self.robot_model, self.robot_data, q_cur)[rotation]
        z_axis = np.array([0., 0., -1.])

        y_des = np.cross(z_axis, f_cur[:, 2])
        y_des_norm = np.linalg.norm(y_des)
        if y_des_norm > 1e-2:
            y_des = y_des / y_des_norm
        else:
            y_des = f_cur[:, 2]

        target = np.arccos(f_cur[:, 1].dot(y_des))

        axis = np.cross(f_cur[:, 1], y_des)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-2:
            axis = axis / axis_norm
        else:
            axis = np.array([0., 0., 1.])

        target = target * axis.dot(f_cur[:, 2])

        if target - q_cur_7 > np.pi / 2:
            target -= np.pi
        elif target - q_cur_7 < -np.pi / 2:
            target += np.pi

        return np.atleast_1d(target)
