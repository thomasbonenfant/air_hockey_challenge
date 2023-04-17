import numpy as np

from air_hockey_agent.agents.hit_agent import HittingAgent
from air_hockey_challenge.utils import forward_kinematics
from air_hockey_agent.agents.atacom.utils.null_space_coordinate import rref, pinv_null
from air_hockey_agent.agents.atacom.constraints import ViabilityConstraint, ConstraintsSet
from mushroom_rl.utils.spaces import Box

import mujoco
from air_hockey_challenge.utils.kinematics import link_to_xml_name

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
            Ka (array, float): the scaling factor for the viability acceleration bound
            time_step (float): the step size for time discretization
        """

        super().__init__(env, **kwargs)

        dim_q = self.env_info['robot']['n_joints']
        Kc = 240.
        timestamp = 1/240.
        # why dim out of 5?
        # cart_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=5, fun=self.cart_pos_g, J=self.cart_pos_J_g, b=self.cart_pos_b_g, K=0.5)
        cart_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=3, fun=self.cart_pos_g, J=self.cart_pos_J_g, b=self.cart_pos_b_g, K=0.5)

        # TODO: still need to check validity of fun, J, b
        joint_pos_g = ViabilityConstraint(dim_q=dim_q, dim_out=2*dim_q, fun=self.joint_pos_g, J=self.joint_pos_J_g,
                                          b=self.joint_pos_b_g, K=1.0)
        f = None
        g = ConstraintsSet(dim_q)
        g.add_constraint(cart_pos_g)
        #g.add_constraint(joint_pos_g)
        #g.add_constraint(joint_vel_g)

        # TODO: if we want to include the jerk we may want to create a accel constraint and put the value here
        acc_max = self.env_info['robot']['joint_acc_limit'][1]
        vel_max = self.env_info['constraints'].get('joint_vel_constr').joint_limits[1:] #takes only max values
        vel_max = np.squeeze(vel_max)
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
            self.s = np.zeros(self.dims['g'])

        self.dims['null'] = self.dims['q'] - self.dims['f']
        self.dims['c'] = self.dims['f'] + self.dims['g']

        if np.isscalar(Kc):
            self.K_c = np.ones(self.dims['c']) * Kc
        else:
            self.K_c = Kc

        self.q = np.zeros(self.dims['q'])
        self.dq = np.zeros(self.dims['q'])

        self._mdp_info = self.env.info.copy()
        self._mdp_info.action_space = Box(low=-np.ones(self.dims['null']), high=np.ones(self.dims['null']))

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

        self.state = self.env.reset()
        self._act_a = None
        self._act_b = None
        self._act_err = None

        self.constr_logs = list()
        # shouldn't be needed
        # self.env.step_action_function = self.step_action_function
        self.mu = 0
        

    def reset(self):
        super().reset()

    def draw_action(self, observation):
        """
        Draw an action from the agent's policy.
        :param observation: The current observation of the environment.
        """
        # Sample policy action αk ∼ π(·|sk).
        alpha = super().draw_action(observation)

        alpha = np.clip(alpha, self.env_info['robot']['joint_acc_limit'][0], self.env_info['robot']['joint_acc_limit'][1])
        alpha = alpha * self.alpha_max
        # Observe the qk, q˙k from sk
        self.q = self.get_joint_pos(observation)
        self.dq = self.get_joint_vel(observation)

        # Compute slack variable mu
        self._compute_slack_variables()

        # Compute Jc, k = Jc(qk, µk), ψk = ψ(qk, q˙k), ck = c(qk, q˙k, µk)
        Jc, psi = self._construct_Jc_psi(self.q, self.mu, self.dq)

        Jc_inv, Nc = pinv_null(Jc)
        # Compute the RCEF of tangent space basis of NcR
        # Nc = rref(Nc[:, :self.dims['null']], row_vectors=False, tol=0.05)
        Nc = rref(Nc[:, :6], row_vectors=False, tol=0.05)

        # Compute the tangent space acceleration [q¨k µ˙ k].T ← −J^†_c,k [K_cck + ψ_k] + N^R_c α_k
        self._act_a = -Jc_inv @ psi
        self._act_b = Nc @ alpha
        self._act_err = self._compute_error_correction(self.q, self.dq, self.s, Jc_inv)
        ddq_ds = self._act_a + self._act_b + self._act_err

        self.s += ddq_ds[self.dims['q']:(self.dims['q'] + self.dims['g'])] * self.time_step
        # Clip the joint acceleration q¨k ← clip(q¨k, al, au)
        ddq = self.acc_truncation(self.dq, ddq_ds[:self.dims['q']])
        # Integrate the slack variable µk+1 = µk + µ˙ k∆T
        ctrl_action = self.acc_to_ctrl_action(ddq)
        return ctrl_action

    def acc_to_ctrl_action(self, ddq):
        # integrate acceleration because we do control the robot with a PD Controller
        next_dq = self.dq + ddq * self.time_step
        next_q = self.q + self.dq * self.time_step + 0.5 * ddq * (self.time_step ** 2)
        return np.concatenate((next_q, next_dq)).reshape(2,3)

        # return self.env.client.calculateInverseDynamics(self.env._model_map['planar_robot_1'], q, dq, ddq)

    def acc_truncation(self, dq, ddq):
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

    def _compute_error_correction(self, q, dq, s, Jc_inv, act_null=None):
        q_tmp = q.copy()
        dq_tmp = dq.copy()
        s_tmp = None

        if self.dims['g'] > 0:
            s_tmp = s.copy()

        if act_null is not None:
            q_tmp += dq_tmp * self.time_step + act_null[:self.dims['q']] * self.time_step ** 2 / 2
            dq_tmp += act_null[:self.dims['q']] * self.time_step
            if self.dims['g'] > 0:
                s_tmp += act_null[self.dims['q']:self.dims['q'] + self.dims['g']] * self.time_step

        return -Jc_inv @ (self.K_c * self._compute_c(q_tmp, dq_tmp, s_tmp, origin_constr=False))

    def cart_pos_g(self, q):
        """ Compute the constraint function g(q) = 0 position of the end-effector"""
        #ee_pos = forward_kinematics(self.robot_model, self.robot_data, q)
        #ee_pos_world = ee_pos + self.env.agents[0]['frame'][:2, 3]
        #g_1 = - ee_pos_world[0] - (self.env.env_spec['table']['length'] / 2 - self.env.env_spec['mallet']['radius'])
        #g_2 = - ee_pos_world[1] - (self.env.env_spec['table']['width'] / 2 - self.env.env_spec['mallet']['radius'])
        #g_3 = ee_pos_world[1] - (self.env.env_spec['table']['width'] / 2 - self.env.env_spec['mallet']['radius'])
        #return np.array([g_1, g_2, g_3])

        g = self.env_info['constraints'].get('ee_constr')
        return g.fun(q, None)[:3] # dq is not used, pick only the first 3 elements (do not need height constraint)

    def cart_pos_J_g(self, q):
        """ Compute the constraint function g'(q) = 0 derivative of the end-effector """
        #ee_jac = self.env_info['constraints'].get('ee_constraint').jacobian(q)
        #J_c = np.array([[-1., 0.], [0., -1.], [0., 1.]])
        #return J_c @ ee_jac
        
        g = self.env_info['constraints'].get('ee_constr')
        return g.jacobian(q, None)[:3,:3] # dq is not used

    def cart_pos_b_g(self, q, dq):
        """ TODO: understand what this is: it should be dJ/dt * dq/dt  """
        #ee_pos = forward_kinematics(self.robot_model, self.robot_data, q)
        # probably not correct
        g = self.env_info['constraints'].get('ee_constr').jacobian(q, dq)
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
        # why in atacom takes only 2 acceleration?
        return J_c @ acc[:2] #Do not understand why acc is used. acc is in theory J * ddq + dJ * dq (it's like we omit the first term)

    def joint_pos_g(self, q):
        """Compute the constraint function g(q) = 0 position of the joints"""
        #return np.array(q ** 2 - self.pino_model.upperPositionLimit ** 2)
        g = self.env_info['constraints'].get('joint_pos')
        return g.fun(q,None)


    def joint_pos_J_g(self, q):
        """Compute constraint jacobian function J_g(q)"""
        g = self.env_info['constraints'].get('joint_pos')
        return g.jacobian(q,None)

    def joint_pos_b_g(self, q, dq):
        """Compute constraint b(q,dq) function. It should be equal to dJ/dt * dq/dt"""
        #return 2 * np.diag(q)
        g = self.env_info['constraint'].get('joint_pos') 
        return np.zeros(g.output_dim) #this constraint is linear so second order derivative goes to zero

    #next constraint is not used
    #def joint_vel_g(self, q, dq):
    #    #return np.array([dq ** 2 - self.pino_model.velocityLimit ** 2])

    #def joint_vel_A_g(self, q, dq):
    #    return 2 * np.diag(dq)

    #def joint_vel_b_g(self, q, dq):
    #    return np.zeros(3)


