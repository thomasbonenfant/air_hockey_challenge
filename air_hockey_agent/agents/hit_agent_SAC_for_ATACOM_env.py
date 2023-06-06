import numpy as np
from mushroom_rl.utils.spaces import Box

from air_hockey_agent.agents.agent_SAC import AgentAirhockeySAC
from air_hockey_challenge.utils import forward_kinematics, inverse_kinematics


class HittingAgentSAC4ATACOM(AgentAirhockeySAC):

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

        #env['rl_info'].action_space = Box(env['robot']['joint_acc_limit'][0], env['robot']['joint_acc_limit'][1])
        env['rl_info'].action_space = Box(np.array([-1, -1, -1, -1, -1, -1, -1]), np.array([1, 1, 1, 1, 1, 1, 1]))
        #env['rl_info'].observation_space = Box(np.append(env['rl_info'].observation_space.low, 0),
        #                                       np.append(env['rl_info'].observation_space.high, 1))
        super().__init__(env, **kwargs)
        #self.add_preprocessor(self.add_observation_preprocessors)
        self.was_hit = 0
        self.time_step = 1/50

        self.last_actions = []

    def reset(self):
        self.was_hit = 0
        super().reset()

    def episode_start(self):
        self.reset()
        super().episode_start()

    """def add_observation_preprocessors(self, state):
        mj_model = self.env_info['robot']['robot_model']
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
        print(mujoco.mj_collision(mj_model, mj_data))
        # TODO: implement better contact mechanism
        #if self.was_hit == 0:
        #    self.was_hit = 0 if np.sum(self.get_puck_vel(state)[:2]) == 0 else 1
            #if self.was_hit == 1:
            #    print("HIT")

        #return np.append(state, self.was_hit)
        return state"""

    def draw_action(self, observation):
        #print(f"\n\npreATACOM: obs : {observation}")
        return super().draw_action(observation)

    """def draw_action(self, observation):
        self.q = self.get_joint_pos(observation)
        self.dq = self.get_joint_vel(observation)
        alpha = super().draw_action(observation)
        self.last_actions.append(alpha)
        return self.acc_to_ctrl_action(alpha)

    def acc_to_ctrl_action(self, ddq):
        # integrate acceleration because we do control the robot with a PD Controller
        #self.robot_data.qacc[:] = ddq
        #self.robot_data.qvel[:] = self.robot_data.qacc[:] * self.time_step
        #mujoco.mj_inverse(self.robot_model, self.robot_data)
        #ddq2 = self.robot_data.qfrc_inverse[:]

        next_dq = self.dq + ddq * self.time_step
        next_q = self.q + self.dq * self.time_step + 0.5 * ddq * (self.time_step ** 2)
        return np.concatenate((next_q, next_dq)).reshape(2,3)
    
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

        super().fit(dataset, **info)"""
