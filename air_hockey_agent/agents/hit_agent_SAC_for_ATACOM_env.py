import numpy as np
from mushroom_rl.utils.spaces import Box

from air_hockey_agent.agents.hit_agent_SAC import HittingAgent


class HittingAgentSAC4ATACOM(HittingAgent):

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
        env['rl_info'].observation_space = Box(np.append(env['rl_info'].observation_space.low, 0),
                                               np.append(env['rl_info'].observation_space.high, 1))
        super().__init__(env, **kwargs)
        self.add_preprocessor(self.add_observation_preprocessors)
        self.was_hit = 0

    def reset(self):
        self.was_hit = 0
        super().reset()

    def episode_start(self):
        self.reset()
        super().episode_start()

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

    def draw_action(self, observation):
        return super().draw_action(observation)
