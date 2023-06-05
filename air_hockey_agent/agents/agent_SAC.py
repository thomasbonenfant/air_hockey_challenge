import numpy as np
from mushroom_rl.algorithms.actor_critic import SAC

from air_hockey_agent.actor_critic_network import SACCriticNetwork, SACActorNetwork
from air_hockey_challenge.utils.kinematics import inverse_kinematics
from air_hockey_challenge.framework.agent_base import AgentBase
from torch import optim
import torch.nn.functional as F

spec = []


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    return AgentAirhockeySAC(env_info, **kwargs)


class AgentAirhockeySAC(SAC):
    def __init__(self, env, **kwargs):
        params = self.configure_SAC(env['rl_info'], **kwargs)[:-1]
        alg_params = params[-1]
        super().__init__(*params[:-1], **alg_params)
        self.env_info = env
        self.new_start = True
        self.hold_position = None

        self.path = kwargs['path'] if 'path' in kwargs else [[-0.8, 0, 0]]

        self.steps_per_action = kwargs['steps_per_action'] if 'steps_per_action' in kwargs else 50

        self.x_shifter_lb = self.env_info["constraints"].get("ee_constr").x_lb
        self.x_shifter_ub = -self.env_info['robot']['base_frame'][0][0, 3]
        self.y_shifter_lb = self.env_info["constraints"].get("ee_constr").y_lb
        self.y_shifter_ub = self.env_info["constraints"].get("ee_constr").y_ub
        self.x_mult = (self.x_shifter_ub - self.x_shifter_lb)
        self.y_mult = (self.y_shifter_ub - self.y_shifter_lb)

        self.step = 0
        self.path_idx = 0
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def save(self, path, full_save=False):
        SAC.save(self, path, full_save)

    def load(self, path):
        SAC.load(path)

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        """
        Draw an action from the agent's policy.
        :param observation: The current observation of the environment.
        """
        action = super().draw_action(observation)
        return action

    def configure_SAC(self, mdp_info, actor_lr=3e-4, critic_lr=3e-4, n_features=[64, 64], batch_size=64,
                        initial_replay_size=5000, max_replay_size=200000, tau=1e-3,
                        warmup_transitions=10000, lr_alpha=3e-4, target_entropy=-6, use_cuda=False,
                        **kwargs):
        actor_mu_params = dict(network=SACActorNetwork,
                               input_shape=mdp_info.observation_space.shape,
                               output_shape=mdp_info.action_space.shape,
                               n_features=n_features,
                               use_cuda=use_cuda)
        actor_sigma_params = dict(network=SACActorNetwork,
                                  input_shape=mdp_info.observation_space.shape,
                                  output_shape=mdp_info.action_space.shape,
                                  n_features=n_features,
                                  use_cuda=use_cuda)

        actor_optimizer = {'class': optim.Adam,
                           'params': {'lr': actor_lr}}
        critic_params = dict(network=SACCriticNetwork,
                             input_shape=(mdp_info.observation_space.shape[0] + mdp_info.action_space.shape[0],),
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': critic_lr}},
                             loss=F.mse_loss,
                             n_features=n_features,
                             output_shape=(1,),
                             use_cuda=use_cuda)

        alg_params = dict(initial_replay_size=initial_replay_size,
                          max_replay_size=max_replay_size,
                          batch_size=batch_size,
                          warmup_transitions=warmup_transitions,
                          tau=tau,
                          lr_alpha=lr_alpha,
                          critic_fit_params=None,
                          target_entropy=target_entropy)

        build_params = dict(compute_entropy_with_states=True,
                            compute_policy_entropy=True)

        return mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params, build_params
        #SAC(mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params,**alg_params)



    def get_puck_state(self, obs):
        """
        Get the puck's position and velocity from the observation

        Args
        ----
        obs: numpy.ndarray
            observed state.

        Returns
        -------
        joint_pos: numpy.ndarray, (3,)
            [x, y, theta] position of the puck w.r.t robot's base frame
        joint_vel: numpy.ndarray, (3,)
            [vx, vy, dtheta] position of the puck w.r.t robot's base frame

        """
        return self.get_puck_pos(obs), self.get_puck_vel(obs)

    def get_joint_state(self, obs):
        """
        Get the joint positions and velocities from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        joint_pos: numpy.ndarray
            joint positions of the robot;
        joint_vel: numpy.ndarray
            joint velocities of the robot.

        """
        return self.get_joint_pos(obs), self.get_joint_vel(obs)

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
        return forward_kinematics(self.robot_model, self.robot_data, self.get_joint_pos(obs))
