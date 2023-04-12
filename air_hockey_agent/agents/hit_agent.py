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
    return HittingAgent(env_info, **kwargs)


class HittingAgent(AgentBase, SAC):
    def __init__(self, env, **kwargs):
        params = self.configure_SAC(env.info, **kwargs)[:-1]
        alg_params = params[-1]
        SAC.__init__(self, *params[:-1], **alg_params)
        kwargs['no_initialization'] = True
        AgentBase.__init__(self, env.env_info, **kwargs)
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

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        """
        Draw an action from the agent's policy.
        :param observation: The current observation of the environment.
        """
        if self.new_start:
            self.new_start = False
            #print(self.env_info['robot']['base_frame'][0][0, 3])

        q = observation[self.env_info['joint_pos_ids']]
        dq = observation[self.env_info['joint_vel_ids']]
        #print(q, dq)
        #print(self.env_info["constraints"].get("ee_constr").fun(q, dq))
        #print(observation)
        action = SAC.draw_action(self, observation)

        pos = action * [self.x_mult, self.y_mult, 1] + [self.x_shifter_lb, self.y_shifter_lb, -0.5]

        #print(pos)
        #action = inverse_kinematics(self.env_info['robot']['robot_model'], self.env_info['robot']['robot_data'], pos)[1]

        return np.array([action, np.ones(3) * 0.05])

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