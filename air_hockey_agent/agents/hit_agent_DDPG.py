import numpy as np
from mushroom_rl.algorithms.actor_critic import DDPG

from air_hockey_agent.actor_critic_network import DDPGCriticNetwork, DDPGActorNetwork
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
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


class HittingAgent(AgentBase, DDPG):
    def __init__(self, env, **kwargs):
        params = self.configure_DDPG(env['rl_info'], **kwargs)[:-1]
        alg_params = params[-1]
        DDPG.__init__(self, *params[:-1], **alg_params)
        kwargs['no_initialization'] = True
        AgentBase.__init__(self, env, **kwargs)
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
        return DDPG.draw_action(self, observation)

    def configure_DDPG(self, mdp_info, actor_lr=3e-4, critic_lr=3e-4, n_features=[64, 64], batch_size=64,
                        initial_replay_size=400, max_replay_size=200000, tau=1e-3,
                        warmup_transitions=10000, lr_alpha=3e-4, target_entropy=-6, use_cuda=False,
                        **kwargs):
        policy_params = dict(
            sigma=np.ones(1) * .2,
            theta=0.15,
            dt=1e-2)

        actor_params = dict(
            network=DDPGActorNetwork,
            input_shape=mdp_info.observation_space.shape,
            output_shape=mdp_info.action_space.shape,
            action_scaling=(mdp_info.action_space.high - mdp_info.action_space.low) / 2,
            n_features=n_features,
            use_cuda=use_cuda)

        actor_optimizer = {
            'class': optim.Adam,
            'params': {'lr': actor_lr}}

        critic_params = dict(
            network=DDPGCriticNetwork,
            optimizer={'class': optim.Adam,
                       'params': {'lr': critic_lr}},
            loss=F.mse_loss,
            n_features=n_features,
            batch_size=batch_size,
            input_shape=(mdp_info.observation_space.shape[0] + mdp_info.action_space.shape[0],),
            action_shape=mdp_info.action_space.shape,
            output_shape=(1,),
            action_scaling=(mdp_info.action_space.high - mdp_info.action_space.low) / 2,
            use_cuda=use_cuda)
        print(mdp_info.observation_space.shape[0] + mdp_info.action_space.shape[0])
        alg_params = dict(
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            tau=tau)

        build_params = dict(compute_entropy_with_states=False,
                            compute_policy_entropy=False)

        return mdp_info, OrnsteinUhlenbeckPolicy, policy_params, actor_params, actor_optimizer, critic_params, alg_params, build_params

