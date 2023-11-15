from air_hockey_agent.agents.agents import Agent
from omegaconf import OmegaConf
import os
from air_hockey_agent.utils.sb3_variant_util import get_configuration
from envs.env_maker import create_producer
from stable_baselines3 import PPO, SAC

alg_dict = {
    'sac': SAC,
    'ppo': PPO
}


class AgentSB3OldEnv(Agent):
    def __init__(self, path, env_info, random=False, **kwargs):
        # dir_path = os.path.dirname(os.path.abspath(__file__))

        # path = os.path.join(dir_path, path)
        agent_path = os.path.join(path, 'best_model')

        env_args, alg = get_configuration(path)

        # create a environment to get its observation and action space
        self.env12 = create_producer(env_args)()
        self.env12 = self.env12.env # skips the EnvCompatibility wrapper
        self.trainer = TrainerAdapter(alg_dict[alg].load(agent_path), random=random)

        self.env_label = "tournament"
        env_args['env'] = env_args['env'].split('oac_')[1]
        variant = env_args

        super().__init__(env_info, variant, **kwargs)

        self.policy = self.trainer.policy


class TrainerAdapter(object):
    def __init__(self, sb3_agent, random=False):
        self.agent = sb3_agent
        self.random = random

    @property
    def policy(self):
        return self

    def get_action(self, state, deterministic: bool):
        if self.random:
            action = self.agent.action_space.sample()
        else:
            action, _ = self.agent.predict(state, deterministic=deterministic)
        return action, None
