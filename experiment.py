from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from envs import create_producer
from stable_baselines3 import PPO, SAC, DQN, HerReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from exp_utils import create_log_directory, get_callbacks
import os

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import random

alg_dict = {
    'sac': SAC,
    'ppo': PPO,
    'dqn': DQN,
}


def configdict_to_dict(config_dict: DictConfig):
    if isinstance(config_dict, DictConfig) or isinstance(config_dict, dict):
        config_dict = dict(config_dict)

        for k in config_dict:
            config_dict[k] = configdict_to_dict(config_dict[k])
    return config_dict


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # print(OmegaConf.to_yaml(cfg))
    if 'environment' not in cfg:
        print('Specify an environment')
        return

    env_args = cfg['environment']
    alg_args = cfg['algorithm']

    seed = cfg['seed'] if cfg['seed'] else random.randint(0, 999999)

    log_dir = create_log_directory(cfg, seed)
    cfg['log_dir'] = log_dir

    tb_log_dir = os.path.join(log_dir, 'tb_logs')

    print(f'Configuration:\n {OmegaConf.to_yaml(cfg)}')

    # dumps configuration
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    env_producer = create_producer(env_args)
    env = make_vec_env(env_producer,
                       n_envs=cfg['parallel'],
                       vec_env_cls=SubprocVecEnv,
                       monitor_dir=log_dir,
                       )

    callback_list = get_callbacks(cfg, env=env)

    alg_cls = alg_dict[cfg.algorithm.alg]

    # remove alg name from arguments
    alg_args = {k: v for k, v in alg_args.items() if k != 'alg'}
    if 'train_freq' in alg_args:
        alg_args['train_freq'] = tuple(alg_args['train_freq'])  # tuple is required and not list
    if 'replay_buffer' in alg_args:
        alg_args = configdict_to_dict(alg_args)

        if alg_args['replay_buffer']['replay_buffer_class'] == 'her':
            alg_args['replay_buffer']['replay_buffer_class'] = HerReplayBuffer
        for k, v in alg_args['replay_buffer'].items():
            alg_args[k] = v
        del alg_args['replay_buffer']

    model = alg_cls(env=env, seed=seed, **alg_args)
    model.learn(**cfg.learn, tb_log_name=tb_log_dir, callback=callback_list, reset_num_timesteps=False)
    model.save(os.path.join(log_dir, "model.zip"))

    if isinstance(model, OffPolicyAlgorithm):
        model.save_replay_buffer(os.path.join(log_dir, "replay_buffer"))


if __name__ == '__main__':
    main()
