from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from envs import create_producer
from stable_baselines3 import PPO, SAC, DQN

from my_scripts.utils import create_log_directory, get_callbacks
from my_scripts.callbacks.reward_logs_callback import RewardLogsCallback
import os

from omegaconf import OmegaConf
import hydra
import random


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    # print(OmegaConf.to_yaml(cfg))
    if 'environment' not in cfg:
        print('Specify an environment')
        return

    env_args = cfg['environment']
    alg_args = cfg['algorithm']

    seed = cfg['seed'] if cfg['seed'] else random.randint(0, 99999)

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

    callback_list = get_callbacks(cfg)

    # try to load an existing experiment:
    if cfg.algorithm.alg == 'ppo':
        alg_cls = PPO
    elif cfg.algorithm.alg == 'sac':
        alg_cls = SAC
    elif cfg.algorithm.alg == 'dqn':
        alg_cls = DQN
    else:
        raise NotImplementedError

    # remove alg name
    alg_args = {k:v for k,v in alg_args.items() if k != 'alg'}


    model = alg_cls(env=env, seed=seed, **alg_args)

    model.learn(**cfg.learn, tb_log_name=tb_log_dir, callback=callback_list)

    model.save(os.path.join(log_dir, "model.zip"))

    if cfg.algorithm.alg in ('dqn', 'sac'):
        model.save_replay_buffer(os.path.join(log_dir, "replay_buffer"))


if __name__ == '__main__':
    main()
