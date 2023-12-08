from stable_baselines3 import PPO, SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
import hydra
from omegaconf import OmegaConf
from envs.env_maker import create_producer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from exp_utils.utils import get_callbacks
import os

alg_dict = {
    'sac': SAC,
    'ppo': PPO
}


@hydra.main(version_base=None, config_path="conf", config_name="load")
def main(cfg):
    path = cfg['path']

    config = OmegaConf.load(os.path.join(path, 'config.yaml'))

    env_args = config['environment']

    # Create Environment
    env_producer = create_producer(env_args)
    env = make_vec_env(env_producer,
                       n_envs=config['parallel'],
                       vec_env_cls=SubprocVecEnv,
                       monitor_dir=path,
                       )

    tb_log_dir = os.path.join(path, 'tb_logs')

    callbacks = get_callbacks(config)

    alg_cls = alg_dict[config.algorithm.alg]

    model = alg_cls.load(os.path.join(path, 'model'), env=env)
    if isinstance(model, OffPolicyAlgorithm):
        model.load_replay_buffer(os.path.join(path, 'replay_buffer'))

    model.learn(**cfg.learn, tb_log_name=tb_log_dir, callback=callbacks, reset_num_timesteps=False)
    model.save(os.path.join(path, "model.zip"))

    if isinstance(model, OffPolicyAlgorithm):
        model.save_replay_buffer(os.path.join(path, "replay_buffer"))


if __name__ == '__main__':
    main()










