from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from my_scripts.callbacks import EpisodicInfoCallback
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
from envs import create_producer
import os


def get_callbacks(cfg):
    callback_list = []
    callback_config = cfg['callbacks']

    for callback_name, callback_args in callback_config.items():

        args = dict(callback_args)

        if callback_name == 'eval':
            cb = EvalCallback

            env_producer = create_producer(cfg['environment'])
            eval_env = make_vec_env(env_producer, n_envs=cfg['parallel'], vec_env_cls=SubprocVecEnv)
            args['eval_env'] = eval_env
        elif callback_name == 'checkpoint':
            cb = CheckpointCallback

            args['save_path'] = os.path.join(cfg['log_dir'], args['save_path'])

        elif callback_name == 'info_log':
            cb = EpisodicInfoCallback

        else:
            raise NotImplementedError

        callback_list.append(cb(**args))

    return callback_list



