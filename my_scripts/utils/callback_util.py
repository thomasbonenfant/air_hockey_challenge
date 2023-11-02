from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
from envs import create_producer


def get_callbacks(cfg):
    callback_list = []
    callback_config = cfg['callbacks']

    for callback_name, callback_args in callback_config.items():
        if callback_name == 'eval':
            cb = EvalCallback

        else:
            raise NotImplementedError

        args = dict(callback_args)

        # create eval environment if it is an eval callback
        if cb == EvalCallback:
            env_producer = create_producer(cfg['environment'])
            eval_env = make_vec_env(env_producer, n_envs=cfg['parallel'], vec_env_cls=SubprocVecEnv)
            args['eval_env'] = eval_env

        callback_list.append(cb(**args))

    return callback_list



