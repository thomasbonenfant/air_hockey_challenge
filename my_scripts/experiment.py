from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from envs import create_producer
from stable_baselines3 import PPO, SAC, DQN

from my_scripts.utils import parse_args, create_log_directory, variant_util
from my_scripts.callbacks.info_logs_callback import EpisodicInfoCallback
import os

def main():
    env_args, alg_args, learn_args, log_args, variant = variant_util(parse_args())

    log_dir = create_log_directory(log_args, variant)
    learn_args["tb_log_name"] = log_dir

    monitor_kwargs = {
        'info_keywords': ('constr_reward',)
    }

    env_producer = create_producer(env_args)
    env = make_vec_env(env_producer,
                       n_envs=variant.parallel,
                       vec_env_cls=SubprocVecEnv,
                       monitor_dir=log_dir,
                       monitor_kwargs=monitor_kwargs
                       )
    alg_args["env"] = env

    eval_env = make_vec_env(env_producer, n_envs=variant.parallel, vec_env_cls=SubprocVecEnv)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
    eval_callback = EvalCallback(n_eval_episodes=variant.n_eval_episodes,
                                 eval_freq=variant.eval_freq,
                                 deterministic=True,
                                 log_path=log_dir,
                                 best_model_save_path=log_dir,
                                 eval_env=eval_env)

    #summary_writer_callback = RewardLogsCallback()

    learn_args['callback'] = [eval_callback] #, summary_writer_callback]

    print(alg_args)

    # try to load an existing experiment:
    if log_args['alg'] == 'ppo':
        alg_cls = PPO
    elif log_args['alg'] == 'sac':
        alg_cls = SAC
    elif log_args['alg'] == 'dqn':
        alg_cls = DQN
    else:
        raise NotImplementedError

    try:
        model_path = os.path.join(log_dir, 'model')
        model = alg_cls.load(model_path)
        print('Retrieving previous experiment')
    except Exception:
        print('Launching new experiment')
        model = alg_cls(**alg_args)

    model.learn(**learn_args)

    model.save(os.path.join(log_dir, "model.zip"))

    if log_args['alg'] == 'dqn' or log_args['alg'] == 'sac':
        model.save_replay_buffer(os.path.join(log_dir, "replay_buffer"))


if __name__ == '__main__':
    main()
