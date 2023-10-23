from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from envs import make_environment, make_hit_env
from stable_baselines3 import PPO, SAC, DQN
from my_scripts.summary_writer import SummaryWriterCallback

from my_scripts.utils import parse_args, create_log_directory, variant_util
import os


def create_producer(env_args):
    env_name = env_args['env']
    if env_name == 'hrl':
        return lambda: make_environment(**env_args)
    if env_name == 'hit':
        return lambda: make_hit_env(**env_args)
    raise NotImplementedError


def main():
    env_args, alg_args, learn_args, log_args, variant = variant_util(parse_args())

    log_dir = create_log_directory(log_args, variant)
    learn_args["tb_log_name"] = log_dir

    env_producer = create_producer(env_args)
    env = make_vec_env(env_producer,
                       n_envs=variant.parallel,
                       vec_env_cls=SubprocVecEnv,
                       monitor_dir=log_dir)
    alg_args["env"] = env

    eval_env = make_vec_env(env_producer, n_envs=variant.parallel, vec_env_cls=SubprocVecEnv)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
    eval_callback = EvalCallback(n_eval_episodes=variant.n_eval_episodes,
                                 eval_freq=variant.eval_freq,
                                 deterministic=True,
                                 log_path=log_dir,
                                 best_model_save_path=log_dir,
                                 eval_env=eval_env)

    summary_writer_callback = SummaryWriterCallback()

    learn_args['callback'] = [eval_callback, summary_writer_callback]

    if log_args['alg'] == 'ppo':
        model = PPO(**alg_args)
    elif log_args['alg'] == 'sac':
        model = SAC(**alg_args)
    elif log_args['alg'] == 'dqn':
        model = DQN(**alg_args)
    else:
        raise NotImplementedError
    model.learn(**learn_args)

    model.save(os.path.join(log_dir, "model.zip"))

    if log_args['alg'] == 'dqn' or log_args['alg'] == 'sac':
        model.save_replay_buffer(os.path.join(log_dir))


if __name__ == '__main__':
    main()
