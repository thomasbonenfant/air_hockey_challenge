from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from envs import make_environment
from stable_baselines3 import PPO

from my_scripts.utils import parse_args, create_log_directory
import os


def main():
    env_args, alg_args, learn_args, log_args, variant = parse_args()

    env_producer = lambda: make_environment(**env_args)
    env = VecMonitor(make_vec_env(env_producer, n_envs=variant.parallel, vec_env_cls=SubprocVecEnv))
    alg_args["env"] = env

    log_dir = create_log_directory(log_args, variant)
    learn_args["tb_log_name"] = log_dir

    eval_env = VecMonitor(make_vec_env(env_producer, n_envs=variant.parallel, vec_env_cls=SubprocVecEnv))
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
    eval_callback = EvalCallback(n_eval_episodes=variant.n_eval_episodes,
                                 eval_freq=variant.eval_freq,
                                 deterministic=True,
                                 log_path=log_dir,
                                 best_model_save_path=log_dir,
                                 eval_env=eval_env,
                                 callback_after_eval=stop_train_callback)

    # summary_writer_callback = SummaryWriterCallback()

    learn_args['callback'] = [eval_callback]  # , summary_writer_callback]

    model = PPO(**alg_args)
    model.learn(**learn_args)

    model.save(os.path.join(log_dir, "model.zip"))


'''
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()'''

if __name__ == '__main__':
    main()
