import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env

from envs import make_environment
from envs.fixed_options_air_hockey import HierarchicalEnv

from stable_baselines3 import A2C, PPO

if __name__ == '__main__':
    env = VecMonitor(make_vec_env(make_environment))


    algo_params = {
        'policy': "MlpPolicy",
        'env': env,
        'n_steps': 5,
        'n_epochs': 1,
        'batch_size': 5,
        'gamma': 0.997,
        'seed': 666,
        'verbose': 1,
        'tensorboard_log': '../tb_logs/',
    }

    model = PPO(**algo_params)
    model.learn(total_timesteps=100, tb_log_name="airhockey_1")

    model.save("training/")

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()