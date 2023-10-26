from envs.env_maker import make_environment, make_hit_env
import numpy as np

env = make_hit_env(include_ee=False,
                   include_ee_vel=True,
                   include_joints=False,
                   include_puck=False,
                   remove_last_joint=False,
                   scale_obs=False,
                   alpha_r=10.0,
                   max_path_len=400,
                   scale_action=True,
                   hit_coeff=1000)

np.random.seed(668)
for i in range(10):
    s, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        s, r, done, truncated, info = env.step(action)

        env.render()
