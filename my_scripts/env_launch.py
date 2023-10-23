from envs.env_maker import make_environment, make_hit_env
import numpy as np

env = make_hit_env(include_ee=True,
                   include_ee_vel=True,
                   include_joints=True,
                   scale_obs=True,
                   alpha_r=100.0)

np.random.seed(668)
for i in range(10):
    s, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        s, r, done, truncated, info = env.step(action)
        print(r)
        env.render()
