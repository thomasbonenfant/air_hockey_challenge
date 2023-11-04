import torch

from envs.env_maker import make_hrl_environment, make_hit_env
import numpy as np

env = make_hit_env(include_ee=False,
                   include_ee_vel=True,
                   include_joints=False,
                   include_puck=False,
                   remove_last_joint=False,
                   scale_obs=False,
                   alpha_r=10.0,
                   max_path_len=50,
                   scale_action=True,
                   hit_coeff=1000)

from envs.air_hockey import AirHockeyEnv
from utils.env_utils import NormalizedBoxEnv
#env = NormalizedBoxEnv(AirHockeyEnv('tournament', interpolation_order=3, use_atacom=True, use_aqp=False, high_level_action=False))

seed = 666
env.reset(seed=seed)
env.action_space.seed(seed)

for i in range(10):
    s, _ = env.reset()

    done = False

    while not done:
        action = env.action_space.sample()
        s, r, done, term, info = env.step(action)

        for k in info:
            # if k.startswith('avg') or k.startswith('max'):
            if k.endswith('joint_vel_constr'):
                print(f'{k}: {info[k]}')

        env.render()

