import torch

from envs.env_maker import *
import numpy as np
from pprint import pprint
from gymnasium.spaces import unflatten

env = make_hit_env(include_ee=True,
                   include_ee_vel=True,
                   include_joints=True,
                   include_puck=True,
                   remove_last_joint=False,
                   scale_obs=True,
                   alpha_r=10.0,
                   max_path_len=50,
                   scale_action=False,
                   hit_coeff=1000,
                   joint_acc_clip=1.0)
'''

env = make_hrl_environment(['hit_rb', 'repel_oac', 'home_sb3', 'prepare_rb'],
                           steps_per_action=15,
                           alpha_r=1.0,
                           include_ee=True,
                           render=True,
                           include_joints=False,
                           include_timer=True,
                           )
env = make_goal_env(include_joints=True,
                    include_ee=True,
                    include_ee_vel=True,
                    include_puck=False,
                    alpha_r=0.0,
                    scale_action=False,
                    remove_last_joint=False,
                    max_path_len=100,
                    scale_obs=True,
                    goal_horizon=30)'''

'''env = make_airhockey_oac('7dof-hit', high_level_action=False,use_atacom=True)'''

seed = 666
env.reset(seed=seed)
env.action_space.seed(seed)

for i in range(10):
    s, _ = env.reset()

    done = False

    while not done:
        action = env.action_space.sample()
        s, r, done, term, info = env.step(action)


        #goal_pos = env.goal['g_ee_pos'].copy()
        #goal_pos[0] -= 1.51

        #env.env.env.base_env._data.site("goal_vis").xpos = np.hstack([goal_pos, 0])

        env.render()

