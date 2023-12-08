import numpy as np


def reward_hit(env, info, done):
    reward = 0

    epsilon = 0.001

    if not env.has_hit:
        reward = - (np.linalg.norm(env.ee_pos[:2] - env.puck_pos[:2]) / (0.5 * env.specs.table_diag))
    '''else:
        reward =  1 / (task.distance(env, info, done) + epsilon)'''

    reward_constraint = env.specs.alpha_r * env._reward_constraints(info)
    reward += reward_constraint
    info['constr_reward'] = reward_constraint
    return reward, info
