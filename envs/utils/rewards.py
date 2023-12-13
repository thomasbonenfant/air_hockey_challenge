import numpy as np
from air_hockey_challenge.utils.transformations import robot_to_world


def linear(distance, max_distance):
    return 1000 - 1000 * distance / max_distance


def hyperbolic(delta, max_delta):
    epsilon = 0.001
    #return 1 / (delta/max_delta + epsilon)
    return 1 / (delta / 100 + epsilon) - (100 * delta / max_delta) / (np.pi + epsilon)


def reward_hit(env, info, done):
    reward = 0

    if not env.has_hit:
        reward = - (np.linalg.norm(env.ee_pos[:2] - env.puck_pos[:2]) / (0.5 * env.specs.table_diag))
    else:
        if not env.hit_rew_given:
            delta = env.task.distance(env, info, done)
            reward = hyperbolic(delta, np.pi)
            reward *= np.linalg.norm(env.puck_vel[:2]) / env.specs.max_vel
            env.hit_rew_given = True

    reward_constraint = env.specs.alpha_r * env._reward_constraints(info)
    reward += reward_constraint
    info['constr_reward'] = reward_constraint
    return reward, info


def reward_defend(env, info, done):
    reward = 0

    w_puck_pos, _ = robot_to_world(env.specs.robot_frame[0], env.puck_pos)

    # if hits the border or makes goal
    if w_puck_pos[0] < -env.specs.table_length / 2 + env.specs.mallet_radius:
        reward = -1000

    if env.has_hit:
        if not env.hit_rew_given:
            delta = env.task.distance(env, info, done)
            reward = linear(delta, np.pi) / (np.linalg.norm(env.puck_vel[:2]) / env.specs.max_vel + 0.01)
            env.hit_rew_given = True

    reward_constraint = env.specs.alpha_r * env._reward_constraints(info)
    reward += reward_constraint
    info['constr_reward'] = reward_constraint
    return reward, info


def reward_prepare(env, info, done):
    reward = 0

    if not env.has_hit:
        reward = - (np.linalg.norm(env.ee_pos[:2] - env.puck_pos[:2]) / (0.5 * env.specs.table_diag))
    else:
        if not env.hit_rew_given:
            reward += 10
            env.hit_rew_given = True

        if done:
            reward = 100 if env.task.distance(env, info, done) < 0.2 else -100


    reward_constraint = env.specs.alpha_r * env._reward_constraints(info)
    reward += reward_constraint
    info['constr_reward'] = reward_constraint
    return reward, info





