import numpy as np
from air_hockey_challenge.utils.transformations import robot_to_world
from envs.utils.vectorize_reward import vectorize


def linear(distance, max_distance):
    return 1000 - 1000 * distance / max_distance


def hyperbolic(delta, max_delta):
    epsilon = 0.001
    # return 1 / (delta/max_delta + epsilon)
    return 1 / (delta / 100 + epsilon) - (100 * delta / max_delta) / (np.pi + epsilon)


@vectorize
def goal_reward_hit(env, achieved_goal, desired_goal, info):
    achieved_goal = env.task.unflatten(achieved_goal)
    desired_goal = env.task.unflatten(desired_goal)

    puck_vel = info['puck_vel']

    if desired_goal['has_hit'] and not achieved_goal['has_hit']:
        reward = - info['puck_distance'] / (0.5 * env.specs.table_diag)
        task_error = 2 * np.pi
    else:
        reward = 100
        delta = np.abs(desired_goal['puck_dir'] - achieved_goal['puck_dir'])

        if delta > np.pi:
            delta = 2 * np.pi - delta

        if delta <= np.pi / 8:  # 15 degrees tolerance
            reward += 500
            reward += 500 * np.linalg.norm(puck_vel[:2]) / env.specs.max_vel

        task_error = np.squeeze(delta)

    reward_constraint = env.specs.alpha_r * env._reward_constraints(info)
    reward += reward_constraint
    info['constr_reward'] = reward_constraint
    info['task_distance'] = task_error

    return reward, info


@vectorize
def goal_reward_prepare(env, achieved_goal, desired_goal, info):
    distance = np.linalg.norm(desired_goal - achieved_goal)
    info['task_distance'] = distance
    puck_vel = info['puck_vel'][:2]

    rew = - info['puck_distance'] / (0.5 * env.specs.table_diag)

    if distance < 0.0633:
        rew += 1000 - 100* np.linalg.norm(puck_vel)

    return rew, info

def reward_hit(env, info, done):
    reward = 0

    if not env.has_hit:
        reward = - (np.linalg.norm(env.ee_pos[:2] - env.puck_pos[:2]) / (0.5 * env.specs.table_diag))
    else:
        if not env.hit_rew_given:
            delta = env.task.distance(env, info, done)
            reward = linear(delta, np.pi)
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


def reward_defend2(env, info, done):
    reward = 0

    w_puck_pos, _ = robot_to_world(env.specs.robot_frame[0], env.puck_pos)

    # if hits the border or makes goal
    if w_puck_pos[0] < -env.specs.table_length / 2 + env.specs.mallet_radius:
        reward = -1000

    if env.puck_vel[0] > 0 and w_puck_pos[0] >= 0:
        reward = -100

    reward += -np.linalg.norm(env.puck_vel[:2])

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
