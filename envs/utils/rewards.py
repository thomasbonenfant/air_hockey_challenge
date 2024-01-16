import numpy as np
from air_hockey_challenge.utils.transformations import robot_to_world
from envs.utils.vectorize_reward import vectorize


def linear(distance, max_distance):
    return 1000 - 1000 * distance / max_distance


def hyperbolic(delta, max_delta):
    if np.abs(delta) > np.pi / 2:
        return np.zeros((1,))

    epsilon = 0.001
    # return 1 / (delta/max_delta + epsilon)
    return 1 / (delta / 100 + epsilon) - (delta / max_delta) / (max_delta / 100 + epsilon)


@vectorize
def goal_reward_hit(env, achieved_goal, desired_goal, info):
    achieved_goal = env.task.unflatten(achieved_goal)
    desired_goal = env.task.unflatten(desired_goal)

    puck_vel = info['puck_vel']

    if not info['has_hit']:
        reward = - info['puck_distance'] / (0.5 * env.specs.table_diag)
    else:
        delta = np.abs(desired_goal['puck_dir'] - achieved_goal['puck_dir'])

        if delta > np.pi:
            delta = 2 * np.pi - delta

        reward = hyperbolic(delta, np.pi / 2)
        reward += 1000 * np.linalg.norm(puck_vel[:2]) / env.specs.max_vel

        task_error = np.squeeze(delta)
        info['task_distance'] = task_error

    reward_constraint = env.specs.alpha_r * env._reward_constraints(info)
    reward += reward_constraint
    info['constr_reward'] = reward_constraint

    return np.array(reward).squeeze(), info


@vectorize
def goal_reward_repel(env, achieved_goal, desired_goal, info):

    # check goal or our border touched
    puck_pos_x = info['puck_pos'][0]
    puck_vel_x = info['puck_vel'][0]

    border = 1.51 - env.specs.table_length / 2

    if puck_pos_x <= border or (not info['has_hit'] and puck_vel_x > 0):
        return -1000, info

    return goal_reward_hit.__wrapped__(env, achieved_goal, desired_goal, info)


@vectorize
def goal_reward_prepare(env, achieved_goal, desired_goal, info):
    distance = np.linalg.norm(desired_goal - achieved_goal)
    info['task_distance'] = distance
    info['is_success'] = False
    puck_vel = info['puck_vel'][:2]

    rew = 0

    if not info['has_hit']:
        rew += - info['puck_distance'] / (0.5 * env.specs.table_diag)
    else:
        threshold = 0.1
        if env.specs.scale_obs:
            threshold /= 0.5 * env.specs.table_diag

        if distance < threshold:
            rew += 1000 - 100 * np.linalg.norm(puck_vel)
            info['is_success'] = True

    return rew, info


def reward_hit(env, info, done):
    reward = 0

    if not env.has_hit:
        reward = - (np.linalg.norm(env.ee_pos[:2] - env.puck_pos[:2]) / (0.5 * env.specs.table_diag))
    else:
        if not env.hit_rew_given:
            delta = env.task.distance(env, info, done)
            reward = hyperbolic(delta, np.pi / 2)
            puck_vel_norm = np.linalg.norm(env.puck_vel[:2])
            reward += 500 * puck_vel_norm / env.specs.max_vel
            env.hit_rew_given = True

            info['task_distance'] = delta

    reward_constraint = env.specs.alpha_r * env._reward_constraints(info)
    reward += reward_constraint
    info['constr_reward'] = reward_constraint
    return np.squeeze(reward), info


def reward_defend(env, info, done):
    reward = 0

    w_puck_pos, _ = robot_to_world(env.specs.robot_frame[0], env.puck_pos)

    # if hits the border or makes goal
    if w_puck_pos[0] < -env.specs.table_length / 2 + env.specs.puck_radius:
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
