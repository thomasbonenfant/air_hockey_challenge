from envs.utils import Specification
import numpy as np
from air_hockey_challenge.utils.kinematics import jacobian


# TODO: remove dependency on env maybe using a State object

def process_state(env, specs: Specification, state, info):
    obs = {}

    if specs.include_puck:
        obs['puck_pos'] = state[specs.f_puck_pos_ids]
        obs['puck_vel'] = state[specs.f_puck_vel_ids]

    if specs.include_joints:
        obs['joint_pos'] = state[specs.f_joint_pos_ids]
        obs['joint_vel'] = state[specs.f_joint_vel_ids]

    if specs.include_opponent:
        obs['opponent_ee'] = state[specs.f_opponent_ee_ids]

    if specs.include_ee:
        obs['ee_pos'] = env.ee_pos[:2]  # do not include z coordinate

    if specs.include_ee_vel:
        obs['ee_vel'] = env.ee_vel[:2]

    if specs.include_hit_flag:
        obs['has_hit'] = env.has_hit

    if specs.scale_obs:
        obs = _scale_obs(specs, obs)

    return obs


def _scale(x, x_min, x_range):
    return (x - x_min) / x_range * 2 - 1


def _scale_obs(specs, obs):
    for k in specs.min_dict:
        obs[k] = _scale(obs[k], specs.min_dict[k], specs.range_dict[k])
    return obs


def post_simulation(env, obs):
    env._obs = obs
    env.puck_pos = env.get_puck_pos(obs)
    env.previous_vel = env.puck_vel if env.t > 0 else None
    env.puck_vel = env.get_puck_vel(obs)
    env.joint_pos = env.get_joint_pos(obs)
    env.joint_vel = env.get_joint_vel(obs)
    env.previous_ee_pos = env.ee_pos if env.t > 0 else None
    env.ee_pos = env.get_ee_pose(obs)

    env.ee_vel = _apply_forward_velocity_kinematics(env.specs, env.joint_pos, env.joint_vel)
    if env.previous_vel is not None:
        previous_vel_norm = np.linalg.norm(env.previous_vel[:2])
        current_vel_norm = np.linalg.norm(env.puck_vel[:2])
        distance = np.linalg.norm(env.puck_pos[:2] - env.ee_pos[:2])
        if previous_vel_norm <= current_vel_norm and distance <= (
                env.specs.puck_radius + env.specs.mallet_radius) * 1.1:
            env.has_hit = True


def _apply_forward_velocity_kinematics(specs, joint_pos, joint_vel):
    robot_model = specs.robot_model
    robot_data = specs.robot_data
    jac = jacobian(robot_model, robot_data, joint_pos)
    jac = jac[:3]  # last part of the matrix is about rotation. no need for it
    ee_vel = jac @ joint_vel
    return ee_vel


def process_info(env, specs, info):
    if 'constraints_value' in info:
        info_constr = info['constraints_value']
        del info['constraints_value']
        for k, v in info_constr.items():
            info[k] = v

    if specs.remove_last_joint:
        idx_to_delete = [6, 13]
    else:
        idx_to_delete = []
    for constr in ['joint_pos_constr', 'joint_vel_constr']:
        info[constr] = np.delete(info[constr], idx_to_delete, axis=0)

    info['puck_vel'] = env.puck_vel[:2]
    info['puck_distance'] = np.linalg.norm(env.puck_pos[:2] - env.ee_pos[:2])
    info['has_hit'] = env.has_hit
    info['puck_pos'] = env.puck_pos[:2]

    del info['jerk']

    return info
