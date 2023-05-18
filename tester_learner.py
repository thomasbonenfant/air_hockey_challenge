import os

import matplotlib.pyplot as plt
import numpy as np
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from air_hockey_challenge.utils import robot_to_world
from air_hockey_challenge.utils.kinematics import forward_kinematics
from air_hockey_challenge.framework import AirHockeyChallengeWrapper, ChallengeCore
import torch
from air_hockey_agent.agents.hit_agent_SAC import HittingAgent
from air_hockey_agent.agents.ATACOM_hit_agent import AtacomHittingAgent
from air_hockey_agent.agents.hit_agent_SAC_for_ATACOM_env import HittingAgentSAC4ATACOM
from air_hockey_agent.agents.ATACOM_challenge import ATACOMChallengeWrapper

def main():
    test = False

    np.random.seed(0)
    torch.manual_seed(0)
    #env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity",
    #                                interpolation_order=3, debug=False, custom_reward_function=reward)
    env = ATACOMChallengeWrapper(env="3dof-hit", action_type="position-velocity",
                                    interpolation_order=3, debug=False, custom_reward_function=reward)

    # MDP
    gamma_eval = 0.999

    # Settings
    # number of initial iterations to fill the replay memory
    initial_replay_size = 20000

    agent = HittingAgentSAC4ATACOM(env.env_info)
    #agent = AtacomHittingAgent(env.env_info)
    #agent = AirHockeyPlanarAtacom(env.env_info, env.info, policy_class, policy_params, actor_params, actor_optimizer, critic_params,
    #        batch_size, initial_replay_size, max_replay_size, tau, task='H', gamma=gamma_eval)

    if test:
        agent.load("hit_agent.msh")

    #agent.load("hit_agent.msh")
    obs = env.reset()
    agent.episode_start()
    # Algorithm
    core = ChallengeCore(agent, env)

    # RUN
    n_epochs = 100
    n_steps = 5000
    n_steps_test = 5000
    history_J = []
    plt.plot(history_J)
    if test:
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = compute_J(dataset, gamma_eval)
        print('evaluatiom:')
        print('J: ', np.mean(J))
        return None

    # Fill the replay memory with random samples
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    dataset = core.evaluate(n_steps=500, render=False)
    J = compute_J(dataset, gamma_eval)
    history_J.append(np.mean(J))
    print('Epoch: 0')
    print('J: ', np.mean(J))
    plot_constraints(env.env_info, dataset, "air_hockey_agent/agents/log")
    # this created the  directory log if not already existing

    for n in range(n_epochs):
        print('\nEpoch: ', n + 1)
        dataset = core.learn(n_steps=n_steps, n_steps_per_fit=10)
        dataset = core.evaluate(n_steps=500, render=False)
        J = compute_J(dataset, gamma_eval)
        print('J: ', np.mean(J))
        history_J.append(np.mean(J))
        plt.plot(history_J)
        plt.savefig(os.path.join("air_hockey_agent/agents/log", "history_J"))

    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = compute_J(dataset, gamma_eval)
    print('J: ', np.mean(J))
    plot_constraints(env.env_info, dataset, "air_hockey_agent/agents/log")
    history_J.append(np.mean(J))
    plt.plot(history_J)
    plt.show()
    agent.save("hit_agent.msh")


def reward(base_env, state, action, next_state, absorbing):
    rew = 0
    if state[-1] == 0:
        puck_pos, puck_vel = base_env.get_puck(next_state)  # get puck position and velocity
        puck_pos = puck_pos[:2]
        ee_pos = forward_kinematics(base_env.env_info['robot']['robot_model'], base_env.env_info['robot']['robot_data'], action[0])[0][:2]
        # compute distance between puck and end effector
        rew = - np.linalg.norm(puck_pos[:2] - ee_pos)
        
    elif absorbing:

        puck_pos, puck_vel = base_env.get_puck(next_state)  # get puck position and velocity
        puck_pos = puck_pos[:2]

        if (puck_pos[0] - base_env.env_info['table']['length'] / 2) > 0 and \
                (np.abs(puck_pos[1]) - base_env.env_info['table']['goal_width']) < 0:
            rew = 200
    #print(rew)
    return rew


def reward2(base_env, state, action, next_state, absorbing):
    r = 0
    puck_pos, puck_vel = base_env.get_puck(next_state) # get puck position and velocity
    puck_pos = puck_pos[:2]

    # If puck is out of bounds
    if absorbing:
        # If puck is in the opponent goal
        if (puck_pos[0] - base_env.env_info['table']['length'] / 2) > 0 and \
                (np.abs(puck_pos[1]) - base_env.env_info['table']['goal_width']) < 0:
            r = 200

    else:
        # compute if puck was hit by the robot (based on current and next velocity)
        was_hit = state[-1] == 0
        if not was_hit:
            # compute end effector position
            ee_pos = forward_kinematics(base_env.env_info['robot']['robot_model'], base_env.env_info['robot']['robot_data'], action[0])[0][:2]
            #ee_pos = robot_to_world(base_env.env_info["robot"]["base_frame"][0], translation=ee_pos)[0]
            # compute distance between puck and end effector
            dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos)

            vec_ee_puck = (puck_pos[:2] - ee_pos) / dist_ee_puck

            # width of table minus radius of puck
            effective_width = 0.51 - 0.03165

            goal = np.array([0.98, 0])

            # Calculate bounce point by assuming incoming angle = outgoing angle
            w = (abs(puck_pos[1]) * goal[0] + goal[1] * puck_pos[0] - effective_width * puck_pos[
                0] - effective_width * goal[0]) / (abs(puck_pos[1]) + goal[1] - 2 * effective_width)

            side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])
            vec_puck_side = (side_point - puck_pos) / np.linalg.norm(side_point - puck_pos)
            cos_ang_side = np.clip(vec_puck_side @ vec_ee_puck, 0, 1)

            # Reward if vec_ee_puck and vec_puck_goal have the same direction
            vec_puck_goal = (goal - puck_pos) / np.linalg.norm(goal - puck_pos)
            cos_ang_goal = np.clip(vec_puck_goal @ vec_ee_puck, 0, 1)
            cos_ang = np.max([cos_ang_goal, cos_ang_side])

            r = np.exp(-8 * (dist_ee_puck - 0.08)) * cos_ang ** 2
        else:
            r_hit = 0.25 + min([1, (0.25 * puck_vel[0] ** 4)])

            r_goal = 0
            if puck_pos[0] > 0.7:
                sig = 0.1
                r_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)

            r = 2 * r_hit + 10 * r_goal

    r -= 1e-3 * np.linalg.norm(action)
    return r

def plot_constraints(env, dataset, save_dir="", suffix="", state_norm_processor=None):
    state_list = list()
    i = 0

    if suffix != '':
        suffix = suffix + "_"

    for data in dataset:
        state = data[0]
        if state_norm_processor is not None:
            state[state_norm_processor._obs_mask] = (state * state_norm_processor._obs_delta + \
                                                     state_norm_processor._obs_mean)[state_norm_processor._obs_mask]
        state_list.append(state)
        if data[-1]:
            i += 1
            state_hist = np.array(state_list)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            ee_pos_list = list()
            for state_i in state_hist:

                ee_pos = forward_kinematics(env['robot']['robot_model'], env['robot']['robot_data'], state_i[6:9])[0][:2]
                ee_pos = robot_to_world(env["robot"]["base_frame"][0], translation = ee_pos)[0]
                ee_pos_list.append(ee_pos)

            ee_pos_list = np.array(ee_pos_list)
            fig1, axes1 = plt.subplots(1, figsize=(10, 10))
            axes1.plot(ee_pos_list[:, 0], ee_pos_list[:, 1], label='position')
            axes1.plot([0.0, -0.91, -0.91, 0.0], [-0.45, -0.45, 0.45, 0.45], label='boundary', c='k', lw='5')
            axes1.set_aspect(1.0)
            axes1.set_xlim(-1, 0)
            axes1.set_ylim(-0.5, 0.5)
            axes1.legend(loc='upper right')
            axes1.set_title('EndEffector')
            axes1.legend(loc='center right')
            file1 = "EndEffector_" + suffix + str(i) + ".pdf"
            plt.savefig(os.path.join(save_dir, file1))
            plt.close(fig1)

            fig2, axes2 = plt.subplots(1, 3, sharey=True, figsize=(21, 8))
            axes2[0].plot(state_hist[:, 6], label='position', c='tab:blue')
            axes2[1].plot(state_hist[:, 7], c='tab:blue')
            axes2[2].plot(state_hist[:, 8], c='tab:blue')
            constraint = env['robot']['joint_pos_limit']
            axes2[0].plot([0, state_hist.shape[0]], [constraint[0][0]] * 2,
                          label='position limit', c='tab:red', ls='--')
            axes2[1].plot([0, state_hist.shape[0]], [constraint[0][1]] * 2, c='tab:red',
                          ls='--')
            axes2[2].plot([0, state_hist.shape[0]], [constraint[0][2]] * 2, c='tab:red',
                          ls='--')
            axes2[0].plot([0, state_hist.shape[0]], [constraint[1][0]] * 2, c='tab:red',
                          ls='--')
            axes2[1].plot([0, state_hist.shape[0]], [constraint[1][1]] * 2, c='tab:red',
                          ls='--')
            axes2[2].plot([0, state_hist.shape[0]], [constraint[1][2]] * 2, c='tab:red',
                          ls='--')

            axes2[0].plot(state_hist[:, 9], label='velocity', c='tab:orange')
            axes2[1].plot(state_hist[:, 10], c='tab:orange')
            axes2[2].plot(state_hist[:, 11], c='tab:orange')
            constraint = env['robot']['joint_vel_limit']
            axes2[0].plot([0, state_hist.shape[0]], [constraint[0][0]] * 2,
                          label='velocity limit', c='tab:pink', ls=':')
            axes2[1].plot([0, state_hist.shape[0]], [constraint[0][1]] * 2, c='tab:pink', ls=':')
            axes2[2].plot([0, state_hist.shape[0]], [constraint[0][2]] * 2, c='tab:pink', ls=':')
            axes2[0].plot([0, state_hist.shape[0]], [constraint[1][0]] * 2, c='tab:pink', ls=':')
            axes2[1].plot([0, state_hist.shape[0]], [constraint[1][1]] * 2, c='tab:pink', ls=':')
            axes2[2].plot([0, state_hist.shape[0]], [constraint[1][2]] * 2, c='tab:pink', ls=':')

            axes2[0].set_title('Joint 1')
            axes2[1].set_title('Joint 2')
            axes2[2].set_title('Joint 3')
            fig2.legend(ncol=4, loc='lower center')

            file2 = "JointProfile_" + suffix + str(i) + ".pdf"
            plt.savefig(os.path.join(save_dir, file2))
            plt.close(fig2)

            state_list = list()


if __name__ == '__main__':
    main()
