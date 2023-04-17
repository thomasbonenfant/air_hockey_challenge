import numpy as np
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J
from air_hockey_challenge.utils.kinematics import forward_kinematics
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
import torch
from air_hockey_agent.agents.hit_agent import HittingAgent
from air_hockey_agent.agents.ATACOM_hit_agent import AtacomHittingAgent


def main():
    test = False

    np.random.seed(0)
    torch.manual_seed(0)
    env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity",
                                    interpolation_order=3, debug=False, custom_reward_function=reward)

    # MDP
    gamma_eval = 0.99

    # Settings
    # number of initial iterations to fill the replay memory
    initial_replay_size = 100

    #print(env.env_info)

    agent = AtacomHittingAgent(env)
    #agent = AirHockeyPlanarAtacom(env.env_info, env.info, policy_class, policy_params, actor_params, actor_optimizer, critic_params,
    #        batch_size, initial_replay_size, max_replay_size, tau, task='H', gamma=gamma_eval)

    if test:
        agent.load("hit_agent.msh")

    obs = env.reset()
    agent.episode_start()
    # Algorithm
    core = Core(agent, env)

    # Fill the replay memory with random samples
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    n_epochs = 100
    n_steps = 100
    n_steps_test = 200

    if test:
        dataset = core.evaluate(n_steps=n_steps_test, render=True)
        J = compute_J(dataset, gamma_eval)
        print('Epoch: 0')
        print('J: ', np.mean(J))
        return None

    dataset = core.evaluate(n_steps=n_steps_test)
    J = compute_J(dataset, gamma_eval)
    print('Epoch: 0')
    print('J: ', np.mean(J))

    for n in range(n_epochs):
        print('\nEpoch: ', n + 1)
        core.learn(n_steps=n_steps, n_steps_per_fit=10, render=True)

    dataset = core.evaluate(n_steps=n_steps_test, render=True)
    J = compute_J(dataset, gamma_eval)
    print('J: ', np.mean(J))
    agent.save("hit_agent.msh")


def reward(base_env, state, action, next_state, absorbing):
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
        puck_prev_vel = base_env.get_puck(state)[1]
        was_hit = np.sum(puck_prev_vel) == 0 and np.sum(puck_vel) != 0
        if not was_hit:
            # compute end effector position
            ee_pos = forward_kinematics(base_env.env_info['robot']['robot_model'], base_env.env_info['robot']['robot_data'], action[0])[0][:2]

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


if __name__ == '__main__':
    main()
