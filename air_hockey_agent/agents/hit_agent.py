import numpy as np
from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.core import Core
from mushroom_rl.environments import AirHockeyHit
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.policy import ClippedGaussianPolicy
from mushroom_rl.utils.dataset import compute_J

from air_hockey_challenge.utils.kinematics import inverse_kinematics
from air_hockey_challenge.utils.kinematics import forward_kinematics
from air_hockey_agent.actor_critic_network import ActorNetwork, CriticNetwork

from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_challenge.framework.agent_base import AgentBase
import torch.optim as optim
import torch.nn.functional as F
import torch

spec = []


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    return HittingAgent(env_info, **kwargs)


class HittingAgent(AgentBase, DDPG):
    def __init__(self, env_info, mdp_info, policy_class, policy_params, actor_params, actor_optimizer,
                 critic_params, batch_size, initial_replay_size, max_replay_size, tau, **kwargs):
        DDPG.__init__(self, mdp_info, policy_class, policy_params, actor_params, actor_optimizer, critic_params,
                      batch_size, initial_replay_size, max_replay_size, tau)
        kwargs['no_initialization'] = True
        AgentBase.__init__(self, env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

        self.path = kwargs['path'] if 'path' in kwargs else [[-0.8, 0, 0]]

        self.steps_per_action = kwargs['steps_per_action'] if 'steps_per_action' in kwargs else 50

        self.x_shifter_lb = self.env_info["constraints"].get("ee_constr").x_lb
        self.x_shifter_ub = -self.env_info['robot']['base_frame'][0][0, 3]
        self.y_shifter_lb = self.env_info["constraints"].get("ee_constr").y_lb
        self.y_shifter_ub = self.env_info["constraints"].get("ee_constr").y_ub
        self.x_mult = (self.x_shifter_ub - self.x_shifter_lb)
        self.y_mult = (self.y_shifter_ub - self.y_shifter_lb)

        self.step = 0
        self.path_idx = 0
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        """
        Draw an action from the agent's policy.
        :param observation: The current observation of the environment.
        """
        if self.new_start:
            self.new_start = False
            #print(self.env_info['robot']['base_frame'][0][0, 3])

        q = observation[self.env_info['joint_pos_ids']]
        dq = observation[self.env_info['joint_vel_ids']]
        #print(q, dq)
        #print(self.env_info["constraints"].get("ee_constr").fun(q, dq))
        DDPG_action = self.sigmoid(DDPG.draw_action(self, observation))
        #print(f"DDPG action: {DDPG_action}")
        pos = DDPG_action * [self.x_mult, self.y_mult, 1] + [self.x_shifter_lb, self.y_shifter_lb, -0.5]

        #print(pos)
        action = inverse_kinematics(self.env_info['robot']['robot_model'], self.env_info['robot']['robot_data'], pos)[1]

        return np.array([action, np.ones(3) * 0.05])


def main():
    test = False

    np.random.seed(0)
    torch.manual_seed(0)
    env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity",
                                    interpolation_order=3, debug=False, custom_reward_function=reward)

    # MDP
    gamma_eval = 0.99

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2)

    # Settings
    initial_replay_size = 10000
    max_replay_size = 10000
    batch_size = 200
    n_features = 80
    tau = .001

    # Approximator
    actor_input_shape = env.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        n_features=n_features,
                        input_shape=actor_input_shape,
                        output_shape=(3,))

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 1e-5}}

    critic_input_shape = (actor_input_shape[0] + env.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 1e-3}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,))


    agent = HittingAgent(env.env_info, env.info, policy_class, policy_params, actor_params, actor_optimizer, critic_params,
            batch_size, initial_replay_size, max_replay_size, tau)
    #agent = DDPG(env.info, policy_class, policy_params, actor_params, actor_optimizer, critic_params,
    #        batch_size, initial_replay_size, max_replay_size, tau)

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
    n_steps = 1000
    n_steps_test = 2000

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
        core.learn(n_steps=n_steps, n_steps_per_fit=10)

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
