import os

from matplotlib import pyplot as plt
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.spaces import Box

from air_hockey_agent.agents.agent_SAC import AgentAirhockeySAC
from mushroom_rl.algorithms.actor_critic import SAC
import numpy as np

from air_hockey_challenge.utils import forward_kinematics


class doubleSACLearner(AgentAirhockeySAC):

    def __init__(self, env, **kwargs):
        # agent that starting from observation output desired x, y of end effector (z always 0)
        env['rl_info'].action_space = Box(low=np.array((0,-1)), high=np.array((1, 1)))
        params = self.configure_SAC(env['rl_info'], **kwargs)[:-1]
        alg_params = params[-1]
        self.traj_policy = SAC(*params[:-1], **alg_params)

        # agent that given x, y of end effector returns q and q' of the joints
        env['rl_info'].action_space = Box(low=-1, high=1, shape=(14,))
        env['rl_info'].observation_space = Box(low=-1, high=1, shape=(2,))
        super().__init__(env, **kwargs)

        self.env = env
        self.dataset_moving_agent = list()
        self.desired_ee_pos = np.zeros(3)
        self.trajectory = []
        self.mallet_radius = self.env['mallet']['radius']
        self.table_mult = np.array([-self.env['table']['length'] / 2 + self.mallet_radius,
                                    self.env['table']['width'] / 2 - self.mallet_radius])
        self.joint_mult = np.concatenate([env['robot']['joint_pos_limit'][1], env['robot']['joint_vel_limit'][1]])
        self.robot_model = env['robot']['robot_model']
        self.robot_data = env['robot']['robot_data']
        self.mover_dataset = []
        self.traj_dataset = []
        self.history_J_mover = []
        self.history_J_traj = []

    def plot_mover_J(self):
        J = compute_J(self.mover_dataset, 0.999)
        self.mover_dataset = []
        print('J: ', np.mean(J))
        self.history_J_mover.append(np.mean(J))
        self.mover_dataset = []
        plt.plot(self.history_J_mover)
        plt.savefig(os.path.join("air_hockey_agent/agents/log", "reward_mover"))

        J = compute_J(self.traj_dataset, 0.999)
        self.mover_dataset = []
        print('J: ', np.mean(J))
        self.history_J_traj.append(np.mean(J))
        self.traj_dataset = []
        plt.plot(self.history_J_traj)
        plt.savefig(os.path.join("air_hockey_agent/agents/log", "reward_traj"))


    def draw_action(self, observation):
        action = np.array([0.8, 0.]) / self.table_mult # self.traj_policy.draw_action(observation)
        self.desired_ee_pos[:2] = action * self.table_mult
        self.trajectory.append(self.desired_ee_pos)
        ctrl_action = super().draw_action(action) * self.joint_mult
        return ctrl_action.reshape((2, 7))

    def reward_traj(self, ee_pos):
        puck_pos = np.array([0.8, 0.0])
        # compute distance between puck and end effector
        rew = - np.linalg.norm(puck_pos - ee_pos)
        return rew

    def fit(self, dataset, **info):
        # dataset: [state, action, reward, next_state, absorbing, last]
        # state:obs, action: q,q', reward: r_traj, next_state: obs', absorbing, last
        dataset_traj = []
        dataset_mover = []
        for index, (state, action, reward, next_state, absorbing, last) in enumerate(dataset):
            # state
            s_mover = self.trajectory[index][:2]
            # action
            a_traj = self.trajectory[index][:2]
            a_mover = action.reshape((14))
            # reward
            # create reward for learner that applies "constrained inverse kinematics"
            r_traj = self.reward_traj(self.trajectory[index][:2])
            ee_pos_next = forward_kinematics(self.robot_model, self.robot_data, next_state[6:9])[0]
            r_mover = -np.linalg.norm(self.trajectory[index] - ee_pos_next)
            # next state
            ns_mover = ee_pos_next[:2]
            dataset_traj.append((state, a_traj, r_traj, next_state, absorbing, last))
            dataset_mover.append((s_mover, a_mover, r_mover, ns_mover, absorbing, last))
        self.trajectory = []

        self.mover_dataset.extend(dataset_mover)
        self.traj_dataset.extend(dataset_traj)
        super().fit(dataset_mover, **info)
        self.traj_policy.fit(dataset_traj, **info)

    def save(self, path, full_save=False):
        self.traj_policy.save("traj_policy" + path)
        super().save(path)

    def load(self, path):
        #self.traj_policy = self.traj_policy.load("traj_policy" + path)
        return super().load(path)

