import numpy as np

from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.utils import inverse_kinematics, world_to_robot, robot_to_world

EE_HEIGHT = 0.0645

class DummyAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, **kwargs):

        super().__init__(env_info, agent_id, **kwargs)

        self.restart = True
        self.last_cmd = None
        self.joint_trajectory = None
        self.cartesian_trajectory = None

        # Environment specs
        self.frame = self.env_info['robot']['base_frame'][0]
        self.table_length = self.env_info['table']['length']
        self.table_width = self.env_info['table']['width']
        self.goal_pos = np.array([self.table_length / 2, 0])
        self.mallet_radius = self.env_info['mallet']['radius']
        self.puck_radius = self.env_info['puck']['radius']

    def reset(self):
        self.restart = True
        self.last_cmd = None
        self.joint_trajectory = []
        self.cartesian_trajectory = []

    def draw_action(self, obs):
        if self.restart:
            self.restart = False
            self.q_cmd = self.get_joint_pos(obs)
            self.dq_cmd = self.get_joint_vel(obs)
            self.last_cmd = np.vstack([self.q_cmd, self.dq_cmd])

            self.cartesian_trajectory = self.plan_cartesian_trajectory(obs)

            self.joint_trajectory = self.plan_joint_trajectory(self.cartesian_trajectory)


        if len(self.joint_trajectory) > 0:
            joint_pos_des = self.joint_trajectory[0]

            self.joint_trajectory = self.joint_trajectory[1:]

            self.last_cmd[0] = joint_pos_des
            #self.last_cmd[1] = joint_vel_des
        else:
            self.last_cmd[1] = np.zeros(self.env_info['robot']['n_joints'])

        return self.last_cmd

    def plan_cartesian_trajectory(self, obs):
        x_init, _ = robot_to_world(self.frame, [0.65, 0.0])
        x_init = x_init[0]
        trajectory = np.array((
                                [x_init, 0.0, EE_HEIGHT],
                                [x_init + 0.001, 0.0, EE_HEIGHT]
                            ))

        steps = 200
        for _ in range(0, steps):
            x = trajectory[-1][0]
            y = trajectory[-1][1]
            value = np.array([x+0.001, y-0.001, 0.0])
            value[-1] = EE_HEIGHT

            trajectory = np.concatenate((trajectory, [value]))

        reverse_traj = trajectory[::-1]
        trajectory = np.concatenate((trajectory, reverse_traj))

        for i in range(0, len(trajectory)):
            trajectory[i], _ = world_to_robot(self.frame, trajectory[i])

        return trajectory

    def plan_joint_trajectory(self, trajectory):
        joint_pos_trajectory = np.zeros((len(trajectory), 7))

        for i in range(0, len(trajectory)):
            _, joint_pos_trajectory[i] = inverse_kinematics(self.robot_model, self.robot_data, trajectory[i])

        # TODO optimize trajectory velocity

        return joint_pos_trajectory
