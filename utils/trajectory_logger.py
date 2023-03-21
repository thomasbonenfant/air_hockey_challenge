import matplotlib.pyplot as plt
import numpy as np

class TrajectoryLogger():

    def __init__(self):
        self.ee_trajectory = []
        self.puck_trajectory = []
        self.action_position = []

    def append2trajectory(self, ee_position, puck_position, action):
        '''
        Saves new position data
        args:
            -   ee_position: np.ndarray of size (2,)
            -   puck_position: np.ndarray of size(2,)
            -   action: np.ndarray of size(2,)  
        '''
        self.ee_trajectory.append(ee_position)
        self.puck_trajectory.append(puck_position)
        self.action_position.append(action)

    def visualize(self):
        fig = plt.figure(figsize=(10,10))

        ee_traj_x = [pos[0] for pos in self.ee_trajectory]
        ee_traj_y = [pos[1] for pos in self.ee_trajectory]

        puck_traj_x = [pos[0] for pos in self.puck_trajectory]
        puck_traj_y = [pos[1] for pos in self.puck_trajectory]

        plt.xlim([-1.2, 1.2])
        plt.ylim([-0.6,0.6])
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.plot(ee_traj_x, ee_traj_y, label='End Effector')
        plt.plot(puck_traj_x, puck_traj_y, label="Puck")
        plt.legend()

        plt.show()


#Test

from air_hockey_challenge.framework.custom_environment_wrapper import CustomEnvironmentWrapper
from air_hockey_challenge.utils.kinematics import forward_kinematics
from air_hockey_challenge.utils.transformations import robot_to_world
from utils.trajectory_logger import TrajectoryLogger
from air_hockey_agent.dummy_agent import DummyAgent

if __name__ == '__main__':
    env = CustomEnvironmentWrapper(env="3dof-defend")

    path = [[-0.8,0,0],[-0.1,0,0]]

    agent = DummyAgent(env.base_env.env_info, path=path, steps_per_action=50)

    logger = TrajectoryLogger()

    obs = env.reset()
    agent.reset()

    step = 0
    while True:
        step +=1 
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()

        puck_pos = obs[env.base_env.env_info['puck_pos_ids']][:2]
        joint_pos = obs[env.base_env.env_info['joint_pos_ids']]

        mj_model = env.base_env.env_info['robot']['robot_model']
        mj_data = env.base_env.env_info['robot']['robot_data']

        ee_pos = forward_kinematics(mj_model, mj_data, joint_pos)[0]

        ee_pos = robot_to_world(env.base_env.env_info['robot']['base_frame'][0], ee_pos)[0][:2]
        puck_pos = robot_to_world(env.base_env.env_info['robot']['base_frame'][0], puck_pos)[0][:2]
        #print(f'EE:{ee_pos}\n\nPUCK:{puck_pos}')


        logger.append2trajectory(ee_pos, puck_pos, action)


        if done or step > env.info.horizon / 2:
            break


    logger.visualize()
