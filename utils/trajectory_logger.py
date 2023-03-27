import matplotlib.pyplot as plt
import numpy as np

class TrajectoryLogger():

    '''
    A support class that implements methods to plot the result of an experiment; it plots
        - borders of the table, dashed red
        - puck's trajectory
        - ee_trajectory
        - action: the command given but the controller
    '''

    def __init__(self):
        self.ee_trajectory = []
        self.puck_trajectory = []
        self.action_position = []

    def append2trajectory(self, ee_position, puck_position, action):
        '''
        Saves new position data
        Arguments:
            -   ee_position: np.ndarray of size (2,)
            -   puck_position: np.ndarray of size(2,)
            -   action: np.ndarray of size(2,)  
        '''
        self.ee_trajectory.append(ee_position)
        self.puck_trajectory.append(puck_position)
        self.action_position.append(action)

    def visualize(self, env_info):
        fig = plt.figure(figsize=(8,8))

        ee_traj_x = [pos[0] for pos in self.ee_trajectory]
        ee_traj_y = [pos[1] for pos in self.ee_trajectory]

        puck_traj_x = [pos[0] for pos in self.puck_trajectory]
        puck_traj_y = [pos[1] for pos in self.puck_trajectory]

        action_traj_x = [pos[0] for pos in self.action_position]
        action_traj_y = [pos[1] for pos in self.action_position]

        plt.xlim([-1.2, 1.2])
        plt.ylim([-0.6,0.6])
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.plot(ee_traj_x, ee_traj_y, label='End Effector')
        plt.plot(puck_traj_x, puck_traj_y, label="Puck")
        plt.plot(action_traj_x, action_traj_y, label="Action")

        # Plot table borders
        half_table_length = env_info['table']['length']/2
        half_table_width = env_info['table']['width']/2
        puck_radius = env_info['puck']['radius']

        [plt.vlines(x , -1 ,1, colors='r', linestyles='dashed') for x in (-half_table_length + puck_radius, half_table_length - puck_radius)]
        [plt.hlines(y, -1,1, colors='r', linestyles='dashed') for y in (-half_table_width + puck_radius, half_table_width - puck_radius)]

        plt.legend()

        plt.show()

    def plot_coordinate_traj(self):
        '''
        Plots the values of x and y of the ee in the trajectory
        '''
        fig, ax = plt.subplots(2,1, figsize=(8,8))

        ee_traj_x = [pos[0] for pos in self.ee_trajectory]
        ee_traj_y = [pos[1] for pos in self.ee_trajectory]

        tt = np.linspace(1, len(ee_traj_x), len(ee_traj_x))

        ax[0].plot(ee_traj_x, label='X trajectory')
        ax[0].legend()
        ax[1].plot(ee_traj_y, label='Y trajectory')
        ax[1].legend()
        plt.show()
        