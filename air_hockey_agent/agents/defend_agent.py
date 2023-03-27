from math import sqrt
import time
import threading
import numpy as np
from scipy.interpolate import CubicSpline
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.framework.custom_environment_wrapper import CustomEnvironmentWrapper
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
from utils.trajectory_planner import plan_minimum_jerk_trajectory

''' Example of a custom defending agent to analyze insights from the 'step' function
 try to give always the same goal point to the agent, even if it does not defend porperly
 and check if the trajectory is smooth or not '''

class SimpleDefendingAgent(AgentBase):

    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.has_to_plan = True
        
        self.trajectory = []
        
        self.defend_line = -0.8
        self.traj_time = 1

        # flags to test different types of corrections
        self.has_to_compute_traj_time = True # if False then the agent will reach the point always in the same time=traj_time

    def reset(self):
        self.has_to_plan = True
    
    def _get_ee_pose_world_frame(self, obs):
        pose = self.get_ee_pose(obs)[0]
        return robot_to_world(self.env_info['robot']['base_frame'][0], pose)[0]
    
    def draw_action(self, obs):
        if self.has_to_plan:
            step = 0
            final_pos = self.plan_defend_point(obs)[:2]

            initial_pos = self._get_ee_pose_world_frame(obs)[:2]

            if self.has_to_compute_traj_time:

                # Compute traj_time conisidering constant puck's velocity

                puck_pos = robot_to_world(self.env_info['robot']['base_frame'][0], self.get_puck_pos(obs))[0][:2]
                distance = np.linalg.norm(puck_pos - final_pos, ord=2)

                #print('\nDistance: ', distance)

                puck_velocities = self.get_puck_vel(obs)[:2]
                velocity_module = np.linalg.norm(puck_velocities)
                
                self.traj_time = distance / velocity_module
                #print('\nTraj time: ', self.traj_time)

                # move the ee with a 10% anticipation on the traj_time
                self.traj_time = self.traj_time*(1 - 0.1)
                #print('\nCorrected traj time: ', self.traj_time)

            self.trajectory = plan_minimum_jerk_trajectory(initial_pos, final_pos, self.traj_time, delta_t=self.env_info['dt'])

        self.has_to_plan = False
        action = None
        
        if len(self.trajectory) > 0:
            action = self.trajectory[0]

            if len(self.trajectory) > 1:
                self.trajectory = self.trajectory[1:]
        
        
        return action
        
        # Uncomment to let the robot try to follow the puck
        #return self.plan_defend_point(obs)

    def plan_defend_point(self, obs):

        '''
        Find the intersection point of the puck's trajectory with the defend line
        '''

        defend_line = -0.80 # in the evaluation (air_hockey_challenge_wrapper) is hardcoded to -0.8 so the success might be distorted if different

        # get puck pos from the environment and convert it to world coordinates
        puck_pos = robot_to_world(self.env_info['robot']['base_frame'][0], self.get_puck_pos(obs))[0][:2]
                        
        #puck_velocities = robot_to_world(self.env_info['robot']['base_frame'][0], self.get_puck_vel(obs))[0][:2]
        puck_velocities = self.get_puck_vel(obs)[:2]
        
        # compute versor of the puck's velocity
        defend_dir_2d = puck_velocities / np.linalg.norm(puck_velocities)
        
        # compute intersection point
        lam = (defend_line - puck_pos[0]) / defend_dir_2d[0]
        ee_pos_y = puck_pos[1] + lam*defend_dir_2d[1]
       
        # check if y position violates the constraints
        lower_bound = -self.env_info['table']['width']/2 + self.env_info['mallet']['radius']
        upper_bound = self.env_info['table']['width']/2 - self.env_info['mallet']['radius']

        if  ee_pos_y > upper_bound: #TODO not working if multible rebounds
            ee_pos_y = upper_bound - (ee_pos_y - upper_bound)
        elif ee_pos_y < lower_bound:
            ee_pos_y = lower_bound + (lower_bound - ee_pos_y)
                    
        ee_pos_x = defend_line

        ee_pos = [ee_pos_x, ee_pos_y, 0]

        return np.array(ee_pos)
    
""" 
def main():
    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("tkAgg")

    env = CustomEnvironmentWrapper(env="3dof-defend")

    agent = SimpleDefendingAgent(env.base_env.env_info, steps_per_action=50)

    obs = env.reset()
    agent.reset()

    steps = 0
    while True:
        steps += 1
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()

        if done or steps > env.info.horizon / 2:
            nq = env.base_env.env_info['robot']['n_joints']
            if env.base_env.debug:
                trajectory_record = np.array(env.base_env.controller_record)
                fig, axes = plt.subplots(5, nq)
                nq_total = nq * env.base_env.n_agents
                for j in range(nq):
                    axes[0, j].plot(trajectory_record[:, j])
                    axes[0, j].plot(trajectory_record[:, j + nq_total])
                    axes[1, j].plot(trajectory_record[:, j + 2 * nq_total])
                    axes[1, j].plot(trajectory_record[:, j + 3 * nq_total])
                    axes[2, j].plot(trajectory_record[:, j + 4 * nq_total])
                    axes[3, j].plot(trajectory_record[:, j + 5 * nq_total])
                    axes[4, j].plot(trajectory_record[:, j + nq_total] - trajectory_record[:, j])
                plt.show()

            steps = 0
            obs = env.reset()
            agent.reset()


if __name__ == '__main__':
    main() """
