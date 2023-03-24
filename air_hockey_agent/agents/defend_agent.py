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
            
            self.trajectory = plan_minimum_jerk_trajectory(initial_pos, final_pos, 2, self.env_info['dt'])


            
        self.has_to_plan = False
        action = None

        if len(self.trajectory) > 0:
            action = self.trajectory[0]

            if len(self.trajectory) > 1:
                self.trajectory = self.trajectory[1:]
        
        
        return action
        
        # Uncomment to let the robot try to follow the puck
        #return self.plan_trajectory(obs)


    def plan_defend_point(self, obs):

        '''
        Find the intersection point of the puck's trajectory with the defend line
        '''

        defend_line = -0.85 # in the evaluation is hardcoded to -0.8 so the success might be distorted

        # get puck pos from the environment and convert it to world coordinates
        puck_pos = robot_to_world(self.env_info['robot']['base_frame'][0], self.get_puck_pos(obs))[0][:2]

        #print('\npuck_pos: ', puck_pos)

        puck_velocities = self.get_puck_vel(obs)[:2]
        #print('\npuck_velocities: ', puck_velocities)

        # compute versor of the puck's velocity
        defend_dir_2d = puck_velocities / np.linalg.norm(puck_velocities)
        #print('\ndefend_dir_2d',defend_dir_2d)
        
        # compute intersection point
        lam = (defend_line - puck_pos[0]) / defend_dir_2d[0]
        ee_pos_y = puck_pos[1] + lam*defend_dir_2d[1]
       
        #ee_pos_y = defend_dir_2d[1] + puck_pos[1] #+ self.env_info['mallet']['radius']

        # check if y position violates the constraints
        lower_bound = -self.env_info['table']['width']/2 + self.env_info['mallet']['radius']
        upper_bound = self.env_info['table']['width']/2 - self.env_info['mallet']['radius']

        if  ee_pos_y > upper_bound: #TODO not working if multible rebounds
            ee_pos_y = upper_bound - (ee_pos_y - upper_bound)
        elif ee_pos_y < lower_bound:
            ee_pos_y = lower_bound + (lower_bound - ee_pos_y)
                    
        #print('\nee_pos_y', ee_pos_y)

        ee_pos_x = defend_line        
        
        ee_pos = [ee_pos_x, ee_pos_y, 0]

        return np.array(ee_pos)
