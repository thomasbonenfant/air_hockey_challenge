from math import sqrt
import time
import threading
import numpy as np
from scipy.interpolate import CubicSpline
from air_hockey_challenge.framework.agent_ee_base import AgentEEBase
from air_hockey_challenge.framework.custom_environment_wrapper import CustomEnvironmentWrapper
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
from utils.trajectory_planner import Planner, make_d_f

''' Example of a custom defending agent to analyze insights from the 'step' function
 try to give always the same goal point to the agent, even if it does not defend porperly
 and check if the trajectory is smooth or not '''

class SimpleDefendingAgent(AgentEEBase):

    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.has_to_plan = True
        
        self.trajectory = []
        
        self.defend_line = -0.8
        self.traj_time = 1

        self.v_min = -3
        self.v_max = 3
        self.a_min = -5
        self.a_max = 5

        self.old_vel = None
        self.old_pos = None

        # flags to test different types of corrections
        self.has_to_compute_traj_time = True # if False then the agent will reach the point always in the same time=traj_time

    def reset(self):
        self.has_to_plan = True
        self.old_vel = None
        self.old_pos = None
    
    def _get_ee_pose_world_frame(self, obs):
        '''
            TODO: remove this function because it is now unnecessary
        '''
        pose = self.get_ee_pose(obs)[0]
        return robot_to_world(self.env_info['robot']['base_frame'][0], pose)[0]

    def _plan_trajectory(self, T, dt, final_pos, final_vel, final_acc, initial_pos, initial_vel, initial_acc):
                
        df = final_pos - initial_pos
        v0 = initial_vel
        a0 = initial_acc
        vf = final_vel
        af = final_acc


        planner = Planner(3, 1e5, 1e3)
        weigths = planner.plan(T, vf, df, af, v0, a0, self.a_min, self.a_max, self.v_min, self.v_max, slack=False)
        
        if weigths is None:
            print('No feasible solution without slack')
            weigths = planner.plan(T, vf, df, af, v0, a0, self.a_min, self.a_max, self.v_min, self.v_max, slack=True)
        
        if weigths is None:
            print("Can't plan")
            return np.array([initial_pos])

        n_steps = int((T-dt)/dt)
        tt = np.linspace(dt, T, n_steps)

        fun = make_d_f(initial_pos, v0, a0, weigths)

        return np.array([fun(t) for t in tt])

    def compute_traj_time(self, obs):
        puck_pos = robot_to_world(self.env_info['robot']['base_frame'][0], self.get_puck_pos(obs))[0][:2]
        puck_vel = self.get_puck_vel(obs)[:2]

        x_dist = np.abs(puck_pos[0] - self.defend_line)
        traj_time = x_dist / np.abs(puck_vel[0])

        traj_time *= 0.9

        return traj_time


    def draw_action(self, obs):

        dt = self.env_info['dt']

        if self.has_to_plan:
            step = 0
            final_pos = self.plan_defend_point(obs)[:2]
            initial_pos = self.get_ee_pose(obs)[:2]
            
            initial_vel = (initial_pos - self.old_pos) / dt if self.old_pos is not None else np.array([0,0])
            initial_acc = (initial_vel - self.old_vel) / dt if self.old_vel is not None else np.array([0,0])
            
            if self.has_to_compute_traj_time:
                self.traj_time = self.compute_traj_time(obs)

            final_vel = np.array([0,0])
            final_acc = np.array([0,0])

            x_traj = self._plan_trajectory(self.traj_time, dt, final_pos[0], final_vel[0], final_acc[0], initial_pos[0], \
                                            initial_vel[0], initial_acc[0])
            y_traj = self._plan_trajectory(self.traj_time, dt, final_pos[1], final_vel[1], final_acc[1], initial_pos[1] ,\
                                            initial_vel[1], initial_acc[1])

            self.trajectory = np.vstack((x_traj, y_traj, [0 for i in range(len(x_traj))])).T    

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
    
