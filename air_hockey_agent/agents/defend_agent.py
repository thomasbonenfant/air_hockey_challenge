import time
import threading
import numpy as np
from scipy.interpolate import CubicSpline
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.framework.custom_environment_wrapper import CustomEnvironmentWrapper
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
from baseline.baseline_agent import BezierPlanner, TrajectoryOptimizer, PuckTracker

''' Example of a custom defending agent to analyze insights from the 'step' function
 try to give always the same goal point to the agent, even if it does not defend porperly
 and check if the trajectory is smooth or not '''

class SimpleDefendingAgent(AgentBase):

    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.has_to_plan = True
        self.action = None
        

    def reset(self):
        self.has_to_plan = True
    

    def draw_action(self, obs):
        if self.has_to_plan:
            self.action = self. plan_trajectory(obs)
            
        self.has_to_plan = False


        return self.action # returns always the same position, computed in the reset
        
        # Uncomment to let the robot try to follow the puck
        #return self.plan_trajectory(obs)


    def plan_trajectory(self, obs):

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
    main()
