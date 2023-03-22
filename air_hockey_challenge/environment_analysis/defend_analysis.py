import time
import threading
import numpy as np
from scipy.interpolate import CubicSpline
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.framework.custom_environment_wrapper import CustomEnvironmentWrapper
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
from baseline.baseline_agent import BezierPlanner, TrajectoryOptimizer, PuckTracker


def build_agent(env_info, **kwargs):

    return DummyDefendingAgent(env_info, **kwargs)


''' Example of a custom defending agent to analyze insights from the 'step' function
 try to give always the same goal point to the agent, even if it does not defend porperly
 and check if the trajectory is smooth or not '''

class DummyDefendingAgent(AgentBase):

    def __init__(self, env_info, agent_id=1, **kwargs):
        super(DummyDefendingAgent, self).__init__(env_info, agent_id, **kwargs)
        self.last_cmd = None
        self.last_position = None # last position on the defend line TODO re-implement its update
        self.action = None # action that the draw_action will return endlessly
        self.joint_trajectory = None
        self.restart = True
        self.optimization_failed = False
        self.has_planned = False
        self.dt = 1 / self.env_info['robot']['control_frequency']
        self.ee_height = self.env_info['robot']["ee_desired_height"]

        
        self.bound_points = np.array([[-(self.env_info['table']['length'] / 2 - 0.05),
                                       -(self.env_info['table']['width'] / 2 - 0.05)],
                                      [-(self.env_info['table']['length'] / 2 - 0.05),
                                       (self.env_info['table']['width'] / 2 - 0.05)],
                                      [-0.3, (self.env_info['table']['width'] / 2 - 0.05)],
                                      [-0.3, -(self.env_info['table']['width'] / 2 - 0.05)]])
        self.bound_points = self.bound_points + np.tile([1.51, 0.], (4, 1))
        self.boundary_idx = np.array([[0, 1], [1, 2], [0, 3]])

        # Boundary points of the table
        table_bounds = np.array([[self.bound_points[0], self.bound_points[1]],
                                 [self.bound_points[1], self.bound_points[2]],
                                 [self.bound_points[2], self.bound_points[3]],
                                 [self.bound_points[3], self.bound_points[0]]])
        
        self.bezier_planner = BezierPlanner(table_bounds, self.dt)
        
        self.optimizer = TrajectoryOptimizer(self.env_info)
        
        if self.env_info['robot']['n_joints'] == 3:
            self.joint_anchor_pos = np.array([-0.9273, 0.9273, np.pi / 2])
        else:
            self.joint_anchor_pos = np.array([6.28479822e-11, 7.13520517e-01, -2.96302903e-11, -5.02477487e-01,
                                              -7.67250279e-11, 1.92566224e+00, -2.34645597e-11])

        self.puck_tracker = PuckTracker(env_info, agent_id=agent_id)
        self._obs = None

    def reset(self, obs):
        self.last_cmd = None
        self.joint_trajectory = []
        self.restart = True
        self.has_planned = False
        self.optimization_failed = False
        self.last_position = np.array([0,0])
        self._obs = None

        self.action = self.plan_trajectory(obs) # computes the action to repeat
    

    def draw_action(self, obs):
        if self.restart:
            self.restart = False
            self.puck_tracker.reset(self.get_puck_pos(obs))
            self.last_cmd = np.vstack([self.get_joint_pos(obs), self.get_joint_vel(obs)])
            
        
        self.puck_tracker.step(self.get_puck_pos(obs))
        self._obs = obs.copy()
        """ 
        if len(self.joint_trajectory) > 0:
            joint_pos_des, joint_vel_des = self.joint_trajectory[0]
            self.joint_trajectory = self.joint_trajectory[1:]
            self.last_cmd[1] = joint_vel_des
            self.last_cmd[0] = joint_pos_des
        else:
            self.last_cmd[1] = np.zeros(self.env_info['robot']['n_joints'])
            if not self.has_planned:
                time.sleep(0.005) """

        return self.action # returns always the same position, computed in the reset
        
        # Uncomment to let the robot try to follow the puck
        #return self.plan_trajectory(obs)


    def plan_trajectory(self, obs):

        '''
        Find the intersection point of the puck's trajectory with the defend line
        '''

        t_predict = 1.0
        defend_line = -0.8

        # get puck pos from the environment and convert it to world coordinates
        puck_pos = robot_to_world(self.env_info['robot']['base_frame'][0], self.get_puck_pos(obs))[0][:2]

        print('\npuck_pos: ', puck_pos)

        # FIXME perchè se converto le coordinate delle velocità non funziona? 
        #puck_velocities = robot_to_world(self.env_info['robot']['base_frame'][0], self.get_puck_vel(obs))[0][:2] # x,y,θ velocities, keep only x and y
        puck_velocities = self.get_puck_vel(obs)[:2]
        print('\npuck_velocities: ', puck_velocities)

        # compute versor of the puck's velocity
        defend_dir_2d = puck_velocities / np.linalg.norm(puck_velocities)
        print('\ndefend_dir_2d',defend_dir_2d)
        
        # compute intersection point
        #ee_pos_y = ((defend_line - puck_pos[0])/defend_dir_2d) * puck_velocities
       
        ee_pos_y = defend_dir_2d[1] + puck_pos[1] #+ self.env_info['mallet']['radius']

        # check if y position violates the constraints
        lower_bound = -self.env_info['table']['width']/2 + self.env_info['mallet']['radius']
        upper_bound = self.env_info['table']['width']/2 - self.env_info['mallet']['radius']

        if  lower_bound < ee_pos_y < upper_bound:
            final_ee_pos_y = ee_pos_y
            self.last_position[1] = ee_pos_y
        else:
            final_ee_pos_y = self.last_position[1]
        
        print('\nee_pos_y', ee_pos_y)

        final_ee_pos_x = ee_pos_x = defend_line + self.env_info['mallet']['radius']
        
        '''
        if puck_pos[0] >= defend_line:
            final_ee_pos_x = ee_pos_x
            self.last_position[0] = ee_pos_x
        else: # don't update if it is behind the defend line
            final_ee_pos_y = self.last_position[1] #0
            final_ee_pos_x = self.last_position[0] #defend_line
        '''

        ee_pos = [final_ee_pos_x, final_ee_pos_y, 0]

        return np.array(ee_pos)



def main():
    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("tkAgg")

    env = CustomEnvironmentWrapper(env="3dof-defend")

    agent = DummyDefendingAgent(env.base_env.env_info, steps_per_action=50)

    obs = env.reset()
    agent.reset(obs)

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
            agent.reset(obs)


if __name__ == '__main__':
    main()
