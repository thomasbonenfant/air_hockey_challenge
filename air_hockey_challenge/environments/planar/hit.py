import numpy
import numpy as np
from air_hockey_challenge.environments.planar.single import AirHockeySingle
from air_hockey_challenge.constraints import *

from copy import deepcopy

class AirHockeyHit(AirHockeySingle):
    """
    Class for the air hockey hitting task.
    """

    def __init__(self, gamma=0.99, horizon=500, moving_init=False, viewer_params={}):
        """
        Constructor
        Args:
            random_init(bool, False): If true, initialize the puck at random position.
            init_robot_state(string, "right"): The configuration in which the robot is initialized. "right", "left",
                                                "random" available
        """
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        self.moving_init = moving_init
        hit_width = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'] - \
                    self.env_info['mallet']['radius'] * 2
        self.hit_range = np.array([[-0.7, -0.2], [-hit_width, hit_width]])  # Table Frame
        self.init_velocity_range = (0, 0.5)  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame

    def setup(self, state=None):
        # Initial position of the puck
        puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]

        # self.init_state = np.array([-0.9273, 0.9273, np.pi / 2])

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])

        if self.moving_init:
            lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
            angle = np.random.uniform(-np.pi/2 - 0.1, np.pi/2 + 0.1)
            puck_vel = np.zeros(3)
            puck_vel[0] = -np.cos(angle) * lin_vel
            puck_vel[1] = np.sin(angle) * lin_vel
            puck_vel[2] = np.random.uniform(-2, 2, 1)

            self._write_data("puck_x_vel", puck_vel[0])
            self._write_data("puck_y_vel", puck_vel[1])
            self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyHit, self).setup(state)
        
    def reward(self, state, action, next_state, absorbing):

        """
        Custom reward function, it is in the form: r_tot = r_hit + alpha * r_const
        
        - r_hit: reward for the hitting part
        - r_const: reward for the constraint part
        - alpha: constant

        r_hit:

              || p_ee - p_puck||
          -  ------------------ - c * ||a||      , if no hit
              0.5 * diag_tavolo


                vx_hit
            -------------- - c * ||a||          , if has hit
              big_number     


               1
            -------                             , if goal
             1 - Ɣ
        """
        # TODO parametri da aggiungere al main.py per gli addestramenti (repo oac-explore)

        alpha = 1           # TODO find a suitable value
        has_hit = False     # TODO implement a way to check if it happened
        c = 0               # constant, set to 0 for the first experiments
        gamma = 0.998       # 0.998 means 500 steps of horizon, they train for 500 steps (0,997 -> ~333 steps)
        big_number = 10**4  # TODO find a suitable value, guardare limiti dello spazio di stato (observation_space)

        # compute table diagonal
        table_length = self.env_info['table']['length']
        table_width = self.env_info['table']['width']
        table_diag = np.sqrt(table_length**2 + table_width**2)

        # get ee and puck position
        ee_pos = self.get_ee()[0] 
        puck_pos = self.get_puck(state)[0][0:2]  # FIXME takes observation as input, state should be fine

        ''' REWARD HIT '''
        # goal case, default option 
        reward_hit = 1 / (1 - gamma)
        
        # no hit case
        if not has_hit:
            reward_hit = - (np.linalg.norm(ee_pos[0:2] - puck_pos) / 0.5 * table_diag) - c * np.linalg.norm(action)
        
        # hit case
        if has_hit:
            ee_x_vel = self.get_ee()[1][3]  # retrieve the linear x velocity

            reward_hit = (ee_x_vel / big_number) - c * np.linalg.norm(action)

        ''' REWARD CONSTRAINTS '''
        # retrieve constraints
        constraint_list = ConstraintList()
        constraint_list.add(JointPositionConstraint(self.env_info))
        constraint_list.add(JointVelocityConstraint(self.env_info))
        constraint_list.add((EndEffectorConstraint(self.env_info)))

        self.env_info['constraints'] = constraint_list
        constraints = self.env_info['constraints']

        # RECOMPUTE CONSTRAINTS
        info = dict()
        info['constraints_value'] = deepcopy(constraints.fun(state[self.env_info['joint_pos_ids']],
                                                             state[self.env_info['joint_vel_ids']]))

        # import penalty weights from a dictionary in evaluate_agent
        from air_hockey_challenge.framework.evaluate_agent import PENALTY_POINTS
        constraint_weights = PENALTY_POINTS

        # retrieve slack variables and multiply for penalty weights
        # if slack_variable > 0 then the constraint is violated
        slack_variables = info['constraints_value']

        for key in slack_variables.keys():
            if key in constraint_weights.keys():
                slack_variables[key] = np.array(list(map(lambda x: x if x > 0 else 0, slack_variables[key])))  # take only positive values
                slack_variables[key] *= constraint_weights[key]  # multiply by the penalty weights
                # FIXME is max of observation space ok?
                normalization_factor = max(self.env_info['rl_info'].observation_space.high)  # correction factor, max of observation space
                slack_variables[key] = slack_variables[key] / normalization_factor

        # sum the slack variables multiplied by their weights divided by correction factor
        sum_slack = 0
        for value in slack_variables.values():
            sum_slack += sum(value)

        reward_constraints = - sum_slack / sum(constraint_weights.values())

        return (reward_hit + alpha * reward_constraints) / 2  # negative rewards should never go below -1

    def is_absorbing(self, obs):
        _, puck_vel = self.get_puck(obs)
        # Stop if the puck bounces back on the opponents wall
        if puck_vel[0] < -0.6:
            return True
        return super(AirHockeyHit, self).is_absorbing(obs)


if __name__ == '__main__':
    env = AirHockeyHit(moving_init=False)

    env.reset()
    #env.render()
    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.zeros(3)

        observation, reward, done, info = env.step(action)
        #env.render()

        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
