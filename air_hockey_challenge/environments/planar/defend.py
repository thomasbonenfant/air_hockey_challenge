import numpy as np

from air_hockey_challenge.environments.planar.single import AirHockeySingle
from air_hockey_challenge.constraints import *
import pandas as pd
from copy import deepcopy


class AirHockeyDefend(AirHockeySingle):
    """
    Class for the air hockey defending task.
    The agent tries to stop the puck at the line x=-0.6.
    If the puck get into the goal, it will get a punishment.
    """

    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):

        self.init_velocity_range = (0, 7)

        self.start_range = np.array([[0.29, 0.65], [-0.4, 0.4]])  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame

        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

    def setup(self, state=None):
        # Set initial puck parameters
        puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]

        lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
        angle = np.random.uniform(-0.5, 0.5)

        puck_vel = np.zeros(3)
        puck_vel[0] = -np.cos(angle) * lin_vel
        puck_vel[1] = np.sin(angle) * lin_vel
        puck_vel[2] = np.random.uniform(-10, 10, 1)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyDefend, self).setup(state)

    def reward(self, state, action, next_state, absorbing):

        """
        Custom reward function, it is in the form: r_tot = r_def + alfa * r_const

        - r_def: reward for the defending part
        - r_const: reward of the contsraints part
        - alpha: constant

            | x_desired - x_puck|
        - ----------------------                       , no short sides hit and no puck hit
              diagonale_tavolo

           |x_desired - x_puck|       ||v_puck||
        - ----------------------  -  -------------     , no short sides hit and puck hit
            diagonale_tavolo         numero_grande

             1
        - -------                                      , received goal by the offender 
           1 - Ɣ 
            
            1
         -------                                       , if success
          1 - Ɣ
        """

        alpha = 1  # TODO find a suitable value
        has_hit = False  # TODO implement a way to check if it happened
        c = 0  # constant, set to 0 for the first experiments
        gamma = 0.99  # TODO set the discount factor
        big_number = 10 ** 4  # TODO find a suitable value

        # compute table diagonal
        table_length = self.env_info['table']['length']
        table_width = self.env_info['table']['width']
        table_diag = np.sqrt(table_length ** 2 + table_width ** 2)

        # get ee position, puck position and velocities
        ee_pos = self.get_ee()[0]
        puck_pos = self.get_puck(state)[0]
        puck_vel = self.get_puck(state)[1][0:2]

        # compute x_desired as half of the defend band, just like the "success" in air_hockey_challenge_wrapper
        x_desired = (- 0.8 + 0.29) / 2 - 0.29  # TODO values are hardcoded from the check for the success

        ''' REWARD DEFEND '''
        has_hit = False  # TODO implement a way to check if it happened
        goal_received = False  # TODO implement a way to check if it happened
        short_sides_hit = True if puck_vel[0] > 0 and not has_hit else False

        # success, default value, no need to check info if we use it as default value
        reward_defend = 1 / (1 - gamma)

        if not short_sides_hit:
            if not has_hit:
                reward_defend = - abs(x_desired - puck_pos[0]) / table_diag
            else:
                reward_defend = - abs(x_desired - puck_pos[0]) / table_diag - (np.linalg.norm(puck_vel) / big_number)

        if goal_received:
            reward_defend = - 1 / (1 - gamma)

        ''' REWARD CONSTRAINTS'''
        # retrieve constraints
        constraint_list = ConstraintList()
        constraint_list.add(JointPositionConstraint(self.env_info))
        constraint_list.add(JointVelocityConstraint(self.env_info))
        constraint_list.add((EndEffectorConstraint(self.env_info)))

        self.env_info['constraints'] = constraint_list
        constraints = self.env_info['constraints']

        # RECOMPUTE CONSTRAINTS # FIXME is state the obs?
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
                slack_variables[key] = np.array(
                    list(map(lambda x: x if x > 0 else 0, slack_variables[key])))  # take only positive values
                slack_variables[key] *= constraint_weights[key]  # multiply by the penalty weights
                # FIXME is max of observation space ok?
                normalization_factor = max(
                    self.env_info['rl_info'].observation_space.high)  # correction factor, max of observation space
                slack_variables[key] = slack_variables[key] / normalization_factor

        # sum the slack variables multiplied by their weights divided by correction factor
        sum_slack = 0
        for value in slack_variables.values():
            sum_slack += sum(value)

        reward_constraints = - sum_slack / sum(constraint_weights.values())

        return (reward_defend + alpha * reward_constraints) / 2  # negative rewards shoud never go below -1

    def is_absorbing(self, state):
        puck_pos, puck_vel = self.get_puck(state)
        # If puck is over the middle line and moving towards opponent
        # if puck_pos[0] > 0 and puck_vel[0] > 0:
        #     return True

        if puck_vel[0] == 0 and puck_vel[1] == 0:
            return True
        # self.dataset.drop(index=self.dataset.index[-1], axis=0, inplace=True)
        absorbing = super().is_absorbing(state)
        if absorbing:
            # print("heeey")
            new_data = pd.DataFrame({'puck current pos X': [0],
                                     'puck current pos Y': [0],
                                     'puck current pos Yaw': [0],
                                     'puck current vel X': [0],
                                     'puck current vel Y': [0],
                                     'puck current vel Yaw': [0]})
            self.dataset = pd.concat([self.dataset, new_data], ignore_index=False)
            self.datasets.append(self.dataset)
            self.dataset = pd.DataFrame()

        return absorbing


if __name__ == '__main__':
    env = AirHockeyDefend()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    env.reset()
    while True:
        action = np.zeros(3)
        observation, reward, done, info = env.step(action)
        # env.render()
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