"""
Agent working via a Rule-based Policy
High-level action: control the end-effector
"""
# Libraries
import math
import os.path
import select
import time
import pickle

import numpy as np
import osqp
import scipy
from scipy import sparse
from scipy.interpolate import CubicSpline
from enum import Enum

from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_agent.model import State
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
from air_hockey_agent.agents.kalman_filter import PuckTracker
from air_hockey_agent.agents.optimizer import TrajectoryOptimizer

from air_hockey_agent.agents.rule_based_agent import PolicyAgent
from air_hockey_agent.agents.agents import DefendAgent
from baseline.baseline_agent.baseline_agent import BaselineAgent
from air_hockey_agent.agents.state_machine import StateMachine

# Macros
EE_HEIGHT = 0.0645  # apparently 0.0646 works better then 0.0645
WINDOW_SIZE = 200
PREV_BETA = 0
FINAL = 10
DEFEND_LINE = -0.8
DEFAULT_POS = np.array([DEFEND_LINE, 0, EE_HEIGHT])
DES_ACC = 0.05
best_hit_sample = np.array([0.993793, 3.008965, 3e-2, 0.01002]) / 100
best_hit = np.array([0.01, 0.03, 3e-4, 1e-4])
BEST_PARAM = dict(
    hit=best_hit_sample,
    defend=None,
    prepare=None,
    home=None
)

# Tasks enumeration
class Tasks(Enum):
    HOME    = 0
    HIT     = 1
    DEFEND  = 2
    REPEL   = 3
    PREPARE = 4


# Class implementing the rule based policy
class HierarchicalAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, task: str = "home", **kwargs):
        # Superclass initialization
        super().__init__(env_info, agent_id, **kwargs)

        # INSTANTIATE AGENTS -------------------------------------------------------------------
        self.rule_based_agent = PolicyAgent(env_info, **kwargs)
        self.baseline_agent = BaselineAgent(env_info, **kwargs)
        #with open("env_info_single_agent/env_infos.pkl", "rb") as fp:
        #    env_info_hit, env_info_defend = pickle.load(fp)
        self.repel_defend_agent = DefendAgent(env_info, env_label="7dof-defend", **kwargs)
        self.defend_agent = DefendAgent(env_info, env_label="tournament", **kwargs)

        self.optimizer = TrajectoryOptimizer(self.env_info)  # optimize joint position of each trajectory point

        # State machine to select the task
        self.state_machine = StateMachine()

        # Kalman filters
        self.puck_tracker = PuckTracker(self.env_info, agent_id)

        self.reset_filter = True

        # Task allocation
        self.task = task
        self.previous_task = task
        self.task_change_counter = 1
        self.step_counter = 0
        self.changing_task = False

        # Thetas
        assert self.task in ["hit", "defend", "prepare", "home"], f"Illegal task: {self.task}"
        self.theta = BEST_PARAM[self.task]

        # HIT: define initial values
        # angle between ee->puck and puck->goal lines
        self.previous_beta = PREV_BETA
        self.prev_side = None  # previous field side the puck was id
        self.last_position = np.zeros(3)
        self.last_r_puck_pos = np.zeros(3)  # puck pos at t-1
        self.last_w_puck_vel = np.zeros(3)  # puck vel at t-1
        self.last_action = np.zeros((2, 7))
        self.q_anchor = None
        self.final = FINAL

        # DEFEND: define initial values
        self.desired_from = None

        # RETURN Position
        self.default_action = DEFAULT_POS

        # Environment specs
        self.frame = self.env_info['robot']['base_frame'][0]
        self.table_length = self.env_info['table']['length']
        self.table_width = self.env_info['table']['width']
        self.goal_pos = np.array([self.table_length / 2, 0])
        self.mallet_radius = self.env_info['mallet']['radius']
        self.puck_radius = self.env_info['puck']['radius']
        self.defend_line = DEFEND_LINE

        # Episode parameters
        self.time = dict(
            abs=0,
            adjustment=0,
            acceleration=0,
            smash=0
        )
        self.timer = 0  # time for which the puck is in a given side, reset to 0 at each side change
        self.iteration_counter = -1  # absolute number of steps, not reset

        self.done = False
        self.has_hit = False
        self.can_hit = False  # can hit considering enemy position
        self.keep_hitting = False  # Whether to hit again or not in the prepare
        self.state = State
        self.last_ds = 0
        self.phase = "wait"

        # Array where to save task change results
        self.task_change_log = []

    def reset(self):
        # Generic stuff
        self.previous_beta = PREV_BETA
        self.prev_side = None
        self.last_position = np.zeros(3)
        self.last_action = np.zeros((2, 7))
        self.last_r_puck_pos = np.zeros(3)
        self.last_w_puck_vel = np.zeros(3)
        self.q_anchor = None
        self.final = FINAL
        self.desired_from = None
        self.task_change_counter = 1
        self.step_counter = 0
        self.changing_task = False

        # Resetting tasks
        self.previous_task = 'home'
        self.task = 'home'

        # Kalman filter
        self.reset_filter = True

        # Episode parameters
        self.done = False
        self.has_hit = False
        self.can_hit = False
        self.keep_hitting = False
        self.time = dict(
            abs=0,
            adjustment=0,
            acceleration=0,
            smash=0
        )
        self.timer = 0

        self.state = State
        self.last_ds = 0
        self.phase = "wait"

        # Resetting agents
        self.rule_based_agent.reset()
        self.defend_agent.reset()
        self.repel_defend_agent.reset()
        self.baseline_agent.reset()

    # New draw action
    def draw_action(self, observation):
        # Increase the time step
        # self.time += self.env_info["dt"]
        self.time["abs"] += 1
        self.iteration_counter += 1

        # set the self.prev
        if self.timer == 0:
            self.prev_side = np.sign(self.get_puck_pos(observation)[0] - 1.51) * -1

        # Get useful info from the state
        self.state.r_puck_pos = observation[0:3]
        self.state.r_puck_vel = observation[3:6]

        if self.time["abs"] == 1:
            self.state.r_joint_pos = observation[6:13]
            self.state.r_joint_vel = observation[13:20]

        if self.time["abs"] > 1:
            self.state.r_joint_pos = self.last_action[0]
            self.state.r_joint_vel = self.last_action[1]

        self.state.r_adv_ee_pos = observation[-3:]  # adversary ee position in robot coordinates

        # Compute EE pos
        self.state.r_ee_pos = self._apply_forward_kinematics(joint_pos=self.state.r_joint_pos)

        # KALMAN FILTER ----------------------------------------------------------
        # reset filter with the initial positions
        if self.reset_filter:
            self.reset_filter = False
            self.puck_tracker.reset(self.state.r_puck_pos)

        # predict next state, step does a single prediction step
        self.puck_tracker.step(self.state.r_puck_pos)

        # Reduce noise with kalman filter
        # self.state.r_puck_pos = self.puck_tracker.state[[0, 1, 4]]  # state contains pos and velocity
        # self.state.r_puck_vel = self.puck_tracker.state[[2, 3, 5]]

        # Convert in WORLD coordinates 2D
        self.state.w_puck_pos, _ = robot_to_world(base_frame=self.frame,
                                                  translation=self.state.r_puck_pos)
        self.state.w_ee_pos, _ = robot_to_world(base_frame=self.frame,
                                                translation=self.state.r_ee_pos)
        self.state.w_adv_ee_pos, _ = robot_to_world(base_frame=self.frame,
                                                    translation=self.state.r_adv_ee_pos)

        # PICK THE ACTION -----------------------------------------------------------------------------------------
        self.previous_task = self.task
        self.task = self.simplified_pick_task()
        #self.task = self.select_task()
        self.task = self.state_machine.select_task(previous_state=self.previous_task, desired_next_state=self.task)

        if self.previous_task != self.task:
            #print(f'{self.previous_task} --> {self.task}')
            # reset agents
            self.rule_based_agent.reset()
            self.defend_agent.reset()
            self.repel_defend_agent.reset()

        '''
        # TODO forcing remove and test
        # Force going home after defend repel or hit
        #if self.previous_task == "defend" or self.previous_task == "repel" or self.previous_task == "hit":
        #    self.task = "home"

        
        if self.previous_task == "hit" and self.task == "prepare":
            self.task = "hit"

        if self.previous_task == "defend" and self.task == "repel":
            self.task = "defend"

        if self.previous_task == "repel" and self.task == "defend":
            self.task = "repel"'''

        # execute the action until completion
        # TODO find a way to understand if the defend and the repel are complete
        if self.done is False:
            if self.task == "defend":
                action = self.defend_agent.draw_action(observation)
                if self.state.r_puck_pos[0] > 1.66:
                    self.done = True
            elif self.task == "repel":
                action = self.repel_defend_agent.draw_action(observation)
                if self.state.r_puck_pos[0] > 1.66:
                    self.done = True
            else:
                # self.task == "hit" or self.task == "prepare" or self.task == "home":
                self.rule_based_agent.set_task(self.task)
                action = self.rule_based_agent.draw_action(observation)

                if self.task == "hit":
                    self.done = self.rule_based_agent.hit_completed
                elif self.task == "prepare":
                    self.done = self.rule_based_agent.prepare_completed
        else:
            self.rule_based_agent.set_task("home")
            action = self.rule_based_agent.draw_action(observation)
            self.done = self.rule_based_agent.hit_completed # todo bring self.done back to False after finishing the hit?
            if self.done is False:
                print('Home completed')

        # UPDATE TIMER
        if np.sign(self.get_puck_pos(observation)[0] - 1.51) == self.prev_side:
            self.timer += self.env_info['dt']
            self.defend_agent.set_timer(self.timer)
            self.repel_defend_agent.set_timer(self.timer)
        else:
            self.prev_side *= -1
            self.timer = 0

        self.step_counter += 1
        return action

    def old_draw_action(self, observation):
        """
        Description:
            This function draws an action given the observation and the task.

        Args:
            observation (numpy.array([...])): see env

        Returns:
            numpy.array(2, 7): desired joint positions and velocities
        """
        # Increase the time step
        # self.time += self.env_info["dt"]
        self.time["abs"] += 1
        self.iteration_counter += 1

        # set the self.prev
        if self.timer == 0:
            self.prev_side = np.sign(self.get_puck_pos(observation)[0] - 1.51) * -1

        # Get useful info from the state
        self.state.r_puck_pos = observation[0:3]
        self.state.r_puck_vel = observation[3:6]

        if self.time["abs"] == 1:
            self.state.r_joint_pos = observation[6:13]
            self.state.r_joint_vel = observation[13:20]

        if self.time["abs"] > 1:
            self.state.r_joint_pos = self.last_action[0]
            self.state.r_joint_vel = self.last_action[1]

        self.state.r_adv_ee_pos = observation[-3:]  # adversary ee position in robot coordinates

        # Compute EE pos
        self.state.r_ee_pos = self._apply_forward_kinematics(joint_pos=self.state.r_joint_pos)

        # KALMAN FILTER ----------------------------------------------------------
        # reset filter with the initial positions
        if self.reset_filter:
            self.reset_filter = False
            self.puck_tracker.reset(self.state.r_puck_pos)

        # predict next state, step does a single prediction step
        self.puck_tracker.step(self.state.r_puck_pos)

        # Reduce noise with kalman filter
        # self.state.r_puck_pos = self.puck_tracker.state[[0, 1, 4]]  # state contains pos and velocity
        # self.state.r_puck_vel = self.puck_tracker.state[[2, 3, 5]]

        # Convert in WORLD coordinates 2D
        self.state.w_puck_pos, _ = robot_to_world(base_frame=self.frame,
                                                  translation=self.state.r_puck_pos)
        self.state.w_ee_pos, _ = robot_to_world(base_frame=self.frame,
                                                translation=self.state.r_ee_pos)
        self.state.w_adv_ee_pos, _ = robot_to_world(base_frame=self.frame,
                                                    translation=self.state.r_adv_ee_pos)

        # PICK THE ACTION ----------------------------------------------------------
        if not self.changing_task:
            self.previous_task = self.task

        # Perform the same task for a given amount of steps
        if self.step_counter > 20:
            #self.task = self.simplified_pick_task()
            self.task = self.select_task()

        if self.previous_task != self.task and not self.changing_task:
            self.last_ds = self.rule_based_agent.last_ds
            self.changing_task = True
            self.step_counter = 0
            # self.rule_based_agent.reset()
            # self.defend_agent.reset()
            # self.repel_defend_agent.reset()

            # Force going home after defend repel or hit
            if self.previous_task == "defend" or self.previous_task == "repel" or self.previous_task == "hit":
                self.task = "home"

            if self.previous_task == "hit" and self.task == "prepare":
                self.task = "hit"

            if self.previous_task == "defend" and self.task == "repel":
                self.task = "defend"

            if self.previous_task == "repel" and self.task == "defend":
                self.task = "repel"

            '''
            if (self.previous_task == "hit") and self.task == "home":
                if self.rule_based_agent.has_hit:
                    radius = np.sqrt(
                        (self.state.w_ee_pos[0] - DEFAULT_POS[0]) ** 2 + (self.state.w_ee_pos[1] - DEFAULT_POS[1]) ** 2
                    )

                    rounded_radius = round(radius - (self.mallet_radius + self.puck_radius), 5)

                    if rounded_radius <= 0.2:
                        self.task = "home"
                        self.changing_task = True
                    else:
                        print('here')
                        self.task = "hit"
                        self.changing_task = False
            '''

        #self.changing_task = False
        if self.changing_task:
            action = self.change_action_smoothly(previous_task=self.previous_task, current_task=self.task, observation=observation, steps=10)
        else:
            if self.task == "defend":
                action = self.defend_agent.draw_action(observation)
            elif self.task == "repel":
                action = self.repel_defend_agent.draw_action(observation)
            #elif self.task == "home":
            #    action = self.baseline_agent.draw_action(observation)
            else:
                self.rule_based_agent.set_task(self.task)
                action = self.rule_based_agent.draw_action(observation)

        if self.state.w_ee_pos[2] <= 0:
            #self.reset()
            print('WRONG EE POS, RESET')

        # UPDATE TIMER
        if np.sign(self.get_puck_pos(observation)[0] - 1.51) == self.prev_side:
            self.timer += self.env_info['dt']
            self.defend_agent.set_timer(self.timer)
            self.repel_defend_agent.set_timer(self.timer)
        else:
            self.prev_side *= -1
            self.timer = 0

        self.step_counter += 1
        return action

    def select_task(self):

        selected_task = self.task  # Keep the same task by default
        r_puck_pos = self.state.r_puck_pos[:2]
        w_puck_pos = self.state.w_puck_pos[:2]
        r_puck_vel = self.state.r_puck_vel[:2]
        w_puck_vel = self.state.w_puck_vel[:2]

        puck_velocity = np.linalg.norm(r_puck_vel)

        defend_vel_threshold = 0.3
        repel_vel_threshold = 0.8

        # HOME CONDITIONS
        if r_puck_pos[0] > 1.36:
            if self.rule_based_agent.hit_completed or self.rule_based_agent.prepare_completed:
                selected_task = "home"

        # HIT CONDITIONS
        if r_puck_pos[0] < 1.36:
            if puck_velocity < defend_vel_threshold:
                if self.check_enough_space():
                    selected_task = "hit"

        # PREPARE CONDITIONS
        if r_puck_pos[0] < 1.36:
            if puck_velocity < defend_vel_threshold:
                if not self.check_enough_space():
                    selected_task = "prepare"

        # DEFEND CONDITIONS
        if r_puck_pos[0] <= 1.66:
            if puck_velocity >= defend_vel_threshold:
                selected_task = "defend"

        # REPEL CONDITIONS
        if r_puck_pos[0] <= 1.66:
            if puck_velocity >= repel_vel_threshold:
                selected_task = "repel"

        return selected_task

    def simplified_pick_task(self):
        """
        Naïve version of the pick task but apparently te best one

        Returns
        -------
            The task from which retrieve the action
        """

        picked_task = self.previous_task  # fixme problem with the fact that it sometimes goes home instead of finishing the task
        defend_vel_threshold = 0.3
        repel_vel_threshold = 0.8  # puck too fast, just send it back

        w_puck_pos = self.state.w_puck_pos
        r_puck_pos = self.state.r_puck_pos
        r_puck_vel = self.state.r_puck_vel

        # Stay home if the puck is the enemy side
        if r_puck_pos[0] >= 1.36:  # don't go over the unreachable side
            picked_task = "home"

        else:
            # Puck coming toward the agent
            if r_puck_vel[0] < 0:
                # fast puck
                puck_velocity = np.linalg.norm(r_puck_vel[:2])
                if puck_velocity > defend_vel_threshold:
                    if puck_velocity > repel_vel_threshold:
                        picked_task = "repel"
                    else:
                        picked_task = "defend"
                else:
                    enough_space = self.check_enough_space()
                    picked_task = "hit" if enough_space else "prepare"

            else:
                if np.linalg.norm(r_puck_vel[:2]) < 10:
                    enough_space = self.check_enough_space()
                    picked_task = "hit" if enough_space else "prepare"
                else:
                    picked_task = "home"

        return picked_task

    # todo remove, old
    def updated_pick_task(self):
        """
        Description:
            This function will retrieve the most proper task to perform among: hit, defend, prepare and home

        Returns:
            the most appropriate task
        """

        picked_task = None
        defend_threshold = 0.05  # the line over the half in the enemy side to start defend
        defend_vel_threshold = 0.1  # puck norm velocity at which start the defend
        prepare_threshold = 0.05  # strip around borders to perform prepare

        # Stay idle at the home if the puck is in the enemy side
        if self.state.w_puck_pos[0] > defend_threshold:
            return "home"

        if self.state.w_puck_pos[0] < defend_threshold and \
                np.linalg.norm(self.state.r_puck_vel[0:2]) > defend_vel_threshold and self.state.r_puck_vel[0] < 0:
            picked_task = "defend"
            if np.linalg.norm(self.state.r_puck_vel[0:2]) < 1:
                picked_task = "prepare"

            return picked_task

        # Puck in agent's side, slow enough
        # self.state.w_puck_pos[0] < 0
        if np.linalg.norm(self.state.r_puck_vel[0:2]) < defend_vel_threshold:

            # Check if there is enough space to perform a hit
            puck_pos = self.state.w_puck_pos[:2]
            ee_pos = self.state.w_ee_pos[:2]
            # Beta computation
            beta = self.get_angle(
                self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos),
                self.world_to_puck(ee_pos, puck_pos)
            )

            if puck_pos[1] <= self.table_width / 2:
                gamma = self.get_angle(
                    self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos),
                    self.world_to_puck(self.goal_pos, puck_pos)
                )
                tot_angle = beta + gamma
            else:
                gamma = self.get_angle(
                    self.world_to_puck(self.goal_pos, puck_pos),
                    self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos)
                )
                tot_angle = beta - gamma

            # todo should the check for enough space become a function?
            enough_space = True
            tolerance = 1.5 * (self.mallet_radius + self.puck_radius)

            tolerance_coordinates = np.array([tolerance * np.cos(np.deg2rad(tot_angle)),
                                              tolerance * np.sin(np.deg2rad(tot_angle))])

            tolerance_coordinates = self.puck_to_world(tolerance_coordinates, puck_pos)

            if (tolerance_coordinates[0] <= -self.table_length / 2) or \
                    (tolerance_coordinates[1] >= self.table_width / 2 or tolerance_coordinates[
                        1] <= -self.table_width / 2):
                enough_space = False

            # if np.abs(self.table_width/2 - self.state.w_puck_pos[1]) < prepare_threshold or not enough_space:
            #    return "prepare"

            # else:
            #    return "hit"

            # if self.state.r_puck_vel[0] > 0:
            #    return "hit"

            if enough_space:
                return "hit"
            else:
                return "prepare"

        return "home"

    # todo remove, old
    def pick_task(self):
        """
        Description:
            This function will retrieve the correct task to perform among hit, defend and prepare

        Returns:
            the most appropriate task
        """

        if self.state.w_puck_pos[0] >= 0 * self.agent_id:
            # stay idle if the puck is in the enemy field
            return "home"

        if self.state.w_puck_pos[0] <= 0 and self.state.r_puck_vel[0] < 0:
            if np.linalg.norm(self.state.r_puck_vel[0:2]) > 0.2:
                return "defend"

        if np.linalg.norm(self.state.r_puck_vel[0:2]) > 1:  # the threshold is trainable
            # if self.state.r_puck_vel[0] < 0:
            #    return "defend"
            # else:
            if self.state.r_puck_vel[0] > 0:
                self.reset()
                self.phase = "wait"
                self.final = FINAL
                return "hit"
        else:
            puck_pos = self.state.w_puck_pos[:2]
            ee_pos = self.state.w_ee_pos[:2]
            # Beta computation
            beta = self.get_angle(
                self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos),
                self.world_to_puck(ee_pos, puck_pos)
            )

            if puck_pos[1] <= self.table_width / 2:
                gamma = self.get_angle(
                    self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos),
                    self.world_to_puck(self.goal_pos, puck_pos)
                )
                tot_angle = beta + gamma
            else:
                gamma = self.get_angle(
                    self.world_to_puck(self.goal_pos, puck_pos),
                    self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos)
                )
                tot_angle = beta - gamma

            # todo should the check for enough space become a function?
            enough_space = True
            tolerance = 1.5 * (self.mallet_radius + self.puck_radius)

            tolerance_coordinates = np.array([tolerance * np.cos(np.deg2rad(tot_angle)),
                                              tolerance * np.sin(np.deg2rad(tot_angle))])

            tolerance_coordinates = self.puck_to_world(tolerance_coordinates, puck_pos)

            if (tolerance_coordinates[0] <= -self.table_length / 2) or \
                    (tolerance_coordinates[1] >= self.table_width / 2 or tolerance_coordinates[
                        1] <= -self.table_width / 2):
                enough_space = False

            if enough_space:
                # self.phase = "wait"
                # self.final = FINAL
                return "hit"
            else:
                self.phase = "wait"
                return "prepare"

    def change_action_smoothly(self, previous_task, current_task, observation, steps=5):
        """
        Change the task smoothly to avoid too fast changes.

        final_action = (weight * current_action) + (1 - weight) * previous_action

        the weight will change step by step

        Parameters
        ----------
        previous_task : the task the agent was doing
        current_task : the task the agent will do
        observation : the observation from the environment
        steps : the number of steps used to perform the change

        Returns
        -------
        A smoothed action, weighted mean of the the ones coming from previous and current task
        """

        # Reset agents the first time the task is changed
        if self.task_change_counter == 1:
            self.defend_agent.reset()
            self.repel_defend_agent.reset()
            self.rule_based_agent.reset()
            self.baseline_agent.reset()
            print(f'From {self.previous_task} --> {self.task}')

            # Save the task changing in a file
            task_number = Tasks[self.task.upper()].value
            log = np.array([task_number, self.iteration_counter])
            self.task_change_log.append(log)
            # np.save('task_change_log.npy', self.task_change_log)

            ''' if os.path.exists(filename):
                with open(filename, 'r+') as f:
                    previous_logs = np.load(filename)
                    previous_logs = np.append(previous_logs, log)
                    np.save(filename, previous_logs)
                    f.close()
            else:
                with open(filename, 'wb') as f:
                    np.save(filename, log)

                    f.close()'''

        new_task_percentage = 100 / (steps * 100)
        new_task_percentage *= self.task_change_counter

        '''# Use a sigmoid step
        sigmoid_step = 9 / steps
        sigmoid_step *= self.task_change_counter
        sigmoid_step -= 4.5

        new_task_percentage = 1 / (1 + np.e**(-sigmoid_step))'''

        if previous_task == "defend":
            previous_action = self.defend_agent.draw_action(observation)
        elif previous_task == "repel":
            previous_action = self.repel_defend_agent.draw_action(observation)
        #elif previous_task == "home":
        #    previous_action = self.baseline_agent.draw_action(observation)
        else:
            self.rule_based_agent.set_task(self.previous_task)
            previous_action = self.rule_based_agent.draw_action(observation)

        if current_task == "defend":
            current_action = self.defend_agent.draw_action(observation)
        elif current_task == "repel":
            current_action = self.repel_defend_agent.draw_action(observation)
        #elif current_task == "home":
        #    current_action = self.baseline_agent.draw_action(observation)
        else:
            self.rule_based_agent.set_task(self.task)
            current_action = self.rule_based_agent.draw_action(observation)

        weighted_action = new_task_percentage * current_action + (1 - new_task_percentage) * previous_action

        if self.task_change_counter == steps:
            self.task_change_counter = 1
            self.changing_task = False
        else:
            self.task_change_counter += 1

        return weighted_action

    def check_enough_space(self):
        """
        Checks if there is enough space to perform a hit, in that case no prepare is needed

        Returns: True if there is enough space, False otherwise
        """

        puck_pos = self.state.w_puck_pos[:2]
        side_tolerance = 0.06
        enough_space = True

        # compute the offset wrt table borders
        x_tol = np.round(self.table_length / 2 - np.abs(puck_pos[0]), 5)
        y_tol = np.round(self.table_width / 2 - np.abs(puck_pos[1]), 5)

        tolerance = self.puck_radius + 2 * self.mallet_radius - side_tolerance

        if y_tol < tolerance or x_tol < tolerance:
            enough_space = False

        return enough_space

    def _apply_forward_kinematics(self, joint_pos):
        # Paramaters
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']

        # Apply FW kinematics
        ee_pos_robot_frame, rotation = forward_kinematics(
            mj_model=mj_model,
            mj_data=mj_data,
            q=joint_pos,
            link="ee"
        )
        return ee_pos_robot_frame

    def _apply_inverse_kinematics(self, ee_pos_robot_frame, intial_q=None):
        # Parameters
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']

        # Apply Inverse Kinemtics
        success, action_joints = inverse_kinematics(
            mj_model=mj_model,
            mj_data=mj_data,
            desired_position=ee_pos_robot_frame,
            desired_rotation=None,
            initial_q=intial_q,
            link="ee"
        )
        return action_joints

    def world_to_puck(self, point, puck_pos):
        return np.array((point - puck_pos))

    def puck_to_world(self, point, puck_pos):
        """
        Description:
            This function returns the the coordinates of "point" wrt "puck_pos"

        Args:
            point (numpy.array([*, *])): the point we want to translate
            puck_pos (numpy.array([*, *])): the new reference origin

        Returns:
            numpy.array([*, *]): new coordinates of point wrt puck_pos
        """
        world = - puck_pos  # world coordinates in the puck's space
        return np.array((point - world))

    def get_angle(self, v1, v2):
        """
        Description:
            Compute the angle counterclockwise, the order of the two vectors matters.

        Args:
            v1 (numpy.array([*, *])): first vector from which start to compute the angle
            v2 (numpy.array([*, *])): second vector to which compute the angle

        Returns:
            int: the angle between v1 and v2 in DEGREEs

        Example:
            v1 <- [1,0] and v2 <- [0,1] ==> 90
            v1 <- [0,1] and v2 <- [1,0] ==> 270
        """

        ang1 = np.arctan2(v1[1], v1[0])
        ang2 = np.arctan2(v2[1], v2[0])

        angle = ang2 - ang1

        if angle < 0:
            angle = 2 * np.pi + angle

        return np.rad2deg(angle)

