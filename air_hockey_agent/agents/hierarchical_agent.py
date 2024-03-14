"""
Agent working with multiple low level agents
High-level action: control the end-effector
"""
import copy
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
from air_hockey_agent.agents.agents import DefendAgent, RepelAgent
from air_hockey_agent.agents.agent_sb3 import AgentSB3
from baseline.baseline_agent.baseline_agent import BaselineAgent
from air_hockey_agent.agents.state_machine import StateMachine

# Macros
EE_HEIGHT = 0.0645  # apparently 0.0646 works better then 0.0645
WINDOW_SIZE = 200
PREV_BETA = 0
FINAL = 10
DEFEND_LINE = -0.8
DEFAULT_POS = np.array([-0.85995711, 0.0, 0.0645572])  # np.array([DEFEND_LINE, 0, EE_HEIGHT])
INIT_STATE = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
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
    HOME = 0
    HIT = 1
    DEFEND = 2
    REPEL = 3
    PREPARE = 4


# Class implementing the rule based policy
class HierarchicalAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, task: str = "home", **kwargs):
        # Superclass initialization
        super().__init__(env_info, agent_id, **kwargs)

        # INSTANTIATE AGENTS -------------------------------------------------------------------
        self.rule_based_agent = PolicyAgent(env_info, **kwargs)
        self.home_agent = AgentSB3(env_info, 'air_hockey_agent/agents/Agents/Home_Agent', acc_ratio=0.1, **kwargs)
        #self.hit_agent = AgentSB3(env_info, 'air_hockey_agent/agents/Agents/Hit_Agent', **kwargs)
        self.baseline_agent = BaselineAgent(env_info, **kwargs)
        self.repel_agent = RepelAgent(env_info, env_label="7dof-defend", **kwargs)
        self.defend_agent = DefendAgent(env_info, env_label="7dof-defend", **kwargs)

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
        self.step_counter = 0  # a generic counter for steps, used to make the same task for a given amount of steps
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
        self.joint_pos_log = []
        self.joint_vel_log = []


        # Parameters for PGPE
        # self.parameters = np.array([0.3, 4.0, 0.2])
        self.parameters = np.array([0.3, 0.2])
        # self.parameters = np.array([0.895, 0.795])
        self.tot_params = self.parameters.shape[0]

    def set_parameters(self, thetas):
        self.parameters = copy.deepcopy(thetas)

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
        self.task = 'home'
        self.previous_task = self.task

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
        self.home_agent.reset()
        #self.hit_agent.reset()
        self.defend_agent.reset()
        self.repel_agent.reset()
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

        # Reduce noise with kalman filter # todo (it is present also in the rule based agent and in agents.py)
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
        self.task = self.switcher()
        # self.task = self.simplified_pick_task()
        # self.task = self.select_task()
        self.task = self.state_machine.explicit_select_state(previous_state=self.previous_task, desired_next_state=self.task)

        if self.previous_task != self.task:
            print(f'{self.previous_task} --> {self.task}')

            # SAVE FILES
            # Save the task changing and timestamp
            task_number = Tasks[self.task.upper()].value
            log = np.array([task_number, self.iteration_counter])
            self.task_change_log.append(log)
            np.save('task_change_log.npy', self.task_change_log)

            # Save the joint positions
            # log_joint_pos = np.array(self.state.r_joint_pos)
            # self.joint_pos_log.append(log_joint_pos)
            # np.save('joint_pos_log.npy', self.joint_pos_log)

            # Save the joint velocities
            # log_joint_vel = np.array(self.state.r_joint_vel)
            # self.joint_vel_log.append(log_joint_vel)
            # np.save('joint_vel_log.npy', self.joint_vel_log)

            # Reset agents
            self.rule_based_agent.reset()
            self.home_agent.reset()
            #self.hit_agent.reset()
            self.defend_agent.reset()
            self.repel_agent.reset()
            self.baseline_agent.reset()

        # execute the action until completion
        if self.task == "defend":
            action = self.defend_agent.draw_action(observation)
            if self.defend_agent.has_hit or self.state.r_puck_vel[0] > 0 or np.linalg.norm(self.state.r_puck_vel[:2]) < 0.2:
                self.done = True
        elif self.task == "repel":
            action = self.repel_agent.draw_action(observation)
            if self.repel_agent.has_hit or self.state.r_puck_vel[0] > 0 or np.linalg.norm(self.state.r_puck_vel[:2]) < 0.2:
                self.done = True
        # elif self.task == "hit":
        #     action = self.hit_agent.draw_action(observation)
        #     if self.hit_agent.has_hit:
        #         self.done = True

        # AGENT THAT GOES HOME
        elif self.task == "home":
            action = self.home_agent.draw_action(observation)
            # Take the current radius
            radius = np.sqrt(
                (self.state.w_ee_pos[0] - DEFAULT_POS[0]) ** 2 + (self.state.w_ee_pos[1] - DEFAULT_POS[1]) ** 2
            )
            rounded_radius = round(radius - (self.mallet_radius + self.puck_radius), 5)

            if rounded_radius <= 5e-2:  # todo reduce to be more sensitive to home position
                self.done = True
        else:
            self.rule_based_agent.set_task(self.task)
            action = self.rule_based_agent.draw_action(observation)

            if self.task == "hit":
                self.done = self.rule_based_agent.hit_completed
            if self.task == "prepare":
                self.done = self.rule_based_agent.prepare_completed
            elif self.task == "home":
                self.done = self.rule_based_agent.home_completed

            if self.state.r_puck_pos[0] > 1.36:
                self.done = True
                self.task = "home"
        # else:
        #    self.rule_based_agent.set_task("home")
        #    action = self.rule_based_agent.draw_action(observation)
        #    self.done = self.rule_based_agent.hit_completed # bring self.done back to False after finishing the hit


        # UPDATE TIMER
        if np.sign(self.get_puck_pos(observation)[0] - 1.51) == self.prev_side:
            self.timer += self.env_info['dt']
            self.defend_agent.set_timer(self.timer)
            self.repel_agent.set_timer(self.timer)
        else:
            self.prev_side *= -1
            self.timer = 0

        # RESET INITIAL JOINT POSITION WHEN IN HOME
        '''if np.linalg.norm(self.state.w_ee_pos - DEFAULT_POS) <= 1e-2 and self.rule_based_agent.home_completed:
            # try:
            #    new_action = self.baseline_agent.draw_action(observation)
            # except ValueError:
            #    new_action = action
            # action = new_action

            # todo provare per step  (punto iniziale finale ed interpolazione)
            #  accelerazione fisse da dare una volta
            #  spostamento quadratico
            #  spostamento del quarto grado (o terzo) per avere accelerazione quadratica

            # current_pos = self.get_joint_pos(observation)
            #
            # vel_needed = (INIT_STATE - current_pos) / self.defend_agent.dt
            # current_vel = self.get_joint_vel(observation)
            #
            # acc_needed = (vel_needed - current_vel) / self.defend_agent.dt
            #
            # action = np.clip(acc_needed, self.defend_agent.low_joints_acc, self.defend_agent.high_joints_acc) * 0.05
            #
            # action[-1] = 0
            #
            # action = self.defend_agent.atacom_transformation.draw_action(observation, action)'''

        self.last_action = action
        return action


    def switcher(self):
        new_task = self.task
        if self.done:
            if self.task != "home":
                new_task = "home"
            else:
                radius = np.sqrt(
                    (self.state.w_ee_pos[0] - DEFAULT_POS[0]) ** 2 + (self.state.w_ee_pos[1] - DEFAULT_POS[1]) ** 2
                )

                rounded_radius = round(radius - (self.mallet_radius + self.puck_radius), 5)
                if rounded_radius <= 5e-2:  # todo reduce to be more sensitive to home position
                #if self.rule_based_agent.home_completed:
                    self.done = False
                    new_task = self.simplified_pick_task()
            # self.done = False
            # self.rule_based_agent.hit_completed = False
            # self.hit_agent.has_hit = False
            # self.rule_based_agent.prepare_completed = False
            # self.rule_based_agent.home_completed = False

        else:
            new_task = self.task

        '''
            # submitted switcher code (25-10)
            if self.done:
            new_task = self.simplified_pick_task()
            self.done = False
            self.rule_based_agent.hit_completed = False
            # self.hit_agent.has_hit = False
            # self.rule_based_agent.prepare_completed = False
            # self.rule_based_agent.home_completed = False
        else:
            new_task = self.task'''

        return new_task

    def simplified_pick_task(self):
        """
        Naïve version of the pick task but apparently te best one

        Returns
        -------
            The task from which retrieve the action
        """

        picked_task = self.previous_task
        defend_vel_threshold = self.parameters[0]  # 0.3
        # repel_vel_threshold = self.parameters[1]  # 4  # puck too fast, just send it back
        prepare_threshold = self.parameters[1]

        w_puck_pos = self.state.w_puck_pos
        r_puck_pos = self.state.r_puck_pos
        r_puck_vel = self.state.r_puck_vel

        puck_velocity = np.linalg.norm(r_puck_vel[:2])

        # Stay home if the puck is the enemy side, don't go over the unreachable side
        if r_puck_pos[0] >= 1.36:
            picked_task = "home"

        else:
            # Puck coming toward the agent
            if r_puck_vel[0] < 0:
                # fast puck
                # if puck_velocity > defend_vel_threshold:
                #     if puck_velocity > repel_vel_threshold:
                #         picked_task = "repel"
                #     else:
                #         picked_task = "repel"
                if puck_velocity > defend_vel_threshold:
                    picked_task = "repel"
                else:
                    enough_space = self.check_enough_space()
                    picked_task = "hit" if enough_space else "prepare"
            else:
                if puck_velocity < prepare_threshold:  # 0.2
                    enough_space = self.check_enough_space()
                    picked_task = "hit" if enough_space else "prepare"
                else:
                    picked_task = "home"

        return picked_task

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