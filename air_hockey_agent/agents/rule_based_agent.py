"""
Agent working via a Rule-based Policy
High-level action: control the end-effector
"""
# Libraries
import math
import time

import numpy as np
import osqp
import scipy
from scipy import sparse
from scipy.interpolate import CubicSpline

from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_agent.model import State
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot
from air_hockey_agent.agents.kalman_filter import PuckTracker
from air_hockey_agent.agents.optimizer import TrajectoryOptimizer

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
best_hit_train = np.array([0.01004139, 0.03003824, 0.00030698, 0.00010313])
BEST_PARAM = dict(
    hit=best_hit_train,
    defend=None,
    prepare=None
)


# Build Agent function
def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    :param env_info: The environment information
    :return: Either Agent ot Policy
    """
    if "hit" in env_info["env_name"]:
        return PolicyAgent(env_info, **kwargs, agent_id=1, task="hit")
    if "defend" in env_info["env_name"]:
        return PolicyAgent(env_info, **kwargs, agent_id=1, task="defend")
    if "prepare" in env_info["env_name"]:
        return PolicyAgent(env_info, **kwargs, agent_id=1, task="prepare")

    return PolicyAgent(env_info, **kwargs)


# Class implementing teh rule based policy
class PolicyAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, task: str = "hit", **kwargs):
        # Superclass initialization
        super().__init__(env_info, agent_id, **kwargs)

        self.optimizer = TrajectoryOptimizer(self.env_info)  # optimize joint position of each trajectory point
        # self.motion_law_predictor = MotionLaw(self.env_info["dt"])  # predict the puck position using motion low

        # Kalman filters
        self.puck_tracker = PuckTracker(self.env_info, agent_id)

        self.reset_filter = True

        # Task allocation
        self.task = task

        # Thetas
        assert self.task in ["hit", "defend", "prepare"], f"Illegal task: {self.task}"
        self.theta = BEST_PARAM['hit']

        # HIT: define initial values
        # angle between ee->puck and puck->goal lines
        self.previous_beta = PREV_BETA
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

        # Completed flags
        self.hit_completed = False
        self.prepare_completed = False

        # QUEUES
        # save last steps to reduce noise effect
        # self.puck_pos_queue = SlidingWindow(window_size=WINDOW_SIZE)
        # self.puck_vel_queue = SlidingWindow(window_size=WINDOW_SIZE)
        # self.joint_pos_queue = SlidingWindowJoints(window_size=WINDOW_SIZE)

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
        self.done = False
        self.has_hit = False
        self.can_hit = False  # Can hit considering enemy position
        self.keep_hitting = False  # Whether to hit again or not in the prepare
        self.state = State
        self.last_ds = 0
        self.phase = "wait"

    def reset(self):
        # Generic stuff
        self.previous_beta = PREV_BETA
        self.last_position = np.zeros(3)
        self.last_action = np.zeros((2, 7))
        self.last_r_puck_pos = np.zeros(3)
        self.last_w_puck_vel = np.zeros(3)
        self.q_anchor = None
        self.final = FINAL
        self.desired_from = None

        # Kalman filter
        self.reset_filter = True

        # Queues
        # self.puck_pos_queue = SlidingWindow(window_size=WINDOW_SIZE)
        # self.puck_vel_queue = SlidingWindow(window_size=WINDOW_SIZE)
        # self.joint_pos_queue = SlidingWindowJoints(window_size=WINDOW_SIZE)

        # Completed flags
        self.hit_completed = False
        self.prepare_completed = False

        # Episode parameters
        self.done = None
        self.has_hit = False
        self.can_hit = False
        self.keep_hitting = False
        self.time = dict(
            abs=0,
            adjustment=0,
            acceleration=0,
            smash=0
        )
        self.state = State
        self.last_ds = 0
        self.last_ee = None
        self.phase = "wait"

    def draw_action(self, observation):
        """
        Description:
            This function draws an action given the observation and the task.

        Args:
            observation (numpy.array([...])): see env

        Returns:
            numpy.array(2, 7): desired joint positions and velocities
        """
        action = self.act(state=observation)
        return action

    def act(self, state):
        """
        Description:
            This function computes the new action to perform to address the
            current task in the policy

        Args:
            state (list): a list of the state elements, please see
            envs/air_hockey/_get_state()

        Returns:
            numpy.array([*, *]): the new position the end effector should
            occupy in the next step
        """
        # Increase the time step
        # self.time += self.env_info["dt"]
        self.time["abs"] += 1

        # Get useful info from the state
        self.state.r_puck_pos = state[0:3]
        self.state.r_puck_vel = state[3:6]
        # self.state.r_joint_pos = state[6:13]
        # self.state.r_joint_vel = state[13:20]

        if self.time["abs"] == 1:
            self.state.r_joint_pos = state[6:13]
            self.state.r_joint_vel = state[13:20]

        if self.time["abs"] > 1:
            self.state.r_joint_pos = self.last_action[0]
            self.state.r_joint_vel = self.last_action[1]

        self.state.r_adv_ee_pos = state[-3:]  # adversary ee position in robot coordinates

        # Compute EE pos
        self.state.r_ee_pos = self._apply_forward_kinematics(joint_pos=self.state.r_joint_pos)

        # KALMAN FILTER ----------------------------------------------------------
        # reset filter with the initial positions
        if self.reset_filter:
            self.reset_filter = False
            self.puck_tracker.reset(self.state.r_puck_pos)
            # self.adv_ee_tracker.reset(self.state.r_adv_ee_pos)
            # self.ee_pos_tracker.reset(self.state.r_ee_pos)

        # predict next state, step does a single prediction step
        self.puck_tracker.step(self.state.r_puck_pos)
        # self.adv_ee_tracker.step(self.state.r_adv_ee_pos)
        # self.ee_pos_tracker.step(self.state.r_ee_pos)

        # Reduce noise with kalman filter
        # self.state.r_puck_pos = self.puck_tracker.state[[0, 1, 4]]  # state contains pos and velocity
        # self.state.r_puck_vel = self.puck_tracker.state[[2, 3, 5]]
        # self.state.r_adv_ee_pos = self.adv_ee_tracker.state[[0, 1, 4]]
        # self.state.r_ee_pos = self.ee_pos_tracker.state[0:3]

        # Convert in WORLD coordinates 2D
        self.state.w_puck_pos, _ = robot_to_world(base_frame=self.frame,
                                                  translation=self.state.r_puck_pos)
        self.state.w_ee_pos, _ = robot_to_world(base_frame=self.frame,
                                                translation=self.state.r_ee_pos)
        self.state.w_adv_ee_pos, _ = robot_to_world(base_frame=self.frame,
                                                    translation=self.state.r_adv_ee_pos)

        # Compute the action based on the task
        # Note: here all the actions must be in WORLD coordinates
        # self.theta = BEST_PARAM[self.task]

        # Select the most appropriate task
        # task choice is done by hierarchical agent
        # self.task = self.pick_task()
        # print(self.task)

        # task is set through a setter
        if self.task == "hit":
            action = self.hit_act_smooth_slow()
            # action = self.hit_act_circumference()
        elif self.task == "defend":
            action = self.defend_act()
        elif self.task == "prepare":
            action = self.prepare_act()
        elif self.task == "home":
            action, point_reached = self.smooth_act()
            if point_reached:
                action = self.return_act(self.state.w_ee_pos)
                self.hit_completed = False
                self.prepare_completed = False
            #action = self.return_act(self.state.w_ee_pos)
        else:
            action = None

        # Post-process the action
        # at this point we have a desired ee_x and ee_y
        # keep ee_z <- 0

        tolerance = 0.0065

        action_x = np.clip(action[0], -self.table_length / 2 + (self.mallet_radius + tolerance),
                           self.table_length / 2 - (self.mallet_radius + tolerance))
        action_y = np.clip(action[1], -self.table_width / 2 + (self.mallet_radius + tolerance),
                           self.table_width / 2 - (self.mallet_radius + tolerance))
        action_z = self.env_info['robot']['universal_height']  # EE_HEIGHT + tolerance
        action = np.array([action_x, action_y, action_z])

        # convert ee coordinates in the robot frame
        action, _ = world_to_robot(base_frame=self.frame, translation=action)

        # apply inverse kinematics
        final_action = np.zeros((2, 7))
        final_action[0] = self._apply_inverse_kinematics(
            ee_pos_robot_frame=action,
            intial_q=self.state.r_joint_pos
        )
        final_action[1] = (final_action[0] - self.state.r_joint_pos) / self.env_info["dt"]

        # Clip joint positions in the limits
        # joint_pos_limits = self.env_info['robot']['joint_pos_limit']
        # joint_vel_limits = self.env_info['robot']['joint_vel_limit']

        # new_final_action = np.zeros((2, 7))
        # new_final_action[0] = np.clip(final_action[0], joint_pos_limits[0], joint_pos_limits[1])
        # new_final_action[1] = np.clip(final_action[1], joint_vel_limits[0], joint_vel_limits[1])
        # new_final_action[2] = (final_action[1] - self.state.r_joint_vel)/self.env_info["dt"]

        # OPTIMIZE TRAJECTORY
        optimized_action = np.zeros((2, 7))

        action2 = np.concatenate((action, action))  # Create a two point trajectory

        '''
        if self.q_anchor is None:
            hit_dir_2d = self.goal_pos - self.state.r_puck_pos[:2]
            hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)

            hit_pos_2d = self.state.r_puck_pos[:2] - hit_dir_2d * (self.env_info['puck']['radius'] + self.env_info['mallet']['radius'])
            hit_vel_2d = hit_dir_2d * 1.0  # 1.0 is the hit_vel

            self.q_anchor = self.solve_anchor_pos_ik_null(hit_pos_2d, hit_vel_2d, self.state.r_joint_pos, 0.1)
        '''

        try:
            # success, opt_joint_pos = self.optimizer.optimize_trajectory(np.array([action]), self.state.r_joint_pos, self.state.r_joint_vel, final_action[0])
            success, joint_velocities = self.solve_aqp(action, self.state.r_joint_pos,
                                                       final_action[1])  # use dq_anchor to avoid robot shaking
            joint_pos = self.state.r_joint_pos + (self.state.r_joint_vel + joint_velocities) / 2 * self.env_info['dt']
        except ValueError:
            print('Value Error: no success into optimizing joint velocities')
            success = False
            joint_pos = final_action[0]
            joint_velocities = final_action[1]

        if success:
            optimized_action = np.vstack([joint_pos, joint_velocities])
            # optimized_action[0] = opt_joint_pos
            # optimized_action[1] = final_action[1]
            # opt_joint_pos = np.concatenate((self.state.r_joint_pos, final_action[0]))
            # optimized_action = np.vstack([optimized_action, self.cubic_spline_interpolation(opt_joint_pos)])
        else:
            optimized_action[0] = final_action[0]
            optimized_action[1] = final_action[1]

        # Clip joint positions in the limits
        joint_pos_limits = self.env_info['robot']['joint_pos_limit']
        joint_vel_limits = self.env_info['robot']['joint_vel_limit']

        new_final_action = np.zeros((2, 7))
        new_final_action[0] = np.clip(optimized_action[0], joint_pos_limits[0], joint_pos_limits[1])
        new_final_action[1] = np.clip(optimized_action[1], joint_vel_limits[0], joint_vel_limits[1])
        # new_final_action[2] = (final_action[1] - self.state.r_joint_vel)/self.env_info["dt"]

        self.last_action = new_final_action
        self.last_r_puck_pos = self.state.r_puck_pos

        return new_final_action

    def set_task(self, task=None):
        """
        Set the self.task parameter. Used to allow hierarchical_agent control over
        the rule_based agent
        """
        if task is not None:
            assert task in ["hit", "defend", "prepare", "home"]
        self.task = task

    def pick_task(self):
        """
        Description:
            This function will retrieve the correct task to perform among hit, defend and prepare

        Returns:
            the most appropriate task
        """

        if self.state.w_puck_pos[0] >= 0 * self.agent_id:
            # always defend is the puck is in the enemy field
            return "defend"

        if np.linalg.norm(self.state.r_puck_vel[0:2]) > 1:  # the threshold is trainable
            if self.state.r_puck_vel[0] < 0:
                return "defend"
            else:
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
                    (tolerance_coordinates[1] >= self.table_width / 2 or tolerance_coordinates[1] <= -self.table_width / 2):
                enough_space = False

            if enough_space:
                # self.phase = "wait"
                # self.final = FINAL
                return "hit"
            else:
                self.phase = "wait"
                return "prepare"

    def solve_anchor_pos_ik_null(self, hit_pos_2d, hit_dir_2d, q_0, solve_max_time):
        hit_pos = np.concatenate([hit_pos_2d, [self.env_info['robot']["ee_desired_height"]]])
        hit_dir = np.concatenate([hit_dir_2d, [0]])
        success, q_star = self.optimizer.solve_hit_config_ik_null(hit_pos, hit_dir, q_0, max_time=solve_max_time)
        return q_star

    def cubic_spline_interpolation(self, joint_pos_traj):
        joint_pos_traj = np.array(joint_pos_traj)
        t = np.linspace(1, joint_pos_traj.shape[0], joint_pos_traj.shape[0]) * 0.02

        f = CubicSpline(t, joint_pos_traj, axis=0)
        df = f.derivative(1)
        return np.stack([f(t), df(t)]).swapaxes(0, 1)

    def solve_aqp(self, x_des, q_cur, dq_anchor):
        robot_model = self.robot_model
        robot_data = self.robot_data
        joint_vel_limits = self.env_info['robot']['joint_vel_limit']
        joint_pos_limits = self.env_info['robot']['joint_pos_limit']
        dt = self.env_info["dt"]
        n_joints = self.env_info["robot"]["n_joints"]

        anchor_weights = np.array([10., 1., 10., 1., 10., 10., 1.])

        x_cur = forward_kinematics(robot_model, robot_data, q_cur)[0]
        jac = jacobian(robot_model, robot_data, q_cur)[:3, :n_joints]
        N_J = scipy.linalg.null_space(jac)
        b = np.linalg.lstsq(jac, (x_des - x_cur) / dt, rcond=None)[0]

        P = (N_J.T @ np.diag(anchor_weights) @ N_J) / 2
        q = (b - dq_anchor).T @ np.diag(anchor_weights) @ N_J
        A = N_J.copy()
        u = np.minimum(joint_vel_limits[1] * 0.92,
                       (joint_pos_limits[1] * 0.92 - q_cur) / dt) - b
        l = np.maximum(joint_vel_limits[0] * 0.92,
                       (joint_pos_limits[0] * 0.92 - q_cur) / dt) - b

        if np.array(u < l).any():
            return False, b

        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=l, u=u, verbose=False, polish=False)

        result = solver.solve()
        if result.info.status == 'solved':
            return True, N_J @ result.x + b
        else:
            return False, b

    def hit_act(self):
        """
        Description:
            this function computes the new action to perform to perform a "hit"

        Returns:
            numpy.array([*, *]): the new position the end effector should
            occupy in the next step
        """

        # INITIALIZATION ------------------------------------------------------
        # Get useful info from the state
        puck_pos = self.state.w_puck_pos[:2]
        ee_pos = self.state.w_ee_pos[:2]

        # Initialization of d_beta and ds
        d_beta = 0
        ds = 0

        # Beta computation
        beta = self.get_angle(
            self.world_to_puck(self.goal_pos, puck_pos),
            self.world_to_puck(ee_pos, puck_pos)
        )

        # Gamma computation
        if puck_pos[1] <= self.table_width / 2:
            gamma = self.get_angle(
                self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos),
                self.world_to_puck(self.goal_pos, puck_pos)
            )
        else:
            gamma = self.get_angle(
                self.world_to_puck(self.goal_pos, puck_pos),
                self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos)
            )

        # Convert the ee position and take the current radius
        ee_pos_puck = self.world_to_puck(ee_pos, puck_pos)
        radius = np.sqrt(
            (ee_pos[0] - puck_pos[0]) ** 2 + (ee_pos[1] - puck_pos[1]) ** 2
        )

        rounded_radius = round(radius - (self.mallet_radius + self.puck_radius), 5)

        # correction term
        correction = np.abs(180 - beta) if puck_pos[1] <= self.table_width / 2 else np.abs(beta - 180)

        # PHASES --------------------------------------------------------------
        # WAIT: wait for the puck to move slowly
        if self.phase == "wait" and (np.linalg.norm(self.state.r_puck_vel) < 3 or self.state.r_puck_vel[0] >= 0):
            self.phase = "adjustment"

        # ADJUSTMENT: adjust the beta angle and go to the right trajectory
        if self.phase == "adjustment":
            self.time[self.phase] += 1
            if correction <= 2.5:
                self.phase = "acceleration"
            else:
                d_beta = (0.01 + 0.03 * self.time[self.phase] * self.env_info["dt"]) * correction
                ds = 3e-4
                self.last_ds = ds

        # ACCELERATION: the ee is in the correct line, accelerate
        if self.phase == "acceleration":
            self.time[self.phase] += 1

            # Check if can hit considering the adversary position
            adv_ee_pos = self.state.w_adv_ee_pos
            lower_bound_goal = - self.env_info["table"]["goal_width"] / 2
            upper_bound_goal = self.env_info["table"]["goal_width"] / 2

            # check if adv_ee moved in around the bounds of the goal
            # todo change this control, it was custom for the bernoulli lemniscate curve
            #  it might be removed
            if (adv_ee_pos[1] <= 0 and np.abs(adv_ee_pos[1] - lower_bound_goal) < 0.08) or \
                    (adv_ee_pos[1] >= 0 and np.abs(adv_ee_pos[1] - upper_bound_goal) < 0.08):
                self.can_hit = True

            self.can_hit = True

            # (np.abs(radius - self.mallet_radius - self.puck_radius) <= 5e-3) and (self.state.w_puck_pos[0] > -0.6 or self.state.w_puck_vel[0] >= 0) violates less constraints but has a lower success rate
            if rounded_radius <= 1e-2 and (self.state.w_puck_pos[0] > - 0.48):  # and self.state.w_puck_vel[0] > 0) and not self.has_hit:
                self.phase = "smash"
                self.has_hit = True

            elif self.can_hit:
                # update ds considering how far it is from the left short-side
                d_beta = correction / 2

                ds = (self.last_ds + 1e-4 * self.env_info["dt"] * self.time[self.phase] / (radius + self.mallet_radius))
                # ds *= (1/self.state.r_puck_pos[0] - 1e-3)
                # ds *= 0.9 / (self.state.r_puck_pos[0] / (self.table_length / 2))
                self.last_ds = ds

            else:
                ds = 1e-3

        # SMASH: perform a smash right after the "has_hit" computation
        if self.phase == "smash":
            self.time[self.phase] += 1
            if self.final == 0:
                self.reset()
                # return self.last_position
                return self.return_act(ee_pos)
            else:
                self.final -= 1
                d_beta = correction
                # ds = self.last_ds + 0.01 * (DES_ACC - self.last_ds) / (FINAL - self.final)
                ds = self.last_ds / ((FINAL - self.final) * 2)
                # ds = self.last_ds * 0.95
                self.last_ds = ds

        # NEXT POINT COMPUTATION ----------------------------------------------
        if puck_pos[1] <= self.table_width / 2:
            # Puck is in the bottom part
            # Check if d_beta needs to be inverted
            if ee_pos_puck[1] < radius * np.sin(np.deg2rad(gamma + 180)):
                d_beta = -d_beta

            # Compute the action
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta + d_beta + gamma))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta + d_beta + gamma))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)
            self.last_position = action
        else:
            # Puck is in the top part
            # Check if d_beta needs to be inverted
            if ee_pos_puck[1] > radius * np.sin(np.deg2rad(180 - gamma)):
                d_beta = -d_beta

            # Compute the action
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta - d_beta - gamma))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta - d_beta - gamma))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)
            self.last_position = action

        return action

    def hit_act_smooth_slow(self):
        """
        Description:
            this function computes the new action to perform to perform a "hit"

        Returns:
            numpy.array([*, *]): the new position the end effector should
            occupy in the next step
        """
        # INITIALIZATION ------------------------------------------------------
        # Get useful info from the state
        puck_pos = self.state.w_puck_pos[:2]
        ee_pos = self.state.w_ee_pos[:2]

        # Initialization of d_beta and ds
        d_beta = 0
        ds = 0

        # If action is already completed don't do anything
        if self.hit_completed:
            return self.last_position

        # Beta computation
        beta = self.get_angle(
            self.world_to_puck(self.goal_pos, puck_pos),
            self.world_to_puck(ee_pos, puck_pos)
        )

        # Gamma computation
        if puck_pos[1] <= self.table_width / 2:
            gamma = self.get_angle(
                self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos),
                self.world_to_puck(self.goal_pos, puck_pos)
            )
        else:
            gamma = self.get_angle(
                self.world_to_puck(self.goal_pos, puck_pos),
                self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos)
            )

        # Convert the ee position and take the current radius
        ee_pos_puck = self.world_to_puck(ee_pos, puck_pos)
        radius = np.sqrt(
            (ee_pos[0] - puck_pos[0]) ** 2 + (ee_pos[1] - puck_pos[1]) ** 2
        )

        rounded_radius = round(radius - (self.mallet_radius + self.puck_radius), 5)

        # correction term
        correction = np.abs(180 - beta) if puck_pos[1] <= self.table_width / 2 else np.abs(beta - 180)

        # PHASES --------------------------------------------------------------
        # WAIT: wait for the puck to move slowly
        if self.phase == "wait" and (np.linalg.norm(self.state.r_puck_vel) < 3 or self.state.r_puck_vel[0] >= 0):
            self.phase = "adjustment"

        # ADJUSTMENT: adjust the beta angle and go to the right trajectory
        if self.phase == "adjustment":
            self.time[self.phase] += 1
            if correction <= 2.5:
                self.phase = "acceleration"
            else:
                # d_beta = (0.01 + 0.03 * self.time[self.phase] * self.env_info["dt"]) * correction
                # ds = 3e-4
                # 0.01
                d_beta = (0.05 + 0.03 * self.time[self.phase] * self.env_info["dt"]) * correction
                ds = 3e-4

                self.last_ds = ds

        # ACCELERATION: the ee is in the correct line, accelerate
        if self.phase == "acceleration":
            self.time[self.phase] += 1
            self.last_w_puck_pos = puck_pos  # it will be used in the slow_down phase

            # Check if can hit considering the adversary position
            adv_ee_pos = self.state.w_adv_ee_pos
            lower_bound_goal = - self.env_info["table"]["goal_width"] / 2
            upper_bound_goal = self.env_info["table"]["goal_width"] / 2

            # check if adv_ee moved in around the bounds of the goal
            if (adv_ee_pos[1] <= 0 and np.abs(adv_ee_pos[1] - lower_bound_goal) < 0.08) or \
                    (adv_ee_pos[1] >= 0 and np.abs(adv_ee_pos[1] - upper_bound_goal) < 0.08):
                self.can_hit = True

            self.can_hit = True

            # (np.abs(radius - self.mallet_radius - self.puck_radius) <= 5e-3) and (self.state.w_puck_pos[0] > -0.6 or self.state.w_puck_vel[0] >= 0) violates less constraints but has a lower success rate
            if rounded_radius <= 1e-2 and (puck_pos[0] > - 0.5):  # and self.state.w_puck_vel[0] > 0) and not self.has_hit:
                self.phase = "slow_down"
                self.has_hit = True

            elif self.can_hit:
                # update ds considering how far it is from the left short-side
                d_beta = correction / 2

                ds = (self.last_ds + 1e-4 * self.env_info["dt"] * self.time[self.phase] / (radius + self.mallet_radius))

                # ds *= (1/self.state.r_puck_pos[0] - 1e-3)
                # ds *= 0.9 / (self.state.r_puck_pos[0] / (self.table_length / 2))  # FIXME problem if puck_vel[0] is close to 0
                self.last_ds = ds

            else:
                ds = 1e-3

        # SMASH: perform a smash right after the "has_hit" computation
        if self.phase == "smash":
            self.time[self.phase] += 1
            self.last_w_puck_pos = puck_pos

            if self.final == 0:
                self.phase = "slow_down"
                # return self.last_position
            else:
                self.final -= 1
                d_beta = correction
                # ds = self.last_ds + 0.01 * (DES_ACC - self.last_ds) / (FINAL - self.final)
                ds = self.last_ds / ((FINAL - self.final) * 2)
                # ds = self.last_ds * 0.95
                self.last_ds = ds

        # SLOW_DOWN: move on a curved trajectory to slow down after a hit
        if self.phase == "slow_down":
            if self.state.w_puck_pos[1] < 0:
                stop_point = np.array([self.last_w_puck_pos[0] - 0.1, self.last_w_puck_pos[1] + 0.1])
                # stop_point, _ = world_to_robot(base_frame=self.frame, translation=stop_point)
            else:
                stop_point = np.array([self.last_w_puck_pos[0] - 0.1, self.last_w_puck_pos[1] - 0.1])
                # stop_point, _ = world_to_robot(base_frame=self.frame, translation=stop_point)

            stop_point = DEFAULT_POS[:2]
            action, point_reached = self.smooth_act(target_point=stop_point, step_size=self.last_ds / 2)

            if not point_reached:
                return action
            else:
                # self.phase = "return_home"
                self.hit_completed = True

        '''# RETURN HOME: bring the end effector back in the default position
        if self.phase == "return_home":
            if np.linalg.norm(self.state.w_puck_vel[:2]) < 0.1:
                self.has_hit = False
                self.phase = "wait"
            else:
                return self.return_act(ee_pos, step_size=self.last_ds)'''

        # NEXT POINT COMPUTATION ----------------------------------------------
        if puck_pos[1] <= self.table_width / 2:
            # Puck is in the bottom part
            # Check if d_beta needs to be inverted
            if ee_pos_puck[1] < radius * np.sin(np.deg2rad(gamma + 180)):
                d_beta = -d_beta

            # Compute the action
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta + d_beta + gamma))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta + d_beta + gamma))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)
            self.last_position = action
        else:
            # Puck is in the top part
            # Check if d_beta needs to be inverted
            if ee_pos_puck[1] > radius * np.sin(np.deg2rad(180 - gamma)):
                d_beta = -d_beta

            # Compute the action
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta - d_beta - gamma))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta - d_beta - gamma))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)
            self.last_position = action

        return action

    def hit_act_circumference(self):
        """
       Description:
           this function computes the new action to perform to perform a
           "prepare"

       Returns:
           numpy.array([*, *]): the new position the end effector should
           occupy in the next step
       """

        # INITIALIZATION ------------------------------------------------------
        # Retrieve useful info from the state
        puck_pos = self.state.w_puck_pos[:2]
        ee_pos = self.state.w_ee_pos[:2]
        puck_vel = self.state.w_puck_vel[:2]

        # Initialize d_beta and ds
        d_beta = 0
        ds = 0

        # Beta computation
        beta = self.get_angle(
            self.world_to_puck(self.goal_pos, puck_pos),
            self.world_to_puck(ee_pos, puck_pos)
        )

        # CHECK ENOUGH SPACE ------------------------------------------------------------------
        # che if there is enough space to perform a hit, in that case no prepare is needed

        '''
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
    
        enough_space = True
        tolerance = 1.5 * (self.mallet_radius + self.puck_radius)
    
        tolerance_coordinates = np.array([tolerance * np.cos(np.deg2rad(tot_angle)),
                                          tolerance * np.sin(np.deg2rad(tot_angle))])
    
        tolerance_coordinates = self.puck_to_world(tolerance_coordinates, puck_pos)
    
        if (tolerance_coordinates[0] <= -self.table_length / 2) or \
                (tolerance_coordinates[1] >= self.table_width / 2 or tolerance_coordinates[1] <= -self.table_width / 2):
            enough_space = False
        '''
        '''
        enough_space = True
    
        # compute the offset wrt table borders
        x_tol = np.round(self.table_length / 2 - np.abs(puck_pos[0]), 5)
        y_tol = np.round(self.table_width / 2 - np.abs(puck_pos[1]), 5)
    
        tolerance = self.puck_radius + 2 * self.mallet_radius
    
        if y_tol < tolerance or x_tol < tolerance:
            enough_space = False
        '''
        enough_space = self.check_enough_space()

        if enough_space:
            self.keep_hitting = False
        else:
            self.keep_hitting = True

        if self.keep_hitting and self.has_hit:
            if np.linalg.norm(ee_pos - DEFAULT_POS[:2]) <= 0.2:
                self.has_hit = False
                # self.keep_hitting = False
                self.phase = "wait"
            pass

        # Take the current radius
        radius = np.sqrt(
            (ee_pos[0] - puck_pos[0]) ** 2 + (ee_pos[1] - puck_pos[1]) ** 2
        )

        rounded_radius = round(radius - (self.mallet_radius + self.puck_radius), 5)

        correction = (beta - 90) if puck_pos[1] <= 0 else (270 - beta)

        # WAIT ------------------------------------------------------------------
        if self.phase == "wait":
            self.phase = "adjustment"

        # ADJUSTMENT ------------------------------------------------------------------
        if self.phase == "adjustment":
            self.time[self.phase] += 1

            if correction <= 1e-2:
                correction = 0
                self.phase = "acceleration"

            d_beta = 0.01 * self.time[self.phase] + correction * 0.03  # fixme clip beta if beta - d_beta < 90?
            ds = 5e-3

            self.last_ds = ds

        if rounded_radius <= 0 and not self.has_hit:
            self.has_hit = True
            self.time["adjustment"] = 0

        # ACCELERATION ------------------------------------------------------------------
        if self.phase == "acceleration":
            d_beta = correction
            ds = 1e-2
            self.last_ds = ds

        '''if self.has_hit and enough_space:
            if np.linalg.norm(puck_vel) > 0.5:
                self.keep_hitting = True
                self.has_hit = False
            else:
                if enough_space:
                    self.keep_hitting = False
                else:
                    self.has_hit = False
                    self.keep_hitting = True
            return self.return_act(ee_pos, step_size=self.last_ds)
        '''

        if self.has_hit:
            # print('return')
            return self.return_act(ee_pos=ee_pos, step_size=self.last_ds * 2)

        '''
        # correction term
        correction = np.abs(beta - 90) if (ee_pos[1] >= puck_pos[1]) else np.abs(270 - beta)
    
        if puck_pos[0] > -0.4:
            print('hit from behind')
            correction = np.abs(beta - 360)
    
        #if puck_vel[0] < -0.01:
        #    print('hit from front')
        #    correction = np.abs(beta - 180)
    
        if self.phase == "wait":
            time.sleep(0.5)
            self.phase = "adjustment"
    
        if self.phase == "adjustment":
    
            print(correction)
            if correction <= 3:
                self.phase = "acceleration"
            else:
                d_beta = 8e-3 * self.time["abs"] + (5e-3 * correction)
                ds = 5e-3
    
        if self.phase == "acceleration":
            if np.abs(radius - self.puck_radius - self.mallet_radius) <= 1e-3:
                self.phase = "return"
            else:
                d_beta = 8e-3 * self.time["abs"]
                ds = 5e-3 + 0.01 * (puck_pos[1] / (self.table_width/2))
    
        if self.phase == "return":
            if self.final == 0:
                enough_space = np.sqrt(
                    (puck_pos[0] - self.puck_radius + 0.5 * self.table_length) ** 2 + \
                    (puck_pos[1] - self.puck_radius + ((0.5 * self.table_length - puck_pos[1]) / (
                        -puck_pos[0])) * 0.5 * self.table_length) ** 2
                )
    
                if enough_space > (self.mallet_radius + self.puck_radius):
                    return self.last_position
                else:
                    self.phase = "wait"
            else:
                d_beta = 8e-3 * self.time["abs"]
                ds = -5e-3
    
                self.final -= 1
        '''

        ee_pos_puck = self.world_to_puck(ee_pos, puck_pos)
        # NEXT POINT COMPUTATION ----------------------------------------------
        if puck_pos[1] <= ee_pos[1]:
            # Puck is in the bottom part
            # if ee_pos_puck[1] < radius * np.sin(np.deg2rad(180)):
            #    d_beta = -d_beta

            # Compute the action
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta - d_beta))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta - d_beta))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)
        else:
            # Puck is in the top part
            # Compute the action
            # if ee_pos_puck[1] > radius * np.sin(np.deg2rad(180)):
            #    d_beta = -d_beta
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta + d_beta))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta + d_beta))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)

        self.last_position = action
        return action

    def defend_act(self):
        """
        Description:
            this function computes the new action to perform a "defend"

        Returns:
              numpy.array([*, *]) the new position the end effector should
              occupy in the next step
        """

        # INITIALIZATION ------------------------------------------------------
        # get useful info from the state
        puck_pos = self.state.w_puck_pos[:2]
        puck_vel = self.state.r_puck_vel[:2]
        ee_pos = self.state.w_ee_pos[:2]

        # Convert the ee_pos and take the current radius
        radius = np.sqrt(
            (ee_pos[0] - puck_pos[0]) ** 2 + (ee_pos[1] - puck_pos[1]) ** 2
        )

        # Check if there was a hit
        if np.abs(radius - self.mallet_radius - self.puck_radius) <= 5e-3:
            self.has_hit = True

        if puck_pos[1] >= ee_pos[1]:
            self.desired_from = "bottom"
        else:
            self.desired_from = "top"

        # Define the targets
        if self.desired_from == "top":
            y_target = puck_pos[1] + 0.2 * (self.puck_radius + self.mallet_radius) * np.linalg.norm(
                puck_vel) / 2
        else:
            y_target = puck_pos[1] - 0.2 * (self.puck_radius + self.mallet_radius) * np.linalg.norm(
                puck_vel) / 2

        x_target = self.defend_line

        # Compute ds
        if not self.has_hit:
            ds_x = (0.1 + 0.07 * np.linalg.norm(puck_vel)) * (x_target - ee_pos[0])
            ds_y = (0.3 + 0.09 * np.linalg.norm(puck_vel)) * (y_target - ee_pos[1])

            # Compute the action
            action = np.array([ee_pos[0] + ds_x, ee_pos[1] + ds_y])
            self.last_position = action
        else:
            action = self.last_position

        return action

    def prepare_act(self):
        """
        Description:
            this function computes the new action to perform to perform a
            "prepare"

        Returns:
            numpy.array([*, *]): the new position the end effector should
            occupy in the next step
        """

        # INITIALIZATION ------------------------------------------------------
        # Retrieve useful info from the state
        puck_pos = self.state.w_puck_pos[:2]
        ee_pos = self.state.w_ee_pos[:2]
        puck_vel = self.state.w_puck_vel[:2]

        # Initialize d_beta and ds
        d_beta = 0
        ds = 0

        # If action is already completed don't do anything
        if self.prepare_completed:
            return self.last_position

        # Beta computation
        beta = self.get_angle(
            self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos),
            self.world_to_puck(ee_pos, puck_pos)
        )

        # CHECK ENOUGH SPACE ------------------------------------------------------------------
        # che if there is enough space to perform a hit, in that case no prepare is needed

        '''
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

        enough_space = True
        tolerance = 1.5 * (self.mallet_radius + self.puck_radius)

        tolerance_coordinates = np.array([tolerance * np.cos(np.deg2rad(tot_angle)),
                                          tolerance * np.sin(np.deg2rad(tot_angle))])

        tolerance_coordinates = self.puck_to_world(tolerance_coordinates, puck_pos)

        if (tolerance_coordinates[0] <= -self.table_length / 2) or \
                (tolerance_coordinates[1] >= self.table_width / 2 or tolerance_coordinates[1] <= -self.table_width / 2):
            enough_space = False
        '''
        '''
        enough_space = True

        # compute the offset wrt table borders
        x_tol = np.round(self.table_length / 2 - np.abs(puck_pos[0]), 5)
        y_tol = np.round(self.table_width / 2 - np.abs(puck_pos[1]), 5)

        tolerance = self.puck_radius + 2 * self.mallet_radius

        if y_tol < tolerance or x_tol < tolerance:
            enough_space = False
        '''
        enough_space = self.check_enough_space()

        # print(f'has hit: {self.has_hit}\nkeep hitting: {self.keep_hitting}\ncompleted: {self.prepare_completed}\n')

        if enough_space:
            self.keep_hitting = False
        else:
            self.keep_hitting = True

        if self.keep_hitting and self.has_hit:
            if np.linalg.norm(ee_pos - DEFAULT_POS[:2]) <= 0.2:
                self.has_hit = False
                # self.keep_hitting = False
                self.phase = "wait"

        # Take the current radius
        radius = np.sqrt(
            (ee_pos[0] - puck_pos[0]) ** 2 + (ee_pos[1] - puck_pos[1]) ** 2
        )

        rounded_radius = round(radius - (self.mallet_radius + self.puck_radius), 5)

        correction = (beta - 90) if puck_pos[1] <= 0 else (270 - beta)

        if self.keep_hitting:
            # WAIT ------------------------------------------------------------------
            if self.phase == "wait":
                self.phase = "adjustment"

            # ADJUSTMENT ------------------------------------------------------------------
            if self.phase == "adjustment":
                self.time[self.phase] += 1

                if correction <= 1e-2:
                    correction = 0
                    self.phase = "acceleration"

                d_beta = 0.01 * self.time[self.phase] + correction * 0.03  # fixme clip beta if beta - d_beta < 90?
                ds = 5e-3

                self.last_ds = ds

            if rounded_radius <= 0 and not self.has_hit:
                self.has_hit = True
                self.time["adjustment"] = 0

            # ACCELERATION ------------------------------------------------------------------
            if self.phase == "acceleration":
                d_beta = correction
                ds = 1e-2
                self.last_ds = ds

        '''if self.has_hit and enough_space:
            if np.linalg.norm(puck_vel) > 0.5:
                self.keep_hitting = True
                self.has_hit = False
            else:
                if enough_space:
                    self.keep_hitting = False
                else:
                    self.has_hit = False
                    self.keep_hitting = True
            return self.return_act(ee_pos, step_size=self.last_ds)
        '''

        if self.has_hit:
            action, point_reached = self.smooth_act(step_size=self.last_ds)

            if not point_reached:
                return action
            else:
                self.prepare_completed = True
                return self.last_position

        if not self.has_hit and not self.keep_hitting:
            action, point_reached = self.smooth_act(step_size=self.last_ds)

            if not point_reached:
                return action
            else:
                self.prepare_completed = True
                return self.last_position

        '''
        # correction term
        correction = np.abs(beta - 90) if (ee_pos[1] >= puck_pos[1]) else np.abs(270 - beta)

        if puck_pos[0] > -0.4:
            print('hit from behind')
            correction = np.abs(beta - 360)

        #if puck_vel[0] < -0.01:
        #    print('hit from front')
        #    correction = np.abs(beta - 180)

        if self.phase == "wait":
            time.sleep(0.5)
            self.phase = "adjustment"

        if self.phase == "adjustment":

            print(correction)
            if correction <= 3:
                self.phase = "acceleration"
            else:
                d_beta = 8e-3 * self.time["abs"] + (5e-3 * correction)
                ds = 5e-3

        if self.phase == "acceleration":
            if np.abs(radius - self.puck_radius - self.mallet_radius) <= 1e-3:
                self.phase = "return"
            else:
                d_beta = 8e-3 * self.time["abs"]
                ds = 5e-3 + 0.01 * (puck_pos[1] / (self.table_width/2))

        if self.phase == "return":
            if self.final == 0:
                enough_space = np.sqrt(
                    (puck_pos[0] - self.puck_radius + 0.5 * self.table_length) ** 2 + \
                    (puck_pos[1] - self.puck_radius + ((0.5 * self.table_length - puck_pos[1]) / (
                        -puck_pos[0])) * 0.5 * self.table_length) ** 2
                )

                if enough_space > (self.mallet_radius + self.puck_radius):
                    return self.last_position
                else:
                    self.phase = "wait"
            else:
                d_beta = 8e-3 * self.time["abs"]
                ds = -5e-3

                self.final -= 1
        '''

        # NEXT POINT COMPUTATION ----------------------------------------------
        if puck_pos[1] <= ee_pos[1]:
            # Puck is in the bottom part
            # if ee_pos_puck[1] < radius * np.sin(np.deg2rad(180)):
            #    d_beta = -d_beta

            # Compute the action
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta - d_beta))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta - d_beta))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)
        else:
            # Puck is in the top part
            # Compute the action
            # if ee_pos_puck[1] > radius * np.sin(np.deg2rad(180)):
            #    d_beta = -d_beta
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta + d_beta))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta + d_beta))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)

        self.last_position = action
        return action

    def original_prepare_act(self):
        """
        Description:
            this function computes the new action to perform a "prepare"

        Returns:
            numpy.array([*, *]): the new position the end effector should
            occupy in the next step
        """

        # INITIALIZATION ------------------------------------------------------
        # Retrieve useful info from the state
        puck_pos = self.state.w_puck_pos[:2]
        ee_pos = self.state.w_ee_pos[:2]
        puck_vel = self.state.w_puck_vel[:2]

        # Initialize d_beta and ds
        d_beta = 0
        ds = 0

        # Beta computation
        beta = self.get_angle(
            self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos),
            self.world_to_puck(ee_pos, puck_pos)
        )

        enough_space = self.check_enough_space()

        verbose = False
        if verbose:
            print(self.phase)
            print(f'enough space: {enough_space}')
            print(f'has_hit: {self.has_hit}')
            print(f'keep_hitting: {self.keep_hitting}\n')

        if enough_space:
            self.keep_hitting = False
        else:
            self.keep_hitting = True

        if self.keep_hitting and self.has_hit:
            if np.linalg.norm(ee_pos - DEFAULT_POS[:2]) <= 0.2:
                self.has_hit = False
                # self.keep_hitting = False
                self.phase = "wait"
            pass

        # Take the current radius
        radius = np.sqrt(
            (ee_pos[0] - puck_pos[0]) ** 2 + (ee_pos[1] - puck_pos[1]) ** 2
        )

        rounded_radius = round(radius - (self.mallet_radius + self.puck_radius), 5)

        correction = (beta - 90) if puck_pos[1] <= 0 else (270 - beta)

        print(correction)

        # if correction <= 1e-2:
        #    correction = 0

        if self.phase == "wait":
            # time.sleep(0.2)
            self.phase = "adjustment"

        if self.phase == "adjustment":
            self.time[self.phase] += 1

            if correction <= 1e-2:
                # correction = 0
                self.phase = "acceleration"

            d_beta = 0.01 * self.time[self.phase] + correction * 0.03
            # d_beta = 0.01 + correction * 0.03
            ds = 5e-3
            self.last_ds = ds

        if rounded_radius <= 0 and not self.has_hit:
            self.has_hit = True
            self.time["adjustment"] = 0

        # ACCELERATION ------------------------------------------------------------------
        if self.phase == "acceleration":
            d_beta = correction
            ds = 1e-2
            self.last_ds = ds

        if self.has_hit:
            return self.return_act(ee_pos=ee_pos, step_size=self.last_ds * 4)

        # NEXT POINT COMPUTATION ----------------------------------------------
        if np.abs(puck_pos[1] - ee_pos[1]) < 2 * self.mallet_radius:
            # Puck is in the bottom part
            # if ee_pos_puck[1] < radius * np.sin(np.deg2rad(180)):
            #    d_beta = -d_beta

            # Compute the action
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta - d_beta))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta - d_beta))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)

        else:
            # Puck is in the top part
            # Compute the action
            # if ee_pos_puck[1] > radius * np.sin(np.deg2rad(180)):
            #    d_beta = -d_beta
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta + d_beta))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta + d_beta))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)

        self.last_position = action
        return action

    def return_act(self, ee_pos, target=DEFAULT_POS, step_size=0.05):
        """
            Implements a standard return act, it brings the ee back
            to a given position moving with a given step size (not optimized)

            The ee_pos must be in the same coordinates of the target
        """
        # ee_pos = self.state.w_puck_pos[:2]

        ds_x = (target[0] - ee_pos[0]) * step_size
        ds_y = (target[1] - ee_pos[1]) * step_size

        action = np.array([ee_pos[0] + ds_x, ee_pos[1] + ds_y])

        return action

    def smooth_act(self, target_point=DEFAULT_POS[:2], step_size=5e-3):
        """
        Reach a given point following a curved trajectory, reach the point from behind
        The target point is in world coordinates
        """
        ee_pos = self.state.w_ee_pos[:2]

        # Beta computation
        beta = self.get_angle(
            self.world_to_puck(target_point + np.array([1, 0]), target_point),
            self.world_to_puck(ee_pos, target_point)
        )

        # Take the current radius
        radius = np.sqrt(
            (ee_pos[0] - target_point[0]) ** 2 + (ee_pos[1] - target_point[1]) ** 2
        )

        rounded_radius = round(radius - (self.mallet_radius + self.puck_radius), 5)

        point_reached = False
        if rounded_radius <= 1e-2:
            point_reached = True

        correction = np.abs(360 - beta) if target_point[1] >= 0 else np.abs(beta - 0.1)

        #d_beta = (self.theta[0] + self.theta[1] * self.env_info["dt"]) * correction
        d_beta = 1e-2
        ds = step_size

        # NEXT POINT COMPUTATION ----------------------------------------------
        if target_point[1] <= ee_pos[1]:
            # Puck is in the bottom part
            # Compute the action
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta - d_beta))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta - d_beta))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), target_point)
        else:
            # Puck is in the top part
            # Compute the action
            x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta + d_beta))
            y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta + d_beta))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), target_point)

        self.last_position = action
        return action, point_reached

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
        y_tol = np.round(((self.table_width / 2) - side_tolerance) - np.abs(puck_pos[1]), 5)

        tolerance = self.puck_radius + 2 * self.mallet_radius

        if y_tol < tolerance or x_tol < tolerance:
            enough_space = False

        return enough_space
