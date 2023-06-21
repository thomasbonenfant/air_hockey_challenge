"""
Agent working via a Rule-based Policy
High-level action: control the end-effector
"""
# Libraries
import math
import numpy as np
from air_hockey_agent.model import State
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot

# Macros
EE_HEIGHT = 0.0645
PREV_BETA = 0
FINAL = 10
DEFAULT_POS = np.array([-0.8, 0, EE_HEIGHT])
DEFEND_LINE = -0.7
DES_ACC = 0.01
BEST_PARAM = dict(
    hit=np.array([9.99302077e-02, 9.65953660e-03, 2.59815166e-05, 5.01348799e-02]),
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

        # Task allocation
        self.task = task

        # Thetas
        assert self.task in ["hit", "defend", "prepare"], f"Illegal task: {self.task}"
        self.theta = BEST_PARAM[self.task]

        # HIT: define initial values
        # angle between ee->puck and puck->goal lines 
        self.previous_beta = PREV_BETA
        self.last_position = np.zeros(3)
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
        self.done = False
        self.has_hit = False
        self.can_hit = False  # Can hit considering enemy position
        self.state = State
        self.last_ds = 0
        self.phase = "wait"

    def reset(self):
        # Generic stuff
        self.previous_beta = PREV_BETA
        self.last_position = np.zeros(3)
        self.final = FINAL
        self.desired_from = None

        # Episode parameters
        self.done = None
        self.has_hit = False
        self.can_hit = False
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
        Descriprion: 
            This funciton draws an action given the observation and the task.

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
        self.state.r_joint_pos = state[6:13]
        self.state.r_joint_vel = state[13:20]
        self.state.r_adv_ee_pos = state[-3:]  # adversary ee position in robot coordinates

        # Compute EE pos
        self.state.r_ee_pos = self._apply_forward_kinematics(joint_pos=self.state.r_joint_pos)

        # Convert in WORLD coordinates 2D
        self.state.w_puck_pos, _ = robot_to_world(base_frame=self.frame,
                                                  translation=self.state.r_puck_pos)
        self.state.w_ee_pos, _ = robot_to_world(base_frame=self.frame,
                                                translation=self.state.r_ee_pos)

        # Compute the action basing on the task
        # Note: here all the actions must be in WORLD coordinates
        self.theta = BEST_PARAM[self.task]
        if self.task == "hit":
            action = self.new_hit_act()
        elif self.task == "defend":
            action = self.defend_act(state)
        elif self.task == "prepare":
            action = self.prepare_act(state)
        else:
            action = None

        # Post-process the action
        # at this point we have a desired ee_x and ee_y
        # keep ee_z <- 0
        action_x = np.clip(action[0], -self.table_length / 2 + self.mallet_radius,
                           self.table_length / 2 - self.mallet_radius)
        action_y = np.clip(action[1], -self.table_width / 2 + self.mallet_radius,
                           self.table_width / 2 - self.mallet_radius)
        action_z = EE_HEIGHT
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
        joint_pos_limits = self.env_info['robot']['joint_pos_limit']
        joint_vel_limits = self.env_info['robot']['joint_vel_limit']

        new_final_action = np.zeros((2, 7))
        new_final_action[0] = np.clip(final_action[0], joint_pos_limits[0], joint_pos_limits[1])
        new_final_action[1] = np.clip(final_action[1], joint_vel_limits[0], joint_vel_limits[1])
        # new_final_action[2] = (final_action[1] - self.state.r_joint_vel)/self.env_info["dt"]

        # if (self.has_hit and self.final == 0) or self.state.r_puck_vel[0] < -0.1:
        if self.phase == "wait" or self.final == 0:
            new_final_action[1] = np.zeros(7)
            # new_final_action[2] = np.zeros(7)

        return new_final_action

    '''def hit_act(self):
        """
        Description:
            this function computes the new action to perform to perform a "hit"

        Returns:
            numpy.array([*, *]): the new position the end effector should
            occupy in the next step
        """
        # Get useful info from the state
        puck_pos = self.state.w_puck_pos[:2]
        ee_pos = self.state.w_ee_pos[:2]

        # Initialization of d_beta and ds
        #d_beta = self.theta[0] + self.theta[1] * self.time + 0.008*np.linalg.norm(self.state.r_puck_vel)
        #d_beta = 0.1 + 0.1 * self.time * self.env_info["dt"] + 0.08 * np.linalg.norm(self.state.r_puck_vel)
        d_beta = 0.1 + 0.1 * self.time * self.env_info["dt"] + 0.1 * np.abs(self.state.r_puck_vel[1])
        #ds = self.theta[2]
        ds = 3e-4

        # beta computation
        beta = self.get_angle(
            self.world_to_puck(self.goal_pos, puck_pos),
            self.world_to_puck(ee_pos, puck_pos)
        )

        # gamma computation
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
        # radius = np.sqrt(ee_pos_puck[0] ** 2 + ee_pos_puck[1] ** 2)
        radius = np.sqrt(
            (ee_pos[0] - puck_pos[0]) ** 2 + (ee_pos[1] - puck_pos[1]) ** 2
        )
        
        # HAS_HIT computation
        tolerance = 5e-3
        if radius - self.mallet_radius - self.puck_radius <= tolerance and not self.has_hit:
            self.has_hit = True

        # Check if there was a hit
        if not self.has_hit or self.final:
            # reset final flag
            if self.has_hit:
                if self.final == FINAL:
                    self.last_ds[1] = self.last_ds[0]
                self.final -= 1
                
            # ds computation
            if not self.has_hit:
                #ds *= self.time / (radius + self.mallet_radius + self.puck_radius)
                ds *= (self.time*self.env_info["dt"] / (radius - self.mallet_radius)) #- 0.08 * np.linalg.norm(self.state.r_puck_vel)
                #ds *= (self.time / (radius)) + 0.008 * self.state.r_puck_vel[0]
                #ds *= self.time / radius
                #ds -= 0.008 * np.linalg.norm(self.state.r_puck_vel)/(self.time)
                #ds -= 0.005/(np.abs(self.state.w_puck_pos[0]))
                #ds += 0.005 / (np.abs(180 - beta) + 1e-4)
                if np.abs(180 - beta) <= 1.8:
                    ds += 0.4*(DES_ACC - ds)
                else:
                    ds -= 0.1*self.last_ds[0]
                self.last_ds[0] = ds
            else:
                #ds = self.last_ds + 0.02
                #ds *= 3 * np.abs(self.state.w_puck_pos[0])
                #ds *= 1 / np.abs(self.state.w_puck_pos[0] + self.table_length/2)
                #ds *= 0.3 / (FINAL - self.final)
                #norm = (self.state.w_puck_pos[0] + self.table_length/2) / (self.table_length/2)
                #ds = self.last_ds
                #ds += 0.03 * (DES_ACC - self.last_ds)
                #ds += 0.03 * (DES_ACC)
                #ds -= 1e-4 * norm
                #ds *= 1 / (FINAL - self.final)
                #ds *= 0.5 / np.abs(self.state.w_puck_pos[0] + self.table_length/2)
                #ds = (self.last_ds[1] + 0.05 * max(0, DES_ACC - self.last_ds[0])) / self.time
                ds = self.last_ds[1] / (FINAL - self.final)
                #ds = max(ds, self.last_ds[0] + DES_ACC)
                self.last_ds[1] = ds

            # Normal hit
            if puck_pos[1] <= self.table_width / 2:
                # Puck is in the bottom part
                # Update d_beta and ds
                d_beta *= np.abs(180 - beta)
    
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
                # Update d_beta and ds
                d_beta *= np.abs(beta - 180)
                
                # Check if d_beta needs to be inverted
                if ee_pos_puck[1] > radius * np.sin(np.deg2rad(180 - gamma)):
                    d_beta = -d_beta

                # Compute the action
                x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta - d_beta - gamma))
                y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta - d_beta - gamma))
                action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)
                self.last_position = action
        else:
            # Already hit, return to the initial position moving with a given step size (not optimized)
            step_size = 0.005

            target = self.default_action

            ds_x = (target[0] - ee_pos[0]) * step_size
            ds_y = (target[1] - ee_pos[1]) * step_size

            action = np.array([ee_pos[0] + ds_x, ee_pos[1] + ds_y])
            action = self.last_position

        return action
    '''

    def defend_act(self, state):
        pass

    def prepare_act(self, state):
        pass

    def new_hit_act(self):
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

        # correction term
        correction = np.abs(180 - beta) if puck_pos[1] <= self.table_width / 2 else np.abs(beta - 180)

        # PHASES --------------------------------------------------------------
        # WAIT: wait for the puck to move slowly
        if self.phase == "wait" and (np.linalg.norm(self.state.r_puck_vel) < 2 or self.state.r_puck_vel[0] >= 0):
            self.phase = "adjustment"

        # ADJUSTMENT: adjust the beta angle and go to the right trajectory
        if self.phase == "adjustment":
            self.time[self.phase] += 1
            if correction <= 2:
                self.phase = "acceleration"
            else:
                d_beta = (0.1 + 0.05 * self.time[self.phase] * self.env_info["dt"]) * correction
                ds = 1e-4
                self.last_ds = ds

        # ACCELERATION: the ee is in the correct line, accelerate
        if self.phase == "acceleration":
            self.time[self.phase] += 1

            # Check if can hit considering the adversary position
            adv_ee_pos = self.state.r_adv_ee_pos
            lower_bound_goal = - 0.25 / 2
            upper_bound_goal = 0.25 / 2

            # check if adv_ee moved in around the bounds of the goal
            if (adv_ee_pos[1] <= 0 and np.abs(adv_ee_pos[1] - lower_bound_goal) < 0.08) or \
                    (adv_ee_pos[1] >= 0 and np.abs(adv_ee_pos[1] - upper_bound_goal) < 0.08):
                self.can_hit = True

            #if np.abs(adv_ee_pos[1]) <= 0.01:
            #    self.can_hit = True

            if np.abs(radius - self.mallet_radius - self.puck_radius) <= 5e-3:
                self.phase = "smash"

            elif self.can_hit:
                # update ds considering how far it is from the left short-side
                d_beta = correction
                #ds = (self.last_ds + 1e-4 * self.env_info["dt"] * self.time[self.phase] / (radius + self.mallet_radius))
                ds = (self.last_ds + 1e-4 * self.env_info["dt"] * self.time[self.phase] / (radius + self.mallet_radius))
                ds *= (1/self.state.r_puck_pos[0] - 1e-3)
                self.last_ds = ds
            else:
                ds = 1e-4

        # SMASH: perform a smash right after the "has_hit" computation
        if self.phase == "smash":
            self.time[self.phase] += 1
            if self.final == 0:
                return self.last_position
            else:
                self.final -= 1
                d_beta = correction
                ds = self.last_ds + 0.01 * (DES_ACC - self.last_ds) / (FINAL - self.final)
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
