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
PREV_BETA = 0   # FIXME: change this value
FINAL = 10
DEFAULT_POS = np.array([-0.65, 0, EE_HEIGHT])
DEFEND_LINE = -0.7
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
        self.last_action = np.zeros((2, 7))  # last action done, joint pos and vel
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
        self.time = 0
        self.done = False
        self.has_hit = False
        self.state = State
        self.init_sign_puck_vel = None
    
    def reset(self):
        # Generic stuff
        self.previous_beta = PREV_BETA
        self.last_position = np.zeros(3)
        self.last_action = np.zeros((2, 7))
        self.final = FINAL
        self.desired_from = None
        
        # Episode parameters
        self.done = None
        self.has_hit = False
        self.time = 0
        self.state = State
        self.init_sign_puck_vel = None
    
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
        self.time += 1
        
        # Get useful info from the state
        self.state.r_puck_pos = state[0:3]
        self.state.r_puck_vel = state[3:6]
        self.state.r_joint_pos = state[6:13]
        self.state.r_joint_vel = state[13:20]
        
        # Register the sign of the initial velocity of the puck
        if self.time == 1:
            self.init_sign_puck_vel = self.state.r_puck_vel[0]
        
        # Compute EE pos
        self.state.r_ee_pos = self._apply_forward_kinematics(joint_pos=self.state.r_joint_pos)
        
        # Convert in WORLD coordinates 2D
        self.state.w_puck_pos, _ = robot_to_world(base_frame=self.frame, 
                                                  translation=self.state.r_puck_pos)
        self.state.w_ee_pos, _ = robot_to_world(base_frame=self.frame, 
                                                translation=self.state.r_ee_pos)
        
        # FIXME Decide the task to accomplish at runtime 
        '''if puck_vel[0] < 0:
            # DEFEND: the puck is going towards the robot
            self.task = "defend"
        else:
            # HIT or PREPARE
            self.theta = BEST_PARAM["hit"]
            enough_space = True
            tolerance = self.theta[0] * self.mallet_radius * 2
            
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
                tot_angle = beta + gamma
            else:
                gamma = self.get_angle(
                    self.world_to_puck(self.goal_pos, puck_pos),
                    self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos)
                )
                tot_angle = beta - gamma

            tolerance_coordinates = np.array(
                [tolerance * np.cos(np.deg2rad(tot_angle)),
                tolerance * np.sin(np.deg2rad(tot_angle))]
            )

            tolerance_coordinates = self.puck_to_world(tolerance_coordinates, puck_pos)

            if (tolerance_coordinates[0] <= -self.table_length / 2) or \
                    (tolerance_coordinates[1] >= self.table_width / 2 or tolerance_coordinates[1] <= -self.table_width / 2):
                enough_space = False

            if enough_space:
                self.task = "hit"
            else:
                self.task = "prepare"'''
        
        # Compute the action basing on the task
        # Note: here all the actions must be in WORLD coordinates
        self.theta = BEST_PARAM[self.task]
        if self.task == "hit":
            action = self.hit_act()
        elif self.task == "defend":
            action = self.defend_act(state)
        elif self.task == "prepare":
            action = self.prepare_act(state)
        else:
            action = None
        
        # Post-process the action
        # at this point we have a desired ee_x and ee_y
        # keep ee_z <- 0
        action_x = np.clip(action[0], -self.table_length / 2 + self.mallet_radius, self.table_length / 2 - self.mallet_radius)
        action_y = np.clip(action[1], -self.table_width / 2 + self.mallet_radius, self.table_width / 2 - self.mallet_radius)
        action_z = EE_HEIGHT
        action = np.array([action_x, action_y, action_z])
        
        # convert ee coordinates in the robot frame
        action, _ = world_to_robot(base_frame=self.frame, translation=action)
        
        # apply inverse kinematics
        final_action = np.zeros((2, 7))
        final_action[0], success = self._apply_inverse_kinematics(
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

        self.last_action = final_action
        return new_final_action
    
    def hit_act(self):
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

        # Chek if there was a hit
        # Naive approach: check if the sign of puck_vel_x has changed
        #if np.linalg.norm(self.state.w_puck_vel[:2]) > 0.1:
        #if self.init_sign_puck_vel * self.state.r_puck_vel[0] < 0 and not self.has_hit:
        #    self.has_hit = True

        #if self.init_sign_puck_vel == 0 and self.state.r_puck_vel[0] > 0 and not self.has_hit:
        #    self.has_hit = True

        # Initialization of d_beta and ds
        d_beta = self.theta[0] + self.theta[1] * self.time
        ds = self.theta[2]

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
        #radius = np.sqrt(ee_pos_puck[0] ** 2 + ee_pos_puck[1] ** 2)

        radius = np.sqrt(
            (ee_pos[0] - puck_pos[0]) ** 2 +
            (ee_pos[1] - puck_pos[1]) ** 2
        )

        tolerance = 1e-2
        # Check if there was a hit by looking at the distance between the ee and the puck
        if radius - self.mallet_radius - self.puck_radius <= tolerance and not self.has_hit:
            #print(radius - self.mallet_radius - self.puck_radius)
            self.has_hit = True

        # Check if there was a hit
        if not self.has_hit or self.final:
            # reset final flag
            if self.has_hit:
                self.final -= 1

            # Normal hit
            if puck_pos[1] <= self.table_width / 2:
                # Puck is in the bottom part
                # Update d_beta and ds
                d_beta *= np.abs(180 - beta)
                if not self.has_hit:
                    ds *= self.time / (radius + self.mallet_radius + self.puck_radius)
                    self.last_ds = ds
                else:
                    # subtract something depending on the puck's speed
                    ds = self.last_ds
                    reduce = self.theta[3] * 0.05 #* np.linalg.norm(puck_vel)
                    ds -= reduce
                    if ds <= 0:
                        ds = 1e-2
                    self.last_ds = ds

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
                if not self.has_hit:
                    ds *= self.time / (radius + self.mallet_radius + self.puck_radius)
                    self.last_ds = ds
                else:
                    # subtract something depending on the puck's speed
                    ds = self.last_ds
                    reduce = self.theta[3] * 0.05 #* np.linalg.norm(puck_vel)
                    ds -= reduce
                    if ds <= 0:
                        ds = 1e-2
                    self.last_ds = ds


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
            step_size = 0.01

            target = self.default_action

            ds_x = (target[0] - ee_pos[0]) * step_size
            ds_y = (target[1] - ee_pos[1]) * step_size

            action = np.array([ee_pos[0] + ds_x, ee_pos[1] + ds_y])
            #action = self.last_position

        return action
    
    def defend_act(self, state):
        pass
    
    def prepare_act(self, state):
        pass

    def return_act(self, state):
        pass

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

        if not success:
            print("OH NO")
            print(self.has_hit, '\n')
            #self.has_hit = True
        return action_joints, success
        
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