import math

import numpy as np
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.utils import forward_kinematics, inverse_kinematics
from air_hockey_challenge.utils.transformations import robot_to_world, world_to_robot


class PolicyAgent(AgentBase):

    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)

        # THETAS
        self.best_hit = np.array([9.99302077e-02, 9.65953660e-03, 2.59815166e-05, 5.01348799e-02])
        self.best_defend = np.array([0.5, 0.1, 0.7, 0.2, 0.9])
        self.best_prepare = np.array([0.01, 0.001, 0.002, 0.01])
        self.theta = self.best_hit
        self.task = 'hit'  # FIXME it must be changed

        self.has_hit = False

        # HIT: define initial values
        self.previous_beta = 0  # angle between ee->puck and puck->goal lines FIXME 0 is not the best idea
        self.last_position = np.zeros(2)
        self.final = 4

        # DEFEND: define initial values
        self.desired_from = None

        # RETURN TRAJECTORY
        self.default_action = np.array([-0.8, 0])  # it will also be the return point after a hit

        # Environment specs
        self.frame = self.env_info['robot']['base_frame'][0]
        self.table_length = self.env_info['table']['length']
        self.table_width = self.env_info['table']['width']
        self.goal_pos = np.array([self.table_length / 2, 0])  # goal coordinates
        self.mallet_radius = self.env_info['mallet']['radius']
        self.puck_radius = self.env_info['puck']['radius']
        self.defend_line = -0.7

        self.time = 0
        self.done = False

    def reset(self):
        # hit/prepare reset
        self.previous_beta = 0
        self.last_position = np.zeros(2)
        self.final = 4

        # defend reset
        self.desired_from = None
        self.default_action = np.array([-0.8, 0])

        # general reset
        self.time = 0
        self.has_hit = False
        self.done = False

    def draw_action(self, observation):
        """
            This function will provide the next action necessary to perform
            a hit a defend or a prepare
        """
        action = self.act(observation)
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

        # Get info from the state
        puck_pos = state[0:2]  # x,y ignore yaw
        puck_vel = state[3:5]  # x,y,yaw
        j_pos = state[6:9]     # joint_positions
        ee_vel = state[9:11]   # joint_velocities
        self.time += 1

        # Define the base frame to perform coordinates conversions
        base_frame = self.env_info['robot']['base_frame'][0]

        # Retrieve intended puck and ee positions
        puck_pos, _ = robot_to_world(base_frame, puck_pos)[:2]
        ee_pos = self._apply_forward_kinematics(j_pos)[:2]

        # Decide the action to perform
        if ee_vel[0] < 0:  # puck is coming towards the robot TODO is there a better way?
            self.task = "defend"
        else:
            # Check if there is enough space to hit
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

            tolerance_coordinates = np.array([tolerance * np.cos(np.deg2rad(tot_angle)),
                                              tolerance * np.sin(np.deg2rad(tot_angle))])

            tolerance_coordinates = self.puck_to_world(tolerance_coordinates, puck_pos)

            if (tolerance_coordinates[0] <= -self.table_length / 2) or \
                    (tolerance_coordinates[1] >= self.table_width / 2 or tolerance_coordinates[1] <= -self.table_width / 2):
                enough_space = False

            if enough_space:
                self.task = "hit"
            else:
                self.task = "prepare"

        if 'hit' in self.env_info['env_name']:
            self.task = 'hit'
        else:
            self.task = 'defend'

        #self.task = "hit"  # FIXME use if hit in env_name to do hit or defend

        if self.task == "hit":
            self.theta = self.best_hit
            action = self.hit_act(state)
        elif self.task == "defend":
            self.theta = self.best_defend
            action = self.defend_act(state)
        elif self.task == "prepare":
            self.theta = self.best_prepare
            action = self.prepare_act(state)
        else:
            action = None

        # Clip the action with respect to the constraint
        action_x = np.clip(action[0], -self.table_length / 2, self.table_length / 2)
        action_y = np.clip(action[1], -self.table_width / 2, self.table_width / 2)
        action = np.array([action_x, action_y])

        # Map the action in robot coordinates and convert them into joint actions
        final_action = np.zeros((2, 3))
        final_action[0] = np.append(action, 0)

        final_action[0], _ = world_to_robot(base_frame, final_action[0])
        final_action[0] = self._apply_inverse_kinematics(final_action[0])

        # define velocity through discrete derivative
        final_action[1] = (final_action[0] - j_pos) / self.env_info['dt']

        # Clip joint positions in the limits
        joint_pos_limits = self.env_info['robot']['joint_pos_limit']
        joint_vel_limits = self.env_info['robot']['joint_vel_limit']

        new_final_action = np.zeros((2, 3))

        new_final_action[0] = np.clip(final_action[0], joint_pos_limits[0], joint_pos_limits[1])
        new_final_action[1] = np.clip(final_action[1], joint_vel_limits[0], joint_vel_limits[1])

        return new_final_action

    def hit_act(self, state):
        """
        Description:
            this function computes the new action to perform to perform a "hit"

        Args:
            state (list): a list of the state elements, please see
            envs/air_hockey/_get_state()

        Returns:
             numpy.array([*, *]): the new position the end effector should
             occupy in the next step
        """

        # Get info from the state
        puck_pos = state[0:2]  # x,y ignore yaw
        puck_vel = state[3:5]  # x,y,yaw
        j_pos = state[6:9]     # joint_positions
        ee_vel = state[9:11]   # joint_velocities

        # Define the base frame to perform coordinates conversions
        base_frame = self.env_info['robot']['base_frame'][0]

        # Retrieve intended puck and ee positions
        puck_pos, _ = robot_to_world(base_frame, puck_pos)
        ee_pos = self._apply_forward_kinematics(j_pos)[:2]

        # Chek if there was a hit
        if np.linalg.norm(puck_vel[:2]) > 0.1:
            self.has_hit = True

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
        radius = np.sqrt(ee_pos_puck[0] ** 2 + ee_pos_puck[1] ** 2)

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
                    ds *= self.time / radius
                else:
                    ds = self.theta[3]

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
                    ds *= self.time / radius
                else:
                    ds = self.theta[3]

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
            step_size = 0.1

            target = self.default_action

            ds_x = (target[0] - ee_pos[0]) * step_size
            ds_y = (target[1] - ee_pos[1]) * step_size

            action = np.array([ee_pos[0] + ds_x, ee_pos[1] + ds_y])

        return action

    def defend_act(self, state):
        """
        Description:
            this function computes the new action to perform to perform a
            "defend"

        Args:
            state (list): a list of the state elements, please see
            envs/air_hockey/_get_state()

        Returns:
            numpy.array([*, *]): the new position the end effector should
            occupy in the next step
        """

        # Get info from the state
        puck_pos = state[0:2]  # x,y ignore yaw
        puck_vel = state[3:5]  # x,y,yaw
        j_pos = state[6:9]     # joint_positions
        ee_vel = state[9:11]   # joint_velocities

        # Define the base frame to perform coordinates conversions
        base_frame = self.env_info['robot']['base_frame'][0]

        # Retrieve intended puck and ee positions
        puck_pos, _ = robot_to_world(base_frame, puck_pos)
        ee_pos = self._apply_forward_kinematics(j_pos)[:2]

        # Check if there was a hit
        if np.linalg.norm(puck_vel[:2]) < 0.2 or np.abs(puck_vel[0]) < 0.1 or np.abs(puck_vel[1]) < 0.1:
            self.has_hit = True

        # Check if we want a top or bottom action
        if self.done:
            if puck_pos[1] >= ee_pos[1]:
                self.desired_from = "bottom"
            else:
                self.desired_from = "top"
        else:
            if self.time == 1:
                if puck_pos[1] >= ee_pos[1]:
                    self.desired_from = "bottom"
                else:
                    self.desired_from = "top"

        '''    
        if self.done:
            self.done = False
            self.final = 4
            self.time = 1
            if puck_pos[1] >= ee_pos[1]:
                self.desired_from = "bottom"
            else:
                self.desired_from = "top"
        else:
            self.time += 1
            if self.time == 1:
                if puck_pos[1] >= ee_pos[1]:
                    self.desired_from = "bottom"
                else:
                    self.desired_from = "top"
        '''
        # TODO
        # Pre-processing of Thetas (if any)
        self.theta[0] = np.clip(self.theta[0], .00000001, .99999999)
        self.theta[1] = np.clip(self.theta[1], .00000001, .99999999)
        self.theta[2] = np.clip(self.theta[2], .00000001, .99999999)
        self.theta[3] = np.clip(self.theta[3], .00000001, .99999999)
        self.theta[4] = np.clip(self.theta[4], .00000001, .99999999)

        # Define the targets
        if self.desired_from == "top":
            y_target = puck_pos[1] + self.theta[0] * (self.puck_radius + self.mallet_radius)
        else:
            y_target = puck_pos[1] - self.theta[0] * (self.puck_radius + self.mallet_radius)

        x_target = self.defend_line

        # Compute ds
        if not self.has_hit:
            ds_x = (self.theta[1] + self.theta[2] * np.linalg.norm(puck_vel)) * (x_target - ee_pos[0])
            ds_y = (self.theta[3] + self.theta[4] * np.linalg.norm(puck_vel)) * (y_target - ee_pos[1])
        else:
            step_size = 0.1

            target = self.default_action

            ds_x = (target[0] - ee_pos[0]) * step_size
            ds_y = (target[1] - ee_pos[1]) * step_size

        # Compute the action
        action = np.array([ee_pos[0] + ds_x, ee_pos[1] + ds_y])

        return action

    def prepare_act(self, state):
        """
        Description:
            this function computes the new action to perform to perform a
            "prepare"

        Args:
            state (list): a list of the state elements, please see
            envs/air_hockey/_get_state()

        Returns:
            numpy.array([*, *]): the new position the end effector should
            occupy in the next step
        """
        # Get info from the state
        puck_pos = state[0:2]  # x,y ignore yaw
        puck_vel = state[3:5]  # x,y,yaw
        j_pos = state[6:9]     # joint_positions
        ee_vel = state[9:11]   # joint_velocities

        # Define the base frame to perform coordinates conversions
        base_frame = self.env_info['robot']['base_frame'][0]

        # Retrieve intended puck and ee positions
        puck_pos, _ = robot_to_world(base_frame, puck_pos)
        ee_pos = self._apply_forward_kinematics(j_pos)[:2]

        # Chek if there was a hit
        if np.linalg.norm(puck_vel[:2]) > 0.1:
            self.has_hit = True

        # Initialization of d_beta and ds
        d_beta = self.theta[0] + self.theta[1] * self.time
        ds = self.theta[2]

        # Convert the ee position and take the current radius
        ee_pos_puck = self.world_to_puck(ee_pos, puck_pos)
        radius = np.sqrt(ee_pos_puck[0] ** 2 + ee_pos_puck[1] ** 2)

        # beta computation
        beta = self.get_angle(
            self.world_to_puck(puck_pos + np.array([1, 0]), puck_pos),
            self.world_to_puck(ee_pos, puck_pos)
        )

        if not self.has_hit or self.final:
            # reset final flag
            if self.has_hit:
                self.final -= 2

            if puck_pos[1] <= 0:
                # Puck in the bottom part
                # update d_beta and ds
                d_beta *= np.abs(90 - beta)

                if not self.has_hit:
                    ds *= (self.time * radius / d_beta)
                else:
                    ds = self.theta[3]

                # clip movement when optimal beta is reached
                if np.abs(beta - 90) <= 0.01:
                    d_beta = 1e-10

                x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta - d_beta))
                y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta - d_beta))
            else:
                # Puck in the top part
                d_beta *= np.abs(270 - beta)

                if not self.has_hit:
                    ds *= (self.time * radius / d_beta)
                else:
                    ds = self.theta[3]

                # clip movement when optimal beta is reached
                if np.abs(beta - 270) <= 0.01:
                    d_beta = 1e-10

                x_inters_reduced = (radius - ds) * np.cos(np.deg2rad(beta + d_beta))
                y_inters_reduced = (radius - ds) * np.sin(np.deg2rad(beta + d_beta))
            action = self.puck_to_world(np.array([x_inters_reduced, y_inters_reduced]), puck_pos)
            self.last_position = action
        else:
            # Already hit, return to the initial position moving with a given
            # step size (not optimized)
            step_size = 0.1

            target = self.default_action

            ds_x = (target[0] - ee_pos[0]) * step_size
            ds_y = (target[1] - ee_pos[1]) * step_size

            action = np.array([ee_pos[0] + ds_x, ee_pos[1] + ds_y])

        return action

    def _apply_inverse_kinematics(self, ee_pos_robot_frame):
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']
        #position_robot_frame, rotation = world_to_robot(self.env_info['robot']['base_frame'][0], ee_pos_robot_frame)
        success, action_joints = inverse_kinematics(mj_model, mj_data, ee_pos_robot_frame) # inverse_kinematics uses robot's frame for coordinates
        return action_joints

    def _apply_forward_kinematics(self, joint_pos):
        mj_model = self.env_info['robot']['robot_model']
        mj_data = self.env_info['robot']['robot_data']
        ee_pos_robot_frame, rotation = forward_kinematics(mj_model, mj_data, joint_pos)
        ee_pos_world_frame, rotation = robot_to_world(self.env_info['robot']['base_frame'][0], ee_pos_robot_frame)
        return ee_pos_world_frame

    def get_intersect(self, a1, a2, b1, b2):
        """
        Returns the intersection point of the lines passing through a1,a2 and b1,b2.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1, a2, b1, b2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return np.array([float('inf'), float('inf')])
        return np.array([x / z, y / z])

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

