from copy import deepcopy

from air_hockey_challenge.constraints import *
from air_hockey_challenge.environments import position_control_wrapper as position
from air_hockey_challenge.utils import robot_to_world
from mushroom_rl.core import Environment


PENALTY_POINTS = {"joint_pos_constr": 2, "ee_constr": 3, "joint_vel_constr": 1, "jerk": 1, "computation_time_minor": 0.5,
                  "computation_time_middle": 1,  "computation_time_major": 2}
contraints = PENALTY_POINTS.keys()
class AirHockeyChallengeWrapper(Environment):
    def __init__(self, env, custom_reward_function=None, interpolation_order=3, **kwargs):
        """
        Environment Constructor

        Args:
            env [string]:
                The string to specify the running environments. Available environments: [3dof-hit, 3dof-defend].
                [7dof-hit, 7dof-defend, 7dof-prepare, tournament] will be available once the corresponding stage starts.
            custom_reward_function [callable]:
                You can customize your reward function here.
            interpolation_order (int, 3): Type of interpolation used, has to correspond to action shape. Order 1-5 are
                    polynomial interpolation of the degree. Order -1 is linear interpolation of position and velocity.
                    Set Order to None in order to turn off interpolation. In this case the action has to be a trajectory
                    of position, velocity and acceleration of the shape (20, 3, n_joints)
        """

        env_dict = {
            "7dof-hit": position.IiwaPositionHit,
            "7dof-defend": position.IiwaPositionDefend,
            "7dof-prepare": position.IiwaPositionPrepare,

            "3dof-hit": position.PlanarPositionHit,
            "3dof-defend": position.PlanarPositionDefend
        }

        self.base_env = env_dict[env](interpolation_order=interpolation_order, **kwargs)
        self.env_name = env
        self.env_info = self.base_env.env_info

        if custom_reward_function:
            self.base_env.reward = lambda state, action, next_state, absorbing: custom_reward_function(self.base_env,
                                                                                                       state, action,
                                                                                                       next_state,
                                                                                                       absorbing)

        constraint_list = ConstraintList()
        constraint_list.add(JointPositionConstraint(self.env_info))
        constraint_list.add(JointVelocityConstraint(self.env_info))
        constraint_list.add(EndEffectorConstraint(self.env_info))

        self.env_info['constraints'] = constraint_list
        self.env_info['env_name'] = self.env_name





        """
        added lines
        """
        # self.base_env.reward = lambda state, action, next_state, absorbing: self._reward(action, absorbing, info)
        self.simple_reward = True
        self.shaped_reward = False

        if "hit" in env:
            self._shaped_r = self._shaped_r_hit
            self.hit_env = True
            self.defend_env = False

        self.large_reward = 50
        self.large_penalty = 50

        self.t = 0


        super().__init__(self.base_env.info)
    def step(self, action):
        percentage = 0.1
        obs, reward, done, info = self.base_env.step(action)
        self.t += 1
        if "tournament" in self.env_name:
            info["constraints_value"] = list()
            info["jerk"] = list()
            for i in range(2):
                obs_agent = obs[i * int(len(obs) / 2): (i + 1) * int(len(obs) / 2)]
                info["constraints_value"].append(deepcopy(self.env_info['constraints'].fun(
                    obs_agent[self.env_info['joint_pos_ids']], obs_agent[self.env_info['joint_vel_ids']])))
                info["jerk"].append(
                    self.base_env.jerk[i * self.env_info['robot']['n_joints']:(i + 1) * self.env_info['robot'][
                        'n_joints']])

            info["score"] = self.base_env.score

        else:
            info["constraints_value"] = deepcopy(self.env_info['constraints'].fun(obs[self.env_info['joint_pos_ids']],
                                                                                  obs[self.env_info['joint_vel_ids']]))
            info["jerk"] = self.base_env.jerk
            info["success"] = self.check_success(obs)
        reward = self._reward(action, done, info)


        flag = np.random.uniform() < percentage

        if flag:


        return obs, reward, done, info

    def render(self):
        self.base_env.render()

    def reset(self, state=None):
        self.t += 1
        return self.base_env.reset(state)

    def check_success(self, obs):
        puck_pos, puck_vel = self.base_env.get_puck(obs)

        puck_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=puck_pos)
        success = 0

        if "hit" in self.env_name:
            if puck_pos[0] - self.base_env.env_info['table']['length'] / 2 > 0 and \
                    np.abs(puck_pos[1]) - self.base_env.env_info['table']['goal_width'] / 2 < 0:
                success = 1

        elif "defend" in self.env_name:
            if -0.8 < puck_pos[0] <= -0.2 and puck_vel[0] < 0.1:
                success = 1

        elif "prepare" in self.env_name:
            if -0.8 < puck_pos[0] <= -0.2 and np.abs(puck_pos[1]) < 0.39105 and puck_vel[0] < 0.1:
                success = 1
        return success

    def _reward(self, action,  done, info):
        if self.simple_reward:
            r = info["success"]
        elif self.shaped_reward:
            r = self._shaped_r(action, info)
        else:
            r = 0
            for constr in ['joint_pos_constr', 'joint_vel_constr', 'ee_constr']:
                if np.any(np.array(info[constr] > 0)):
                    r -= PENALTY_POINTS[constr]
            r += info["success"] * self.large_reward
        if done and not info["success"] and self.t < self.base_env._mdp_info.horizon:
            r -= self.large_penalty
        return r

    def _shaped_r_hit(self, action, info):
        return 0
    #     """
    #            Custom reward function, it is in the form: r_tot = r_hit + alpha * r_const
    #
    #            - r_hit: reward for the hitting part
    #            - r_const: reward for the constraint part
    #            - alpha: constant
    #            r_hit:
    #                  || p_ee - p_puck||
    #              -  ------------------ - c * ||a||      , if no hit
    #                  0.5 * diag_tavolo
    #                    vx_hit
    #                -------------- - c * ||a||          , if has hit
    #                  big_number
    #                   1
    #                -------                             , if goal
    #                 1 - Ɣ
    #            """
    #     # compute table diagonal
    #     table_length = self.env_info['table']['length']
    #     table_width = self.env_info['table']['width']
    #     table_diag = np.sqrt(table_length ** 2 + table_width ** 2)
    #
    #     # get ee and puck position
    #     ee_pos = self.ee_pos[:2]
    #     puck_pos = self.puck_pos[:2]
    #
    #     ''' REWARD HIT '''
    #     # goal case, default option
    #     if info["success"]:
    #         reward_hit = self.large_reward
    #     # no hit case
    #     elif not self.has_hit and not self.hit_reward_given:
    #         reward_hit = - (np.linalg.norm(ee_pos[:2] - puck_pos) / (0.5 * table_diag))
    #     elif self.has_hit and not self.hit_reward_given:
    #         ee_x_vel = self.ee_vel[0]  # retrieve the linear x velocity
    #         reward_hit = 100 + 10 * (ee_x_vel / self.max_vel)
    #         self.hit_reward_given = True
    #     else:
    #         reward_hit = 0
    #     reward_hit -= self.c_r * np.linalg.norm(action)
    #     reward_constraints = self._reward_constraints(info)
    #     return (reward_hit + self.alpha_r * reward_constraints) / 2  # negative rewards should never go below -1



if __name__ == "__main__":
    env = AirHockeyChallengeWrapper(env="3dof-hit")
    env.reset()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.random.uniform(-1, 1, (2, env.env_info['robot']['n_joints'])) * 3
        observation, reward, done, info = env.step(action)
        env.render()
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
