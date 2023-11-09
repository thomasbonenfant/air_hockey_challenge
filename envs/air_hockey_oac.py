import gym
from gym import spaces
import numpy as np
import scipy
from scipy import sparse
import osqp
import copy
from baseline.baseline_agent.baseline_agent import BaselineAgent


from air_hockey_challenge.framework.air_hockey_challenge_wrapper import \
    AirHockeyChallengeWrapper
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, \
    jacobian
from utils.ATACOM_transformation import AtacomTransformation, build_ATACOM_Controller
from pathlib import Path
import yaml
from air_hockey_agent.agent_builder import build_agent

PENALTY_POINTS = {"joint_pos_constr": 2, "ee_constr": 3, "joint_vel_constr": 1, "jerk": 1,
                  "computation_time_minor": 0.5,
                  "computation_time_middle": 1, "computation_time_major": 2}
contraints = PENALTY_POINTS.keys()


class AirHockeyOAC(gym.Env):
    def __init__(self, env, interpolation_order=3,
                 simple_reward=False, high_level_action=True, agent_id=1, delta_action=False, delta_ratio=0.1,
                 jerk_only=False, include_joints=True, shaped_reward=False, clipped_penalty=0.5, large_reward=100,
                 large_penalty=100, min_jerk=10000, max_jerk=10000, alpha_r=1., c_r=0., include_hit=True, history=0,
                 use_atacom=False, stop_after_hit=False, punish_jerk=False, acceleration=False, max_accel=0.2,
                 include_old_action=False, use_aqp=True, aqp_terminates=False, speed_decay=0.5, clip_vel=False,
                 whole_game_reward=False, score_reward=10, fault_penalty=5, load_second_agent=False,
                 dont_include_timer_in_states=False, action_persistence=1, stop_when_puck_otherside=False,
                 curriculum_learning_step1=False, curriculum_learning_step2=False, curriculum_learning_step3=False
                 , start_from_defend=False, original_env=False, start_curriculum_transition=3000,
                 end_curriculum_transition=4000, curriculum_transition=False, max_path_length=400, **kwargs):
        self._env = AirHockeyChallengeWrapper(env=env, interpolation_order=interpolation_order, **kwargs)
        self.interpolation_order = interpolation_order
        self.env_label = env
        self.env_info = env_info = self._env.env_info
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        self.puck_radius = env_info["puck"]["radius"]
        self.mallet_radius = env_info["mallet"]["radius"]
        self.dt = self.env_info['dt']
        ac_space = self.env_info["rl_info"].action_space
        obs_space = self.env_info["rl_info"].observation_space
        self.gamma = self.env_info["rl_info"].gamma
        self.simple_reward = simple_reward
        self.shaped_reward = shaped_reward
        self.large_reward = large_reward
        self.large_penalty = large_penalty
        self.min_jerk = min_jerk
        self.max_jerk = max_jerk
        self.alpha_r = alpha_r
        self.c_r = c_r
        self.jerk_only = jerk_only
        self.include_hit = include_hit
        self.high_level_action = high_level_action
        self.delta_action = delta_action
        self.acceleration = acceleration
        self.delta_ratio = delta_ratio
        self.max_accel = max_accel
        self.use_aqp = use_aqp
        self.aqp_terminates = aqp_terminates
        self.aqp_failed = False
        self.include_joints = include_joints
        self.dont_include_timer_in_states = dont_include_timer_in_states
        self.desired_height = self.env_info['robot']['ee_desired_height']
        self.low_position = np.array([0.54, -0.5, 0])
        self.high_position = np.array([1.5, 0.5, 0.3])
        self.goal = None
        self.clip_penalty = clipped_penalty
        self.puck_pos_ids = puck_pos_ids = self.env_info["puck_pos_ids"]
        self.puck_vel_ids = puck_vel_ids = self.env_info["puck_vel_ids"]
        self.history = history
        self.include_old_action = include_old_action
        self.use_atacom = use_atacom
        self.stop_after_hit = stop_after_hit
        self.punish_jerk = punish_jerk
        self.speed_decay = np.clip(speed_decay, 0.05, 0.95)
        self.clip_vel = clip_vel
        self.whole_game_reward = whole_game_reward
        self.score_reward = score_reward
        self.fault_penalty = fault_penalty
        self.last_scores = [0, 0]
        self.last_faults = [0, 0]
        self.action_persistence = action_persistence
        self.stop_when_puck_otherside = stop_when_puck_otherside

        self.max_path_length = max_path_length

        self.curriculum_learning_step1 = curriculum_learning_step1
        self.curriculum_learning_step2 = curriculum_learning_step2
        self.curriculum_learning_step3 = curriculum_learning_step3

        self.start_from_defend = start_from_defend
        if self.action_persistence > 0:
            self.horizon = self._env._mdp_info.horizon * (self.action_persistence + 1)
        else:
            self.horizon = self._env._mdp_info.horizon
        # if load_second_agent:
        #     print("must load") # TODO load the agent here
        # else:

        self.epoch_number = None
        self.curriculum_transition = curriculum_transition
        if self.curriculum_transition:
            self.start_curriculum_transition = start_curriculum_transition
            self.end_curriculum_transition = end_curriculum_transition
            self.duration_curriculum_transition = end_curriculum_transition - start_curriculum_transition
            # 9 is bcs of -4.5 and +4.5 for sigmoid function, which is 0.01 and 0.99 respectively
            self.transition_step_size = 9 / self.duration_curriculum_transition

        if load_second_agent and original_env:
            agent_config_path = Path(__file__).parent.joinpath(
                "envs/air_hockey_challenge/air_hockey_agent/agent_config.yml")
            agent_config_path = ("/home/amirhossein/Research codes/oac-explore/envs/air_hockey_challenge"
                                 "/air_hockey_agent/agent_config.yml")
            with open(agent_config_path) as stream:
                agent_config = yaml.safe_load(stream)

            self.second_agent = build_agent(self._env.env_info, **agent_config)
            print("load second agent here")

        elif self.env_label == "tournament":
            self.second_agent = BaselineAgent(self._env.env_info, 2)

            self.second_agent_obs = None

            self.action_idx = (np.arange(self._env.base_env.action_shape[0][0]),
                               np.arange(self._env.base_env.action_shape[1][0]))

        joint_pos_ids = self.env_info["joint_pos_ids"]
        low_joints_pos = self.env_info["rl_info"].observation_space.low[joint_pos_ids]
        high_joints_pos = self.env_info["rl_info"].observation_space.high[joint_pos_ids]
        joint_pos_norm = high_joints_pos - low_joints_pos
        joint_vel_ids = self.env_info["joint_vel_ids"]

        low_joints_vel = self.env_info["rl_info"].observation_space.low[joint_vel_ids]
        high_joints_vel = self.env_info["rl_info"].observation_space.high[joint_vel_ids]
        self.low_joints_vel = 0.9 * low_joints_vel
        self.high_joints_vel = 0.9 * high_joints_vel

        low_joints_acc = self.env_info["robot"]["joint_acc_limit"][0, :]
        high_joints_acc = self.env_info["robot"]["joint_acc_limit"][1, :]
        self.low_joints_acc = 0.9 * low_joints_acc
        self.high_joints_acc = 0.9 * high_joints_acc

        joint_vel_norm = high_joints_vel - low_joints_vel
        ee_pos_nom = self.high_position - self.low_position
        self.has_hit = False
        self.hit_reward_given = False
        self.defend_reward_given = False
        self.normalizations = {
            'joint_pos_constr': np.concatenate([joint_pos_norm, joint_pos_norm]),
            'joint_vel_constr': np.concatenate([joint_vel_norm, joint_vel_norm]),
            'ee_constr': np.concatenate([ee_pos_nom[:2], ee_pos_nom[:2], ee_pos_nom[:2]])[:5]
        }
        if "hit" in self.env_label:
            self._shaped_r = self._shaped_r_hit
            self.hit_env = True
            self.defend_env = False
        elif "defend" in self.env_label:
            self._shaped_r = self._shaped_r_defend
            self.hit_env = False
            self.defend_env = True
        else:
            self._shaped_r = self._shaped_r_hit
            self.hit_env = False
            self.defend_env = False

        low_position = self.env_info["rl_info"].observation_space.low[puck_pos_ids]
        low_velocity = self.env_info["rl_info"].observation_space.low[puck_vel_ids]
        high_velocity = self.env_info["rl_info"].observation_space.high[puck_vel_ids]
        low_timer = np.array([0.0])
        high_timer = np.array([15.0])
        self.dim = 3 if "3" in self.env_label else 7
        self.max_vel = high_velocity[0]
        if self.high_level_action:
            if self.acceleration:
                low_action = - np.ones(2) * self.max_accel
                high_action = np.ones(2) * self.max_accel
            else:
                low_action = np.array([0.58, -0.45])
                high_action = np.array([1.5, 0.45])
            low_position = np.array([0.54, -0.5, 0])
            high_position = np.array([1.5, 0.5, 0.3])
            if self.include_joints:
                low_state = np.concatenate([low_position[:2], low_velocity[:2], low_position[:2], low_velocity[:2],
                                            low_joints_pos,
                                            low_joints_vel])
                high_state = np.concatenate([high_position[:2], high_velocity[:2], high_position[:2], high_velocity[:2],
                                             high_joints_pos,
                                             high_joints_vel])
            else:
                low_state = np.concatenate([low_position[:2], low_velocity[:2], low_position[:2], low_velocity[:2]])
                high_state = np.concatenate(
                    [high_position[:2], high_velocity[:2], high_position[:2], high_velocity[:2]])
        else:
            low_action = ac_space.low
            high_action = ac_space.high
            low_state = obs_space.low
            high_state = obs_space.high
            idx_to_delete = [2, 5, 20, 21, 22]
            low_position = np.array([0.54, -0.5, 0])
            high_position = np.array([1.5, 0.5, 0.3])
            low_state = np.concatenate([low_state, low_position, low_velocity[:2]])
            high_state = np.concatenate([high_state, high_position, high_velocity[:2]])
        if self.delta_action and not self.acceleration:
            range_action = np.abs(high_action - low_action) * delta_ratio
            low_action = - range_action
            high_action = range_action
        if self.hit_env and self.shaped_reward and self.include_hit:
            low_state = np.concatenate([low_state, np.array([0.])])
            high_state = np.concatenate([high_state, np.array([1.])])

        if 'opponent_ee_ids' in env_info and len(env_info["opponent_ee_ids"]) > 0:
            self.opponent = False
            #self.opponent = True
            #low_state = np.concatenate([low_state, low_position])  # -.1 z of ee
            #high_state = np.concatenate([high_state, high_position])
        else:
            self.opponent = False

        if self.include_old_action:
            low_state = np.concatenate([low_state, low_action])
            high_state = np.concatenate([high_state, high_action])

        if not self.dont_include_timer_in_states:
            low_state = np.concatenate([low_state, low_timer])
            high_state = np.concatenate([high_state, high_timer])

        if self.history > 1:
            low_state = np.tile(low_state, self.history)
            high_state = np.tile(high_state, self.history)
        if self.use_atacom:
            if not self.high_level_action:
                low_action = env_info['robot']['joint_acc_limit'][0]
                high_action = env_info['robot']['joint_acc_limit'][1]
            atacom = build_ATACOM_Controller(env_info, slack_type='soft_corner', slack_tol=1e-06, slack_beta=4)
            self.atacom_transformation = AtacomTransformation(env_info, False, atacom)
            # low_action = low_action[:self.atacom_transformation.ee_pos_dim_out]
            # high_action = high_action[:self.atacom_transformation.ee_pos_dim_out]

        self.action_space = spaces.Box(low=low_action, high=high_action)
        self.observation_space = spaces.Box(low=low_state, high=high_state)
        self.t = 0

        if self.high_level_action:
            self.env_info["new_puck_pos_ids"] = [0, 1]
            self.env_info["new_puck_vel_ids"] = [2, 3]
            self.env_info["ee_pos_ids"] = [4, 5]
            self.env_info["ee_vel_ids"] = [6, 7]
            if self.include_joints:
                self.env_info["new_joint_pos_ids"] = [8, 9, 10]
                self.env_info["new_joint_vel_ids"] = [11, 12, 13]
        else:
            self.env_info["new_puck_pos_ids"] = self.env_info["puck_pos_ids"]
            self.env_info["new_puck_vel_ids"] = self.env_info["puck_vel_ids"]
            self.env_info["new_joint_pos_ids"] = self.env_info["joint_pos_ids"]
            self.env_info["new_joint_vel_ids"] = self.env_info["joint_vel_ids"]
            self.env_info["ee_pos_ids"] = [-4, -3]
            self.env_info["ee_vel_ids"] = [-2, -1]
            self.env_info["ee_vel_ids"] = [-2, -1]

        self._state_queue = []
        self.np_random = np.random
        self.old_action = np.zeros_like(low_action)
        self.state = self.reset()

    def epoch_number_setter(self, value):
        self.epoch_number = value

    def _reward_constraints(self, info):
        reward_constraints = 0
        penalty_sums = 0
        for constr in ['joint_pos_constr', 'joint_vel_constr', 'ee_constr']:
            slacks = info[constr]
            norms = self.normalizations[constr]
            slacks[slacks < 0] = 0
            slacks /= norms
            reward_constraints += PENALTY_POINTS[constr] * np.mean(slacks)
            penalty_sums += PENALTY_POINTS[constr]
        if self.punish_jerk:
            jerk = (np.clip(np.array(info["jerk"]), self.min_jerk, self.max_jerk + self.min_jerk) - self.min_jerk) / \
                   self.max_jerk
            reward_constraints += PENALTY_POINTS["jerk"] * np.mean(jerk)
            penalty_sums += PENALTY_POINTS["jerk"]
        reward_constraints = - reward_constraints / penalty_sums
        return reward_constraints

    def _action_transform(self, action):
        command = np.concatenate([action, np.array([self.desired_height])])
        self.command = command  # for analysis

        if self.use_aqp:
            success, joint_velocities = self.solve_aqp(command, self.joint_pos, 0)
            if not self.use_atacom:
                new_joint_pos = self.joint_pos + (self.joint_vel + joint_velocities) / 2 * self.dt
        else:
            success, new_joint_pos = inverse_kinematics(self.robot_model, self.robot_data, command)
            joint_velocities = (new_joint_pos - self.joint_pos) / self.env_info['dt']
        if not success:
            self._fail_count += 1
        if not self.use_atacom:
            action = np.vstack([new_joint_pos, joint_velocities])
        else:
            return joint_velocities

        return action

    def _shaped_r_defend(self, action, info):
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
        r = 0

        if info["success"]:
            r = self.large_reward

        if not self.puck_is_in_otherside and self.shaped_reward:
            # if not self.has_hit and not self.hit_reward_given:
            #     r = - (np.linalg.norm(ee_pos[:2] - puck_pos) / (0.5 * table_diag))
            if self.has_hit and not self.hit_reward_given:
                # ee_x_vel = np.abs(self.ee_vel[0])  # retrieve the linear x velocity
                puck_vel_norm = np.linalg.norm(self.puck_vel[:2])
                max_norm = 28.5
                # r += 1000 - 10 * (ee_x_vel / self.max_vel)
                # r += 1000 + (10 * (self.max_vel - 20 * ee_x_vel))
                # r += 50 + 300 * puck_vel_norm
                r += 50 + 500 * self.puck_vel[0]

                self.hit_reward_given = True

        # in case of goal #fixme all must be like this
        if (np.abs(self.puck_pos[1]) - self.env_info['table']['goal_width'] / 2) <= 0:
            if (self.puck_pos[0] - 1.51) > self.env_info['table']['length'] / 2:
                if not self.start_from_defend:
                    # r += self.score_reward
                    r += self.large_reward
            if (self.puck_pos[0] - 1.51) < -self.env_info['table']['length'] / 2:
                if self.curriculum_learning_step2 or self.start_from_defend:
                    r -= self.large_reward

        if self.curriculum_learning_step3:  # fixme add the variable before the rewards to shift slowly
            r -= self.c_r * np.linalg.norm(action)
            reward_constraints = self._reward_constraints(info)
        else:
            reward_constraints = 0
        return (r + self.alpha_r * reward_constraints) / 2

    def _shaped_r_prepare(self, action, info):
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
        # compute table diagonal
        table_length = self.env_info['table']['length']
        table_width = self.env_info['table']['width']
        table_diag = np.sqrt(table_length ** 2 + table_width ** 2)

        # get ee and puck position
        ee_pos = self.ee_pos[:2]
        puck_pos = self.puck_pos[:2]

        ''' REWARD HIT '''
        # goal case, default option
        if info["success"]:
            reward_hit = self.large_reward
        # no hit case
        elif not self.has_hit and not self.hit_reward_given:
            reward_hit = - (np.linalg.norm(ee_pos[:2] - puck_pos) / (0.5 * table_diag))
        elif self.has_hit and not self.hit_reward_given:
            ee_x_vel = self.ee_vel[0]  # retrieve the linear x velocity
            reward_hit = 100 + 10 * (ee_x_vel / self.max_vel)
            self.hit_reward_given = True
        else:
            reward_hit = 0
        reward_hit -= self.c_r * np.linalg.norm(action)
        reward_constraints = self._reward_constraints(info)
        return (reward_hit + self.alpha_r * reward_constraints) / 2  # negative rewards should never go below -1

    def _shaped_r_hit(self, action, info):
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
        # compute table diagonal
        table_length = self.env_info['table']['length']
        table_width = self.env_info['table']['width']
        table_diag = np.sqrt(table_length ** 2 + table_width ** 2)

        # get ee and puck position
        ee_pos = self.ee_pos[:2]
        puck_pos = self.puck_pos[:2]

        ''' REWARD HIT '''
        # goal case, default option

        reward_hit = 0
        if self.has_hit:
            if self.puck_vel[0] > 0:
                reward_hit = 1000 * (self.puck_vel[0] / self.max_vel)
                #reward = self.hit_coeff * (np.linalg.norm(self.puck_vel[:2]) / self.max_vel)
        else:
            reward_hit = - (np.linalg.norm(self.ee_pos[:2] - self.puck_pos[:2]) / (0.5 * table_diag))

        reward_hit -= self.c_r * np.linalg.norm(action)
        reward_constraints = self._reward_constraints(info)
        return (reward_hit + self.alpha_r * reward_constraints) / 2  # negative rewards should never go below -1

    def _game_reward(self, action, info):
        r = 0
        scores = info['score']
        faults = info['faults']

        # compute table diagonal
        table_length = self.env_info['table']['length']
        table_width = self.env_info['table']['width']
        table_diag = np.sqrt(table_length ** 2 + table_width ** 2)

        # get ee and puck position
        ee_pos = self.ee_pos[:2]
        puck_pos = self.puck_pos[:2]

        if not self.puck_is_in_otherside and self.shaped_reward:
            if not self.has_hit and not self.hit_reward_given:
                r = - (np.linalg.norm(ee_pos[:2] - puck_pos) / (0.5 * table_diag))
            elif self.has_hit and not self.hit_reward_given:
                ee_x_vel = self.ee_vel[0]  # retrieve the linear x velocity
                r = 100 + 10 * (ee_x_vel / self.max_vel)
                self.hit_reward_given = True

        # in case of goal #fixme all must be like this
        if (np.abs(self.puck_pos[1]) - self.env_info['table']['goal_width'] / 2) <= 0:
            if (self.puck_pos[0] - 1.51) > self.env_info['table']['length'] / 2:
                if not self.start_from_defend:
                    r += self.score_reward
            if (self.puck_pos[0] - 1.51) < -self.env_info['table']['length'] / 2:
                if self.curriculum_learning_step2 or self.start_from_defend:
                    r -= self.score_reward


        self.last_scores = copy.deepcopy(scores)

        if self._fault_changed(faults):
            if faults[0] > self.last_faults[0]:
                r -= self.fault_penalty
        self.last_faults = copy.deepcopy(faults)
        r -= self.c_r * np.linalg.norm(action)
        reward_constraints = self._reward_constraints(info)
        return (r + self.alpha_r * reward_constraints) / 2

    def _game_reward_curriculum(self, action, info):
        r = 0
        scores = info['score']
        faults = info['faults']
        # compute table diagonal
        info['score deviation'] = scores[0] - scores[1]
        info['faults deviation'] = faults[0] - faults[1]
        info['agent1 score'] = scores[0]
        info['agent2 score'] = scores[1]

        table_length = self.env_info['table']['length']
        table_width = self.env_info['table']['width']
        table_diag = np.sqrt(table_length ** 2 + table_width ** 2)

        # get ee and puck position
        ee_pos = self.ee_pos[:2]
        puck_pos = self.puck_pos[:2]

        if not self.puck_is_in_otherside and self.shaped_reward:
            if not self.has_hit and not self.hit_reward_given:
                r = - (np.linalg.norm(ee_pos[:2] - puck_pos) / (0.5 * table_diag))
            if self.has_hit and not self.hit_reward_given:
                ee_x_vel = self.ee_vel[0]  # retrieve the linear x velocity
                r = 100 + 100 * (ee_x_vel / self.max_vel)
                self.hit_reward_given = True

        # in case of goal #fixme all must be like this
        if (np.abs(self.puck_pos[1]) - self.env_info['table']['goal_width'] / 2) <= 0:
            if (self.puck_pos[0] - 1.48) > self.env_info['table']['length'] / 2:
                if not self.start_from_defend:
                    r += self.score_reward
            if (self.puck_pos[0] - 1.51) < -self.env_info['table']['length'] / 2:
                if self.curriculum_learning_step2 or self.start_from_defend:
                    r -= self.score_reward

        self.last_scores = copy.deepcopy(scores)

        if self.curriculum_learning_step3 and self._fault_changed(faults):
            if faults[0] > self.last_faults[0]:
                r -= self.fault_penalty
        self.last_faults = copy.deepcopy(faults)
        if self.curriculum_learning_step3:# fixme add the variable before the rewards to shift slowly
            r -= self.c_r * np.linalg.norm(action)
            reward_constraints = self._reward_constraints(info)
        else:
            reward_constraints = 0
        return (r + self.alpha_r * reward_constraints) / 2

    def _game_reward_curriculum_defend(self, action, info):
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
        r = 0
        scores = info['score']
        faults = info['faults']
        # compute table diagonal
        info['score deviation'] = scores[0] - scores[1]
        info['faults deviation'] = faults[0] - faults[1]
        info['agent1 score'] = scores[0]
        info['agent2 score'] = scores[1]

        table_length = self.env_info['table']['length']
        table_width = self.env_info['table']['width']
        table_diag = np.sqrt(table_length ** 2 + table_width ** 2)

        # get ee and puck position
        ee_pos = self.ee_pos[:2]
        puck_pos = self.puck_pos[:2]

        if np.abs(self.puck_vel[0]) < 0.1 and self.has_hit and not self.defend_reward_given and not self.puck_is_in_otherside:
            r += self.large_reward
            self.defend_reward_given = True
        elif (np.abs(self.puck_vel[0]) < 0.1 and
              (np.abs(self.puck_vel[1])) < 0.1 and
              self.has_hit and not self.defend_reward_given and not self.puck_is_in_otherside):
            r += 2 * self.large_reward
            self.defend_reward_given = True

        elif self.has_hit and self.puck_is_in_otherside:
            r = -self.large_penalty


        # if 0.71 < self.puck_pos[0] <= 1.31 and np.abs(self.puck_vel[0]) < 0.1 and self.has_hit and not self.defend_reward_given:
        #     r += self.large_reward
        #     self.defend_reward_given = True
        # elif np.abs(self.puck_vel[0]) < 0.1 and self.has_hit:
        #     r = -self.large_penalty
        # elif self.has_hit and self.puck_is_in_otherside:
        #     r = -self.large_penalty

        # if self.defend_reward_given:
        #     print(f"puck pose 0 : {self.puck_pos[0]} puck vel 0 : {np.abs(self.puck_vel[0])} and has hit {self.has_hit}")

        if not self.puck_is_in_otherside and self.shaped_reward:
            # if not self.has_hit and not self.hit_reward_given:
            #     r = - (np.linalg.norm(ee_pos[:2] - puck_pos) / (0.5 * table_diag))
            if self.has_hit and not self.hit_reward_given:
                # ee_x_vel = np.abs(self.ee_vel[0])  # retrieve the linear x velocity
                puck_vel_norm = np.linalg.norm(self.puck_vel[:2])
                max_norm = 28.5
                # r += 1000 - 10 * (ee_x_vel / self.max_vel)
                # r += 1000 + (10 * (self.max_vel - 20 * ee_x_vel))
                r += 1000 - 300 * puck_vel_norm

                self.hit_reward_given = True

        # in case of goal #fixme all must be like this
        if (np.abs(self.puck_pos[1]) - self.env_info['table']['goal_width'] / 2) <= 0:
            if (self.puck_pos[0] - 1.51) > self.env_info['table']['length'] / 2:
                if not self.start_from_defend:
                    # r += self.score_reward
                    r += self.large_reward
            if (self.puck_pos[0] - 1.51) < -self.env_info['table']['length'] / 2:
                if self.curriculum_learning_step2 or self.start_from_defend:
                    r -= self.score_reward

        self.last_scores = copy.deepcopy(scores)

        if self.curriculum_learning_step3 and self._fault_changed(faults):
            if faults[0] > self.last_faults[0]:
                r -= self.fault_penalty
        self.last_faults = copy.deepcopy(faults)
        if self.curriculum_learning_step3:  # fixme add the variable before the rewards to shift slowly
            r -= self.c_r * np.linalg.norm(action)
            reward_constraints = self._reward_constraints(info)
        else:
            reward_constraints = 0
        return (r + self.alpha_r * reward_constraints) / 2

    def _score_changed(self, scores):
        previous_scores = self.last_scores
        for i in range(len(scores)):
            if previous_scores[i] != scores[i]:
                # self.last_scores = scores
                return True
        # self.last_scores = scores
        return False

    def _fault_changed(self, faults):
        previous_faults = self.last_faults
        for i in range(len(faults)):
            if previous_faults[i] != faults[i]:
                # self.last_faults = faults
                return True
        # self.last_faults = faults
        return False

    def _post_simulation(self, obs):
        self._obs = obs
        self.puck_pos = self.get_puck_pos(obs)
        self.pre_previous_vel = self.previous_vel if self.t > 1 else None
        self.previous_vel = self.puck_vel if self.t > 0 else None
        self.puck_vel = self.get_puck_vel(obs)
        self.joint_pos = self.get_joint_pos(obs)
        self.joint_vel = self.get_joint_vel(obs)
        self.previous_ee_pos = self.ee_pos if self.t > 0 else None
        self.ee_pos = self.get_ee_pose(obs)
        if self.opponent:
            self.opponent_ee_pos = self.get_opponent_ee_pose(obs)
        self.ee_vel = self._apply_forward_velocity_kinematics(self.joint_pos, self.joint_vel)
        # if self.previous_vel is not None:
        #     previous_vel_norm = np.linalg.norm(self.previous_vel[:2])
        #     current_vel_norm = np.linalg.norm(self.puck_vel[:2])
        #     distance = np.linalg.norm(self.puck_pos[:2] - self.ee_pos[:2])
        #
        #     # if previous_vel_norm <= current_vel_norm and distance <= (self.puck_radius + self.mallet_radius) * 1.1:
        #     #     self.has_hit = True
        #     if np.abs(previous_vel_norm - current_vel_norm) > 0.4 and distance <= (self.puck_radius + self.mallet_radius) * 1.1:
        #         self.has_hit = True

        if self.pre_previous_vel is not None:
            previous_vel_norm = np.linalg.norm(self.previous_vel[:2])
            current_vel_norm = np.linalg.norm(self.puck_vel[:2])
            distance = np.linalg.norm(self.puck_pos[:2] - self.ee_pos[:2])

            previous_delta_vel = np.sign(self.previous_vel[:2] - self.pre_previous_vel[:2])
            current_delta_vel = np.sign(self.puck_vel[:2] - self.previous_vel[:2])

            if (np.abs(previous_vel_norm - current_vel_norm) > 0.4 or
                not np.array_equal(previous_delta_vel, current_delta_vel)) and distance <= (self.puck_radius + self.mallet_radius) * 1.2:
                self.has_hit = True
                # print("hit True")

        if self.puck_pos[0] < 1.51 and (self.puck_is_in_otherside or self.t == 0):  # puck is in our side
            self.puck_is_in_otherside = False
            self.hit_reward_given = False
            self.defend_reward_given = False
            self.has_hit = False
        elif self.puck_pos[0] < 1.51:
            self.puck_is_in_otherside = False
        else:
            self.puck_is_in_otherside = True

    def _process_info(self, info):
        if self.env_label == "tournament":
            info["joint_pos_constr"] = info["constraints_value"][0]["joint_pos_constr"]
            info["joint_vel_constr"] = info["constraints_value"][0]["joint_vel_constr"]
            info["ee_constr"] = info["constraints_value"][0]["ee_constr"]
            info.pop('constraints_value', None)
        else:
            info["joint_pos_constr"] = info["constraints_value"]["joint_pos_constr"]
            info["joint_vel_constr"] = info["constraints_value"]["joint_vel_constr"]
            info["ee_constr"] = info["constraints_value"]["ee_constr"]
            info.pop('constraints_value', None)
        return info

    def _reward(self, action, done, info):
        if self.simple_reward:
            r = info["success"]
        elif self.shaped_reward and not self.whole_game_reward:
            r = self._shaped_r(action, info)
            # r = self._shaped_r(action, info)
        elif self.whole_game_reward and not self.curriculum_learning_step1:
            r = self._game_reward(action, info)
        elif self.whole_game_reward and self.curriculum_learning_step1 and not self.start_from_defend:
            r = self._game_reward_curriculum(action, info)
        elif self.whole_game_reward and self.curriculum_learning_step1 and self.start_from_defend:
            r = self._game_reward_curriculum_defend(action, info)
        else:
            r = 0
            if not self.jerk_only:
                for constr in ['joint_pos_constr', 'joint_vel_constr', 'ee_constr']:
                    if np.any(np.array(info[constr] > 0)):
                        r -= PENALTY_POINTS[constr]
            if np.any(np.array(info["jerk"] > self.min_jerk)):
                r -= PENALTY_POINTS["jerk"] + \
                     np.clip(np.array(info["jerk"]), self.min_jerk, self.max_jerk).mean() / self.max_jerk
            r += info["success"] * self.large_reward
        # if done and not info["success"] and self.t < self._env._mdp_info.horizon:
        # todo in bemune?
        # if not self.start_from_defend and done and self.t < self._env._mdp_info.horizon:
        if not self.start_from_defend and done and self.t < self.horizon:
            r -= self.large_penalty

        # if self.start_from_defend and not done and self.t == self._env._mdp_info.horizon:
        # if self.start_from_defend and not done and self.t == self.horizon:
        if not done and self.t == self.horizon:
            r -= self.large_penalty

        if done and self.puck_is_in_otherside:
            r -= self.large_penalty

        if self.high_level_action:
            if self.clipped_state:
                r -= self.clip_penalty
            if self.use_aqp and self.aqp_failed and self.aqp_terminates:
                r -= self.large_penalty
        # if r!=0:
        #     print(f"{r} : khar")
        return r

    def solve_aqp(self, x_des, q_cur, dq_anchor):
        robot_model = self.robot_model
        robot_data = self.robot_data
        joint_vel_limits = self.env_info['robot']['joint_vel_limit']
        joint_pos_limits = self.env_info['robot']['joint_pos_limit']
        dt = self.dt
        n_joints = self.dim

        if n_joints == 3:
            anchor_weights = np.ones(3)
        else:
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
            self.aqp_failed = True
            return False, b

        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=l, u=u, verbose=False, polish=False)

        result = solver.solve()
        if result.info.status == 'solved':
            return True, N_J @ result.x + b
        else:
            return False, b

    def _process_action(self, action):
        if self.delta_action and not self.acceleration:
            if self.high_level_action:
                ee_pos = self.ee_pos
                action = ee_pos[:2] + action[:2]
            else:
                joint_pos = self.joint_pos
                joint_vel = self.joint_vel
                action = np.concatenate([joint_pos, joint_vel]) + action
        if self.high_level_action:
            if self.acceleration:
                ee_pos = self.ee_pos
                # action = np.array([2., 1.])
                ee_vel = self.ee_vel[:2] + self.dt * action[:2]
                delta_pos = ee_vel * self.dt

                action = ee_pos[:2] + delta_pos[:2]
            tolerance = 0.0065
            padding = np.array([self.puck_radius + tolerance, self.puck_radius + tolerance])
            low_clip = self.low_position[:2] + padding
            high_clip = self.high_position[:2] - padding
            action_clipped = np.clip(action, a_min=low_clip, a_max=high_clip)
            if np.any(action_clipped - action > 1e-6):
                self.clipped_state = True
            else:
                self.clipped_state = False
            action = action_clipped
            action = self._action_transform(action[:2])
        else:
            action = np.reshape(action, (2, -1))
        if self.clip_vel:
            joint_vel = action[1, :]
            new_joint_vel = np.clip(joint_vel, self.low_joints_vel, self.high_joints_vel)
            new_joint_pos = self.joint_pos + (self.joint_vel + new_joint_vel) / 2 * self.dt
            action = np.vstack([new_joint_pos, new_joint_vel])
        return action

    def _atacom_plus_aqp_action_transform(self, action):
        if self.delta_action and not self.acceleration:
            if self.high_level_action:
                ee_pos = self.ee_pos
                action = ee_pos[:2] + action[:2]
            else:
                joint_pos = self.joint_pos
                joint_vel = self.joint_vel
                action = np.concatenate([joint_pos, joint_vel]) + action
        if self.high_level_action:
            if self.acceleration:
                ee_pos = self.ee_pos
                # action = np.array([2., 1.])
                ee_vel = self.ee_vel[:2] + self.dt * action[:2]
                delta_pos = ee_vel * self.dt

                action = ee_pos[:2] + delta_pos[:2]
            tolerance = 0.0065
            padding = np.array([self.puck_radius + tolerance, self.puck_radius + tolerance])
            low_clip = self.low_position[:2] + padding
            high_clip = self.high_position[:2] - padding
            action_clipped = np.clip(action, a_min=low_clip, a_max=high_clip)
            if np.any(action_clipped - action > 1e-6):
                self.clipped_state = True
            else:
                self.clipped_state = False
            action = action_clipped
            new_joint_vel = self._action_transform(action[:2])
        else:
            action = np.reshape(action, (2, -1))
        if self.clip_vel:
            joint_vel = action
            new_joint_vel = np.clip(joint_vel, self.low_joints_vel, self.high_joints_vel)

        current_joint_vel = self.joint_vel

        acceleration = (new_joint_vel - current_joint_vel) / self.dt
        new_acc = np.clip(acceleration, self.low_joints_acc, self.high_joints_acc)
        action = new_acc

        return action

    def _get_state(self, obs):
        ee_pos = self.ee_pos
        ee_vel = self.ee_vel
        if self.high_level_action:
            puck_pos = self.puck_pos
            puck_vel = self.puck_vel
            state = np.concatenate([puck_pos[:2], puck_vel[:2], ee_pos[:2], ee_vel[:2]])
            if self.include_joints:
                joint_pos = self.joint_pos
                joint_vel = self.joint_vel
                state = np.concatenate([state, joint_pos, joint_vel])
        else:
            state = np.concatenate([obs, ee_pos, ee_vel[:2]])
        if self.hit_env and self.shaped_reward and self.include_hit:
            state = np.concatenate([state, np.array([self.has_hit])])
        if self.opponent:
            state = np.concatenate([state, self.opponent_ee_pos])

        if self.include_old_action:
            state = np.concatenate([state, self.old_action])

        if not self.dont_include_timer_in_states:
            state = np.concatenate([state, [self._env.base_env.timer]])

        if self.history > 1:
            self._state_queue.append(state)
            if self.t == 0:
                for i in range(self.history - len(self._state_queue)):
                    self._state_queue.append(state)
            if len(self._state_queue) > self.history:
                self._state_queue.pop(0)
            state = np.concatenate(self._state_queue)
        return state

    def _apply_forward_velocity_kinematics(self, joint_pos, joint_vel):
        robot_model = self.env_info['robot']['robot_model']
        robot_data = self.env_info['robot']['robot_data']
        jac = jacobian(robot_model, robot_data, joint_pos)
        jac = jac[:3]  # last part of the matrix is about rotation. no need for it
        ee_vel = jac @ joint_vel
        return ee_vel

    def seed(self, seed=None):
        return self._env.seed(seed)

    def get_puck_pos(self, obs):
        """
        Get the Puck's position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            Puck's position of the robot

        """
        return obs[self.env_info['puck_pos_ids']]

    def get_puck_vel(self, obs):
        """
        Get the Puck's velocity from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            Puck's velocity of the robot

        """
        return obs[self.env_info['puck_vel_ids']]

    def get_joint_pos(self, obs):
        """
        Get the joint position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint position of the robot

        """
        return obs[self.env_info['joint_pos_ids']]

    def get_joint_vel(self, obs):
        """
        Get the joint velocity from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint velocity of the robot

        """
        return obs[self.env_info['joint_vel_ids']]

    def get_ee_pose(self, obs):
        """
        Get the End-Effector's Position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            end-effector's position

        """
        res = forward_kinematics(self.robot_model, self.robot_data, self.get_joint_pos(obs))
        return res[0]

    def get_opponent_ee_pose(self, obs):
        """
        Get the Opponent's End-Effector's Position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            opponent's end-effector's position

        """
        return obs[self.env_info['opponent_ee_ids']]

    def step(self, action):
        obs, reward, done, info = self._step(action)
        self.old_action = action

        if self.aqp_failed and self.aqp_terminates:
            done = True


        if self.has_hit and self.stop_after_hit and not done:
            if self.delta_action or self.use_atacom or self.acceleration:
                action *= self.speed_decay
            r = reward
            discount = 1
            # while not done and self.t <= self._env._mdp_info.horizon:
            while not done and self.t <= self.horizon:

                discount *= self.gamma
                obs, reward, done, info = self._step(action)
                # obs_1, obs_2 = np.split(obs, 2)
                #
                # obs = obs_1
                # self.second_agent_obs = obs_2

                r += discount * reward
            reward = r
            done = True

        if self.puck_is_in_otherside and self.stop_when_puck_otherside and not done:
            if self.delta_action or self.use_atacom or self.acceleration:
                # action *= self.speed_decay
                action *= 0

            r = reward
            discount = 1
            # while not done and self.t <= self._env._mdp_info.horizon and self.puck_is_in_otherside:
            while not done and self.t <= self.horizon and self.puck_is_in_otherside:

                discount *= self.gamma
                obs, reward, done, info = self._step(action)
                # return obs, reward, done, info # fixme this line is just for visualization

                r += discount * reward
            reward = r
            # fixme check if the tuple is ok?
        # if reward != 0:
        #   print(reward)
        return obs, reward, done, info

    def _step(self, action):
        initial_action = copy.deepcopy(action)
        if self.use_atacom:
            if self.use_aqp:
                action = self._atacom_plus_aqp_action_transform(action)
                action = self.atacom_transformation.draw_action(self._obs, action)
                self.clipped_state = False
            else:
                # action = self.atacom_transformation.draw_action(action, self.joint_pos, self.joint_vel)
                action = self.atacom_transformation.draw_action(self._obs, action)
                self.clipped_state = False
        else:
            action = self._process_action(action)
        if self.interpolation_order in [1, 2]:
            _action = action.flatten()
        else:
            _action = action

        first_agent_action = _action

        if self.env_label == "tournament":
            second_agent_action = self.second_agent.draw_action(self.second_agent_obs)

            dual_action = (first_agent_action, second_agent_action)

            obs, reward, done, info = self._env.step((dual_action[0][self.action_idx[0]],
                                                      dual_action[1][self.action_idx[1]]))

            obs_1, obs_2 = np.split(obs, 2)

            obs = obs_1
            self.second_agent_obs = obs_2
        else:
            obs, reward, done, info = self._env.step(_action)

        #
        # if info["success"] and not done:
        #     info["success"] = 0

        self._post_simulation(obs)
        if (not self.start_from_defend and self.curriculum_learning_step1 and not self.curriculum_learning_step2
        and self.puck_vel[0] < 0):
            done = True
            # print("1")
            # #fixme
            # done = False
        # if (self.start_from_defend and self.curriculum_learning_step1 and not self.curriculum_learning_step2
        # and np.abs(self.puck_vel[0]) < 0.1 and not self.puck_is_in_otherside):
        if (self.start_from_defend and self.curriculum_learning_step1 and not self.curriculum_learning_step2
            and np.abs(self.puck_vel[0]) < 0.1 and self.has_hit):
            done = True
            # print("2")

            # print(f"puck vel x axis : {np.abs(self.puck_vel[0])}")
            # #fixme
            # done = False
            # print("done kir khar")
        if (self.start_from_defend and self.curriculum_learning_step1 and not self.curriculum_learning_step2
                and self.puck_is_in_otherside and self.has_hit):
            done = True
            # print("3")

            # #fixme
            # done = False
        info = self._process_info(info)
        r = self._reward(action, done, info)
        # # print(r)
        # if r!=0:
        #     print(r)
        obs = self._get_state(obs)
        self.state = obs
        self.t += 1

        discount = 1
        if self.action_persistence > 0:
            k = 0
            # if (not done and self.t <= self._env._mdp_info.horizon and k < self.action_persistence):
            #     print("eshak")
            # else:
            #     print("kheily khari")
            # while not done and self.t <= self._env._mdp_info.horizon and k < self.action_persistence:
            while not done and self.t <= self.horizon and k < self.action_persistence:

                # discount *= self.gamma # fixme
                action = initial_action
                if self.use_atacom:
                    if self.use_aqp:
                        action = self._atacom_plus_aqp_action_transform(action)
                        action = self.atacom_transformation.draw_action(self._obs, action)
                        self.clipped_state = False
                    else:
                        # action = self.atacom_transformation.draw_action(action, self.joint_pos, self.joint_vel)
                        action = self.atacom_transformation.draw_action(self._obs, action)
                        self.clipped_state = False
                else:
                    action = self._process_action(action)
                if self.interpolation_order in [1, 2]:
                    _action = action.flatten()
                else:
                    _action = action
                first_agent_action = _action
                if self.env_label == "tournament":
                    second_agent_action = self.second_agent.draw_action(self.second_agent_obs)

                    dual_action = (first_agent_action, second_agent_action)

                    obs, reward, done, info = self._env.step((dual_action[0][self.action_idx[0]],
                                                              dual_action[1][self.action_idx[1]]))

                    obs_1, obs_2 = np.split(obs, 2)

                    obs = obs_1
                    self.second_agent_obs = obs_2
                else:
                    obs, reward, done, info = self._env.step(_action)#
                # if info["success"] and not done:
                #     info["success"] = 0
                self._post_simulation(obs)
                if (not self.start_from_defend and self.curriculum_learning_step1 and not self.curriculum_learning_step2
                        and self.puck_vel[0] < 0):
                    done = True
                    # print("action pers 1")

                # if (self.start_from_defend and self.curriculum_learning_step1 and not self.curriculum_learning_step2
                # and np.abs(self.puck_vel[0]) < 0.1 and not self.puck_is_in_otherside):
                if (self.start_from_defend and self.curriculum_learning_step1 and not self.curriculum_learning_step2
                        and np.abs(self.puck_vel[0]) < 0.1 and self.has_hit):
                    done = True
                    # print("action pers 2")

                    # print("done kir khar")
                if (self.start_from_defend and self.curriculum_learning_step1 and not self.curriculum_learning_step2
                        and self.puck_is_in_otherside and self.has_hit):
                    done = True
                    # print("action pers 3")


                info = self._process_info(info)
                reward = self._reward(action, done, info)
                # print(r, " inside action pers")
                obs = self._get_state(obs)
                self.state = obs
                self.t += 1

                r += discount * reward
                k += 1

        done = self.has_hit or self.t >= self.max_path_length # ends episode only if reached max_path or hit puck
        reward = r

        return obs, reward, done, info

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # using sigmoid, we will have a smooth transition
    def curriculum_transition_func(self):
        difference_velocity_range = ((self._env.base_env.second_velocity_range - self._env.base_env.first_velocity_range) *
            self.sigmoid(-4.5 + ((self.epoch_number - self.start_curriculum_transition) * self.transition_step_size)))
        self._env.base_env.velocity_range = self._env.base_env.first_velocity_range + difference_velocity_range


        difference_start_range = ((self._env.base_env.second_start_range - self._env.base_env.first_start_range) *
            self.sigmoid(-4.5 + ((self.epoch_number - self.start_curriculum_transition) * self.transition_step_size)))
        self._env.base_env.start_range = self._env.base_env.first_start_range + difference_start_range


        difference_init_ee_range = ((self._env.base_env.second_init_ee_range - self._env.base_env.first_init_ee_range) *
            self.sigmoid(-4.5 + ((self.epoch_number - self.start_curriculum_transition) * self.transition_step_size)))
        self._env.base_env.init_ee_range = self._env.base_env.first_init_ee_range + difference_init_ee_range



        difference_init_angle_range = ((self._env.base_env.second_init_angle_range - self._env.base_env.first_init_angle_range) *
            self.sigmoid(-4.5 + ((self.epoch_number - self.start_curriculum_transition) * self.transition_step_size)))
        self._env.base_env.init_angle_range = self._env.base_env.first_init_angle_range + difference_init_angle_range

    def reset(self):
        self.t = 0
        self.has_hit = False
        self.hit_reward_given = False
        self.defend_reward_given = False
        self._fail_count = 0
        self._state_queue = []
        self.old_action = np.zeros_like(self.old_action)

        if self.curriculum_transition and self.epoch_number is not None and self.epoch_number >= self.start_curriculum_transition:
            self.curriculum_transition_func()
        obs = self._env.reset()

        if self.env_label=="tournament":
            obs_1, obs_2 = np.split(obs, 2)
            obs = obs_1
            self.second_agent_obs = obs_2

        self.puck_is_in_otherside = False
        self._post_simulation(obs)
        self.state = self._get_state(obs)
        if self.use_atacom:
            self.atacom_transformation.reset()

        self.last_scores = [0, 0]
        self.last_faults = [0, 0]

        if self.state[0] < 1.51:  # puck is in our side

            self.puck_is_in_otherside = False
        else:
            self.puck_is_in_otherside = True

        return self.state

    def render(self, mode='human'):
        self._env.render()
