import numpy as np
import copy
import yaml
from yaml.loader import SafeLoader
import scipy
import osqp
from scipy import sparse
import json
# Open the file and load the file
from air_hockey_challenge.utils.kinematics import inverse_kinematics, forward_kinematics, \
    jacobian

from air_hockey_agent.agents.kalman_filter import PuckTracker

from utils.variant_util import env_producer, get_policy_producer, get_q_producer

from utils.ATACOM_transformation import AtacomTransformation, build_ATACOM_Controller
from air_hockey_challenge.framework.agent_base import AgentBase

import os
from utils.pythonplusplus import load_gzip_pickle
from utils.variant_util import build_variant

from trainer.particle_trainer import ParticleTrainer
from trainer.gaussian_trainer import GaussianTrainer
from trainer.gaussian_trainer_soft import GaussianTrainerSoft
from trainer.trainer import SACTrainer

import torch

class Agent(AgentBase):
    def __init__(self, env_info, varient, **kwargs):
        super().__init__(env_info, **kwargs)



        torch.set_num_threads(4)
        self.interpolation_order = varient['interpolation_order']
        self.env_label = env_info['env_name']
        self.env_info = env_info
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        self.puck_radius = env_info["puck"]["radius"]
        self.mallet_radius = env_info["mallet"]["radius"]
        self.dt = self.env_info['dt']
        ac_space = self.env_info["rl_info"].action_space
        obs_space = self.env_info["rl_info"].observation_space
        self.gamma = self.env_info["rl_info"].gamma

        self.restart = True

        self.include_hit = True  # varient['include_hit'] #fixme
        self.high_level_action = varient['high_level_action']
        self.delta_action = varient['delta_action']
        self.acceleration = varient['acceleration']
        self.delta_ratio = varient['delta_ratio']
        self.max_accel = varient['max_accel']
        self.use_aqp = varient['use_aqp']
        self.include_joints = varient['include_joints']
        self.desired_height = self.env_info['robot']['ee_desired_height']
        self.low_position = np.array([0.54, -0.5, 0])
        self.high_position = np.array([1.5, 0.5, 0.3])
        self.goal = None
        self.puck_pos_ids = puck_pos_ids = self.env_info["puck_pos_ids"]
        self.puck_vel_ids = puck_vel_ids = self.env_info["puck_vel_ids"]
        self.history = varient['history']
        self.include_old_action = varient['include_old_action']
        self.use_atacom = varient['use_atacom']
        self.shaped_reward = varient['shaped_reward']

        joint_pos_ids = self.env_info["joint_pos_ids"]
        low_joints_pos = self.env_info["rl_info"].observation_space.low[joint_pos_ids]
        high_joints_pos = self.env_info["rl_info"].observation_space.high[joint_pos_ids]
        joint_pos_norm = high_joints_pos - low_joints_pos
        joint_vel_ids = self.env_info["joint_vel_ids"]
        low_joints_vel = self.env_info["rl_info"].observation_space.low[joint_vel_ids]
        high_joints_vel = self.env_info["rl_info"].observation_space.high[joint_vel_ids]
        joint_vel_norm = high_joints_vel - low_joints_vel
        ee_pos_nom = self.high_position - self.low_position
        self.normalizations = {
            'joint_pos_constr': np.concatenate([joint_pos_norm, joint_pos_norm]),
            'joint_vel_constr': np.concatenate([joint_vel_norm, joint_vel_norm]),
            'ee_constr': np.concatenate([ee_pos_nom[:2], ee_pos_nom[:2], ee_pos_nom[:2]])[:5]
        }
        if "hit" in self.env_label:
            self.hit_env = True
            self.defend_env = False
        elif "defend" in self.env_label:
            self.hit_env = False
            self.defend_env = True
        else:
            self.hit_env = False
            self.defend_env = False

        low_position = self.env_info["rl_info"].observation_space.low[puck_pos_ids]
        low_velocity = self.env_info["rl_info"].observation_space.low[puck_vel_ids]
        high_velocity = self.env_info["rl_info"].observation_space.high[puck_vel_ids]
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
            low_position = np.array([0.54, -0.5, 0])
            high_position = np.array([1.5, 0.5, 0.3])
            low_state = np.concatenate([low_state, low_position, low_velocity[:2]])
            high_state = np.concatenate([high_state, high_position, high_velocity[:2]])
        if self.delta_action and not self.acceleration:
            range_action = np.abs(high_action - low_action) * self.delta_ratio
            low_action = - range_action
            high_action = range_action
        if self.hit_env and self.shaped_reward and self.include_hit:
            low_state = np.concatenate([low_state, np.array([0.])])
            high_state = np.concatenate([high_state, np.array([1.])])

        if 'opponent_ee_ids' in env_info and len(env_info["opponent_ee_ids"]) > 0:
            self.opponent = True
            low_state = np.concatenate([low_state, low_position])  # -.1 z of ee
            high_state = np.concatenate([high_state, high_position])
        else:
            self.opponent = False

        if self.include_old_action:
            low_state = np.concatenate([low_state, low_action])
            high_state = np.concatenate([high_state, high_action])

        if self.history > 1:
            low_state = np.tile(low_state, self.history)
            high_state = np.tile(high_state, self.history)

        if self.use_atacom:
            low_action = env_info['robot']['joint_acc_limit'][0]
            high_action = env_info['robot']['joint_acc_limit'][1]
            if self.dim == 3:
                self.atacom_transformation = AtacomTransformation(env_info, self.get_joint_pos, self.get_joint_vel)
                low_action = low_action[:self.atacom_transformation.ee_pos_dim_out]
                high_action = high_action[:self.atacom_transformation.ee_pos_dim_out]
            else:
                low_action = env_info['robot']['joint_acc_limit'][0]
                high_action = env_info['robot']['joint_acc_limit'][1]
                atacom = build_ATACOM_Controller(env_info, slack_type='soft_corner', slack_tol=1e-06, slack_beta=4)
                self.atacom_transformation = AtacomTransformation(env_info, False, atacom)

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

        self.policy = None





        self.initial_ee_pos = None
        self.initial = True


        self.stop_after_hit_ee_pos = None
        self.stopped = False

        self.puck_tracker = PuckTracker(env_info)
    def _action_transform(self, action):
        command = np.concatenate([action, np.array([self.desired_height])])
        self.command = command  # for analysis

        if self.use_aqp:
            success, joint_velocities = self.solve_aqp(command, self.joint_pos, 0)
            new_joint_pos = self.joint_pos + (self.joint_vel + joint_velocities) / 2 * self.dt
        else:
            success, new_joint_pos = inverse_kinematics(self.robot_model, self.robot_data, command)
            joint_velocities = (new_joint_pos - self.joint_pos) / self.env_info['dt']
        if not success:
            self._fail_count += 1

        action = np.vstack([new_joint_pos, joint_velocities])
        return action

    def _post_simulation(self, obs):
        self.puck_pos = self.get_puck_pos(obs)
        self.previous_vel = self.puck_vel if self.t > 0 else None
        self.puck_vel = self.get_puck_vel(obs)
        self.joint_pos = self.get_joint_pos(obs)
        self.joint_vel = self.get_joint_vel(obs)
        self.previous_ee_pos = self.ee_pos if self.t > 0 else None
        self.ee_pos = self.get_ee_pose(obs)
        if self.opponent:
            self.opponent_ee_pos = self.get_opponent_ee_pose(obs)
        self.ee_vel = self._apply_forward_velocity_kinematics(self.joint_pos, self.joint_vel)
        if self.previous_vel is not None:
            previous_vel_norm = np.linalg.norm(self.previous_vel[:2])
            current_vel_norm = np.linalg.norm(self.puck_vel[:2])
            distance = np.linalg.norm(self.puck_pos[:2] - self.ee_pos[:2])
            if previous_vel_norm <= current_vel_norm and distance <= (self.puck_radius + self.mallet_radius) * 1.1:
                self.has_hit = True

    def _process_info(self, info):
        info["joint_pos_constr"] = info["constraints_value"]["joint_pos_constr"]
        info["joint_vel_constr"] = info["constraints_value"]["joint_vel_constr"]
        info["ee_constr"] = info["constraints_value"]["ee_constr"]
        info.pop('constraints_value', None)
        return info

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
            action_clipped = np.clip(action, a_min=self.low_position[:2], a_max=self.high_position[:2])
            if np.any(action_clipped - action > 1e-6):
                self.clipped_state = True
            else:
                self.clipped_state = False
            action = action_clipped
            action = self._action_transform(action[:2])
        else:
            action = np.reshape(action, (2, -1))
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

    def reset(self):
        self.restart = True
        self.t = 0
        self.has_hit = False
        self.hit_reward_given = False
        self._fail_count = 0
        self._state_queue = []
        self.old_action = np.zeros_like(self.old_action)
        if self.use_atacom:
            self.atacom_transformation.reset()

    def load_agent(self, path, baseline_agent=False, env_label=None, random_agent=False, env_info=None):
        variant = json.load(open(os.path.join(path, 'variant.json')))
        seed = variant['seed']
        domain = variant['domain']
        env_args = {}
        if "hockey" in domain:
            env_args['env'] = variant['hockey_env']
            env_args['simple_reward'] = variant['simple_reward']
            env_args['shaped_reward'] = variant['shaped_reward']
            env_args['large_reward'] = variant['large_reward']
            env_args['large_penalty'] = variant['large_penalty']
            env_args['min_jerk'] = variant['min_jerk']
            env_args['max_jerk'] = variant['max_jerk']
            env_args['history'] = variant['history']
            env_args['use_atacom'] = variant['use_atacom']
            env_args['large_penalty'] = variant['large_penalty']
            env_args['alpha_r'] = variant['alpha_r']
            env_args['c_r'] = variant['c_r']
            env_args['high_level_action'] = variant['high_level_action']
            env_args['clipped_penalty'] = variant['clipped_penalty']
            env_args['include_joints'] = variant['include_joints']
            env_args['jerk_only'] = variant['jerk_only']
            env_args['delta_action'] = variant['delta_action']
            env_args['delta_ratio'] = variant['delta_ratio']
            # env_args['punish_jerk'] = variant['punish_jerk']
            env_args['stop_after_hit'] = False
            env_args['gamma'] = variant['trainer_kwargs']['discount']
            env_args['horizon'] = variant['algorithm_kwargs']['max_path_length'] + 1
            try:
                env_args["acceleration"] = variant["acceleration"]
                env_args['max_accel'] = variant['max_accel']
                env_args['interpolation_order'] = variant['interpolation_order']
                env_args['include_old_action'] = variant['include_old_action']
                env_args['use_aqp'] = variant['use_aqp']
            except:
                env_args['use_aqp'] = True

        if baseline_agent:
            env_args['high_level_action'] = False
            env_args['include_joints'] = True
            env_args['delta_action'] = False
            env_args['use_atacom'] = False

        if env_label is not None:
            env_args['env'] = env_label

        env = env_producer(domain, seed, **env_args)
        obs_dim = env.observation_space.low.size #TODO needs hardcode
        action_dim = env.action_space.low.size #TODO needs hardcode
        # action_space = env.action_space

        # Get producer function for policy and value functions
        M = variant['layer_size']
        N = variant['num_layers']
        if not baseline_agent and not random_agent:
            alg = variant['alg']
            output_size = 1
            q_producer = get_q_producer(obs_dim, action_dim, hidden_sizes=[M] * N, output_size=output_size)
            policy_producer = get_policy_producer(
                obs_dim, action_dim, hidden_sizes=[M] * N)
            q_min = variant['r_min'] / (1 - variant['trainer_kwargs']['discount'])
            q_max = variant['r_max'] / (1 - variant['trainer_kwargs']['discount'])
            alg_to_trainer = {
                'sac': SACTrainer,
                'oac': SACTrainer,
                'p-oac': ParticleTrainer,
                'g-oac': GaussianTrainer,
                'gs-oac': GaussianTrainerSoft
            }
            trainer = alg_to_trainer[variant['alg']]

            kwargs = {}
            if alg in ['p-oac', 'g-oac', 'g-tsac', 'p-tsac', 'gs-oac']:
                kwargs = dict(
                    delta=variant['delta'],
                    q_min=q_min,
                    q_max=q_max,
                )
            kwargs.update(dict(
                policy_producer=policy_producer,
                q_producer=q_producer,
                action_space=env.action_space,
            ))
            kwargs.update(variant['trainer_kwargs'])
            trainer = trainer(**kwargs)

            experiment = path + '/params.zip_pkl'
            exp = load_gzip_pickle(experiment)
            trainer.restore_from_snapshot(exp['trainer'])
        elif baseline_agent:
            from envs.air_hockey_challenge.examples.control.hitting_agent import build_agent
            trainer = build_agent(env.wrapped_env._env.env_info)
            # trainer = build_agent(env_info)

            trainer.reset()
            env = env.wrapped_env
        else:
            trainer = None
        return trainer, env
        # return trainer, None

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

    def draw_action(self, observation):
        # Noise removal

        noisy_puck_pos = self.get_puck_pos(observation)

        if self.restart:
            self.puck_tracker.reset(noisy_puck_pos)

        self.puck_tracker.step(noisy_puck_pos)

        puck_pos = self.puck_tracker.state[[0, 1, 4]].copy()
        puck_vel = self.puck_tracker.state[[2, 3, 5]].copy()

        observation[self.env_info['puck_pos_ids']] = puck_pos

        if not self.restart:
            observation[self.env_info['puck_vel_ids']] = puck_vel

        self.restart = False

        ##

        self._post_simulation(observation)
        self.state = self._get_state(observation)
        self.state = self.env12.to_n1p1(self.state)

        # action, agent_info = self.trainer.target_policy.get_action(self.state, deterministic=True)


        if hasattr(self.trainer, 'target_policy'):
            action, agent_info = self.trainer.target_policy.get_action(self.state, deterministic=True)
        elif hasattr(self.trainer, 'policy'):
            action, agent_info = self.trainer.policy.get_action(self.state, deterministic=True)
        else:
            action, agent_info = self.trainer.policy.get_action(self.state, deterministic=True)


        lb = self.env12._wrapped_env.action_space.low
        ub = self.env12._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        if self.use_atacom:
            action = self.atacom_transformation.draw_action(observation, action)
            self.clipped_state = False
        else:
            action = self._process_action(scaled_action)


        """
        here must be removed 
        """
        # if self.initial:
        #     self.initial_ee_pos = self.get_ee_pose(observation)
        #     self.initial = False
        #
        # action = self.initial_ee_pos
        #
        #
        # action = self._action_transform(action[:2])

        """
        until here
        """


        if self.has_hit and not self.stopped:
            self.stopped = True
            self.stop_after_hit_ee_pos = self.get_ee_pose(observation)


        if self.has_hit:
            print("slm")

        if self.stop_after_hit_ee_pos is not None:
            action = self.stop_after_hit_ee_pos
            action = self._action_transform(action[:2])


        if self.interpolation_order in [1, 2]:
            _action = action.flatten()
        else:
            _action = action


        return _action


class HitAgent(Agent):
    def __init__(self, env_info, **kwargs):
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # path = 'air_hockey_agent/agents/Agents/Hit_Agent'
        path = 'Agents/Hit_Agent'
        # path = 'envs/air_hockey_challenge/air_hockey_agent/agents/Agents/Hit_Agent'
        env_label = "7dof-hit"
        path = os.path.join(dir_path, path)

        trainer, self.env12 = self.load_agent(path, baseline_agent=False, env_label=env_label, random_agent=False, env_info=env_info)

        variant = json.load(open(os.path.join(path, 'variant.json')))


        # self.obs_dim = env.observation_space.low.size  # TODO needs hardcode
        # self.action_dim = env.action_space.low.size  # TODO needs hardcode
        # self.action_space = env.action_space

        super().__init__(env_info, variant, **kwargs)

        self.policy = trainer.policy
        self.trainer = trainer




class DefendAgent(Agent):
    def __init__(self, env_info, **kwargs):
        dir_path = os.path.dirname(os.path.abspath(__file__))

        path = 'air_hockey_agent/agents/Agents/Defend_Agent'
        path = 'Agents/Defend_Agent'

        # path = 'envs/air_hockey_challenge/air_hockey_agent/agents/Agents/Defend_Agent'
        env_label = "7dof-defend"
        path = os.path.join(dir_path, path)

        trainer, self.env12 = self.load_agent(path, baseline_agent=False, env_label=env_label, random_agent=False, env_info=env_info)

        variant = json.load(open(os.path.join(path, 'variant.json')))


        # self.obs_dim = env.observation_space.low.size  # TODO needs hardcode
        # self.action_dim = env.action_space.low.size  # TODO needs hardcode
        # self.action_space = env.action_space


        super().__init__(env_info, variant, **kwargs)

        self.policy = trainer.policy
        self.trainer = trainer






class PrepareAgent(Agent):
    def __init__(self, env_info, **kwargs):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        path = 'air_hockey_agent/agents/Agents/Prepare_Agent'
        path = 'Agents/Prepare_Agent'

        # path = 'envs/air_hockey_challenge/air_hockey_agent/agents/Agents/Prepare_Agent'
        env_label = "7dof-prepare"
        path = os.path.join(dir_path, path)

        trainer, self.env12 = self.load_agent(path, baseline_agent=False, env_label=env_label, random_agent=False, env_info=env_info)

        variant = json.load(open(os.path.join(path, 'variant.json')))


        # self.obs_dim = env.observation_space.low.size  # TODO needs hardcode
        # self.action_dim = env.action_space.low.size  # TODO needs hardcode
        # self.action_space = env.action_space

        super().__init__(env_info, variant, **kwargs)

        self.policy = trainer.policy
        self.trainer = trainer


