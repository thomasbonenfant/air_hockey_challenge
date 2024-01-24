import os
import numpy as np
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian
from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.utils.ATACOM_transformation import AtacomTransformation, build_ATACOM_Controller
from air_hockey_agent.agents.kalman_filter import PuckTracker
from air_hockey_agent.utils.sb3_variant_util import get_configuration
from stable_baselines3 import SAC, PPO
from envs.utils import Specification
from gymnasium.spaces import Dict, flatten_space, flatten
from envs.env_maker import task_dict

alg_dict = {
    'sac': SAC,
    'ppo': PPO
}


class OptionSB3(AgentBase):
    def __init__(self, env_info, path, **kwargs):
        super().__init__(env_info, agent_id=1, **kwargs)

        agent_path = os.path.join(path, 'best_model')

        env_args, alg = get_configuration(path)

        del env_args['env']
        task = env_args.pop('task')
        task_args = task_dict[task]['task_args']
        self.task = task_dict[task]['task_obj'](**task_args)

        self.agent = alg_dict[alg].load(agent_path)
        self.specs = Specification(env_info=env_info, **env_args)

        self.observation_space = Dict({"observation": flatten_space(self.specs.observation_space)})
        self.action_space = self.specs.action_space

        atacom = build_ATACOM_Controller(self.env_info, slack_type='soft_corner', slack_tol=1e-06, slack_beta=4)
        self.atacom_transformation = AtacomTransformation(self.env_info, False, atacom)

        self.task.init(self.specs)
        self.observation_space = self.task.update_space(self.observation_space, self.specs)

        self._obs = None
        self.has_hit = False

        self.t = 0
        self.restart = True
        self.puck_tracker = PuckTracker(env_info)
        self.last_joint_pos_action = None
        self.last_joint_vel_action = None

    def reset(self):
        self.restart = True
        self.has_hit = False
        self.atacom_transformation.reset()

    def process_state(self, state):
        obs = {}

        if self.specs.include_puck:
            obs['puck_pos'] = state[self.specs.f_puck_pos_ids]
            obs['puck_vel'] = state[self.specs.f_puck_vel_ids]

        if self.specs.include_joints:
            obs['joint_pos'] = state[self.specs.f_joint_pos_ids]
            obs['joint_vel'] = state[self.specs.f_joint_vel_ids]

        if self.specs.include_opponent:
            obs['opponent_ee'] = state[self.specs.f_opponent_ee_ids]

        if self.specs.include_ee:
            obs['ee_pos'] = self.ee_pos[:2]  # do not include z coordinate

        if self.specs.include_ee_vel:
            obs['ee_vel'] = self.ee_vel[:2]

        if self.specs.include_hit_flag:
            obs['has_hit'] = self.has_hit

        if self.specs.scale_obs:
            obs = self._scale_obs(obs)

        state = {
            'observation': flatten(self.specs.observation_space, obs),
            'desired_goal': self.task.get_desired_goal()
        }

        if self.task.include_achieved:
            state.update({'achieved_goal': self.task.get_achieved_goal(self)})

        return state

    def _scale_obs(self, obs):
        for k in self.specs.min_dict:
            obs[k] = self._scale(obs[k], self.specs.min_dict[k], self.specs.range_dict[k])
        return obs

    @staticmethod
    def _scale(x, x_min, x_range):
        return (x - x_min) / x_range * 2 - 1

    def _post_simulation(self, obs):
        self._obs = obs
        self.puck_pos = self.get_puck_pos(obs)
        self.previous_vel = self.puck_vel if self.t > 0 else None
        self.puck_vel = self.get_puck_vel(obs)
        self.joint_pos = self.get_joint_pos(obs)
        self.joint_vel = self.get_joint_vel(obs)
        self.previous_ee_pos = self.ee_pos if self.t > 0 else None
        self.ee_pos = self.get_ee_pose(obs)

        self.ee_vel = self._apply_forward_velocity_kinematics(self.joint_pos, self.joint_vel)
        if self.previous_vel is not None:
            previous_vel_norm = np.linalg.norm(self.previous_vel[:2])
            current_vel_norm = np.linalg.norm(self.puck_vel[:2])
            distance = np.linalg.norm(self.puck_pos[:2] - self.ee_pos[:2])
            if previous_vel_norm <= current_vel_norm and distance <= (
                    self.specs.puck_radius + self.specs.mallet_radius) * 1.1:
                self.has_hit = True

    def _apply_forward_velocity_kinematics(self, joint_pos, joint_vel):
        robot_model = self.specs.robot_model
        robot_data = self.specs.robot_data
        jac = jacobian(robot_model, robot_data, joint_pos)
        jac = jac[:3]  # last part of the matrix is about rotation. no need for it
        ee_vel = jac @ joint_vel
        return ee_vel
