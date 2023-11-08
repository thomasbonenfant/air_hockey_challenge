from envs.airhockeydoublewrapper import AirHockeyDouble
from air_hockey_agent.agents.agents import DefendAgent, HitAgent, PrepareAgent, RepelAgent
from air_hockey_agent.agents.agent_sb3 import AgentSB3
from air_hockey_agent.agents.rule_based_agent import PolicyAgent
from envs.fixed_options_air_hockey import HierarchicalEnv
from envs.air_hockey_hit import AirHockeyHit
from envs.air_hockey_goal import AirHockeyGoal
from gymnasium.wrappers import FlattenObservation
from envs.info_accumulator_wrapper import InfoStatsWrapper

policy_dict = {
    'hit_rb': lambda env_info: PolicyAgent(env_info, agent_id=1, task="hit"),
    'hit_oac': lambda env_info: HitAgent(env_info),
    'hit_sb3': lambda env_info: AgentSB3(env_info, path="Agents/Hit_Agent"),
    'defend_rb': lambda env_info: PolicyAgent(env_info, agent_id=1, task="defend"),
    'defend_oac': lambda env_info: DefendAgent(env_info, env_label='tournament'),
    'repel_oac': lambda env_info: RepelAgent(env_info, env_label='7dof-defend'),
    'prepare_rb': lambda env_info: PolicyAgent(env_info, agent_id=1, task='prepare'),
    'home_rb': lambda env_info: PolicyAgent(env_info, agent_id=1, task='home'),
    'home_sb3': lambda env_info: AgentSB3(env_info, acc_ratio=0.1, path="Agents/Home_Agent")
}


def make_hrl_environment(policies, steps_per_action=100, include_timer=False, include_faults=False,
                         render=False, large_reward=100, fault_penalty=33.33, fault_risk_penalty=0.1,
                         scale_obs=False, alpha_r=1., include_joints=False, include_ee=False,
                         include_prev_action=False):
    env = AirHockeyDouble(interpolation_order=3)
    env_info = env.env_info

    policy_state_processors = {}

    env = HierarchicalEnv(env=env,
                          steps_per_action=steps_per_action,
                          policies=[policy_dict[name](env_info) for name in policies],
                          policy_state_processors=policy_state_processors,
                          render_flag=render,
                          include_joints=include_joints,
                          include_timer=include_timer,
                          include_faults=include_faults,
                          large_reward=large_reward,
                          fault_penalty=fault_penalty,
                          fault_risk_penalty=fault_risk_penalty,
                          scale_obs=scale_obs,
                          alpha_r=alpha_r,
                          include_ee=include_ee,
                          include_prev_action=include_prev_action)

    env = FlattenObservation(env)
    # env = InfoStatsWrapper(env, info_keywords=['joint_pos_constr', 'joint_vel_constr', 'ee_constr'])
    return env


def make_hit_env(include_joints, include_ee, include_ee_vel, include_puck, remove_last_joint,
                 scale_obs, scale_action, alpha_r, hit_coeff, max_path_len):
    env = AirHockeyDouble(interpolation_order=3)
    env = AirHockeyHit(env, include_joints=include_joints, include_ee=include_ee, include_ee_vel=include_ee_vel,
                       include_puck=include_puck, remove_last_joint=remove_last_joint, scale_obs=scale_obs,
                       scale_action=scale_action, alpha_r=alpha_r, hit_coeff=hit_coeff,
                       max_path_len=max_path_len)
    env = FlattenObservation(env)
    # env = InfoStatsWrapper(env, info_keywords=['joint_pos_constr', 'joint_vel_constr', 'ee_constr'])
    return env


def make_goal_env(include_joints, include_ee, include_ee_vel, include_puck, remove_last_joint,
                  scale_obs, scale_action, alpha_r, max_path_len):
    env = AirHockeyDouble(interpolation_order=3)
    env = AirHockeyGoal(env, include_joints=include_joints, include_ee=include_ee, include_ee_vel=include_ee_vel,
                        include_puck=include_puck, remove_last_joint=remove_last_joint, scale_obs=scale_obs,
                        scale_action=scale_action, alpha_r=alpha_r, max_path_len=max_path_len)
    #env = FlattenObservation(env)
    return env


def create_producer(env_args):
    env_name = env_args['env']
    print(f'Environment: {env_name}')

    env_args = {k: v for k, v in env_args.items() if k != 'env'}  # exclude env name

    if env_name == 'hrl':
        return lambda: make_hrl_environment(**env_args)
    if env_name == 'hit':
        return lambda: make_hit_env(**env_args)
    if env_name == 'goal':
        return lambda: make_goal_env(**env_args)
    raise NotImplementedError
