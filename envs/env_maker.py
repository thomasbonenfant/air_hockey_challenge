from envs.airhockeydoublewrapper import AirHockeyDouble
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_agent.agents.agents import DefendAgent, HitAgent, PrepareAgent, RepelAgent
from air_hockey_agent.agents.agent_sb3 import AgentSB3
from air_hockey_agent.agents.rule_based_agent import PolicyAgent
from envs.fixed_options_air_hockey import HierarchicalEnv
from envs.air_hockey_hit import AirHockeyHit
from envs.air_hockey_goal import AirHockeyGoal
from envs.air_hockey_oac import AirHockeyOAC
from envs.air_hockey_option_task import AirHockeyOptionTask
from envs.air_hockey_option import AirHockeyOption
from gymnasium.wrappers import FlattenObservation, EnvCompatibility
from utils.env_utils import NormalizedBoxEnv
from envs.info_accumulator_wrapper import InfoStatsWrapper
from envs.utils import *

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

REWARD_HER = 'reward_HER'
REWARD_FN = 'reward_fn'
TASK_CLS = 'task_obj'
TASK_ARGS = 'task_args'


task_dict = {
    'hit_her': {
        REWARD_HER: goal_reward_hit,
        REWARD_FN: None,
        TASK_CLS: PuckDirectionTask,
        TASK_ARGS: {'include_achieved': True, 'include_hit_flag': False},
        'flatten_space': False
    },
    'hit': {
        REWARD_HER: None,
        REWARD_FN: reward_hit,
        TASK_CLS: PuckDirectionTask,
        TASK_ARGS: {'include_achieved': False, 'include_hit_flag': False},
        'flatten_space': True
    },
    'repel_her': {
        REWARD_HER: goal_reward_repel,
        REWARD_FN: None,
        TASK_CLS: PuckDirectionTask,
        TASK_ARGS: {'include_achieved': True, 'include_hit_flag': False},
        'flatten_space': False
    },
    'defend': {
        REWARD_HER: None,
        REWARD_FN: reward_defend2,
        TASK_CLS: None,
        TASK_ARGS: {},
        'flatten_space': True
    },
    'prepare_her': {
        REWARD_HER: goal_reward_prepare,
        REWARD_FN: None,
        TASK_CLS: PuckPositionTask,
        TASK_ARGS: {},
        'flatten_space': False

    }
}


def make_option_environment(task, include_opponent, include_joints, include_ee, include_ee_vel, joint_acc_clip,
                            include_puck, remove_last_joint,
                            scale_obs, scale_action, alpha_r, max_path_len, stop_after_hit, include_hit_flag):
    reward_fn = task_dict[task][REWARD_FN]
    reward_her = task_dict[task][REWARD_HER]
    task_cls = task_dict[task][TASK_CLS]
    task_args = task_dict[task][TASK_ARGS]
    flatten_space = task_dict[task]['flatten_space']

    assert (reward_fn and not reward_her) or (reward_her and not reward_fn)

    task = task.split('_')[0]

    # convert repel to defend
    if task == "repel":
        task = "defend"

    env = AirHockeyChallengeWrapper('7dof-' + task, interpolation_order=3)

    specs = Specification(env_info=env.env_info,
                          include_joints=include_joints,
                          include_ee=include_ee,
                          include_ee_vel=include_ee_vel,
                          include_puck=include_puck,
                          include_opponent=include_opponent,
                          joint_acc_clip=joint_acc_clip,
                          scale_obs=scale_obs,
                          scale_action=scale_action,
                          max_path_len=max_path_len,
                          remove_last_joint=remove_last_joint,
                          alpha_r=alpha_r,
                          stop_after_hit=stop_after_hit,
                          include_hit_flag=include_hit_flag)
    if task_cls:
        env = AirHockeyOptionTask(specification=specs, reward_her=reward_her, reward_fn=reward_fn, env=env)
        task_obj = task_cls(**task_args)

        env.set_task(task_obj)
    else:
        env = AirHockeyOption(env=env, specification=specs, reward_fn=reward_fn)

    if flatten_space:
        env = FlattenObservation(env)

    return env


def make_hrl_environment(policies, steps_per_action=100, include_timer=False, include_faults=False,
                         render=False, large_reward=100, fault_penalty=33.33, fault_risk_penalty=0.1,
                         scale_obs=False, alpha_r=1., include_joints=False, include_ee=False,
                         include_prev_action=False, include_opponent=False):
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
                          include_opponent=include_opponent,
                          include_prev_action=include_prev_action)

    env = FlattenObservation(env)
    # env = InfoStatsWrapper(env, info_keywords=['joint_pos_constr', 'joint_vel_constr', 'ee_constr'])
    return env


def make_hit_env(include_joints, include_opponent, include_ee, include_ee_vel, joint_acc_clip, include_puck,
                 remove_last_joint,
                 scale_obs, scale_action, alpha_r, hit_coeff, aim_coeff, max_path_len, stop_after_hit):
    env = AirHockeyDouble(interpolation_order=3)
    env = AirHockeyHit(env, include_joints=include_joints, include_opponent=include_opponent,
                       include_ee=include_ee, include_ee_vel=include_ee_vel,
                       include_puck=include_puck, remove_last_joint=remove_last_joint, scale_obs=scale_obs,
                       joint_acc_clip=joint_acc_clip, scale_action=scale_action, alpha_r=alpha_r, hit_coeff=hit_coeff,
                       aim_coeff=aim_coeff, max_path_len=max_path_len, stop_after_hit=stop_after_hit)
    env = FlattenObservation(env)
    # env = InfoStatsWrapper(env, info_keywords=['joint_pos_constr', 'joint_vel_constr', 'ee_constr'])
    return env


def make_goal_env(include_joints, include_ee, include_ee_vel, include_puck, remove_last_joint,
                  scale_obs, scale_action, alpha_r, goal_horizon, max_path_len, joint_acc_clip):
    env = AirHockeyDouble(interpolation_order=3)
    env = AirHockeyGoal(env, include_joints=include_joints, include_ee=include_ee, include_ee_vel=include_ee_vel,
                        include_puck=include_puck, remove_last_joint=remove_last_joint, scale_obs=scale_obs,
                        scale_action=scale_action, alpha_r=alpha_r, goal_horizon=goal_horizon,
                        max_path_len=max_path_len,
                        joint_acc_clip=joint_acc_clip)
    env = FlattenObservation(env)
    return env


def make_airhockey_oac(env, interpolation_order=3,
                       simple_reward=False, high_level_action=False, agent_id=1, delta_action=False, delta_ratio=0.1,
                       jerk_only=False, include_joints=True, shaped_reward=True, clipped_penalty=0.5, large_reward=100,
                       large_penalty=100, min_jerk=10000, max_jerk=10000, alpha_r=1., c_r=0., include_hit=False,
                       history=0,
                       use_atacom=True, stop_after_hit=False, punish_jerk=False, acceleration=False, max_accel=0.2,
                       include_old_action=False, use_aqp=False, aqp_terminates=False, speed_decay=0.5, clip_vel=False,
                       whole_game_reward=False, score_reward=10, fault_penalty=5, load_second_agent=True,
                       dont_include_timer_in_states=True, action_persistence=1, stop_when_puck_otherside=False,
                       curriculum_learning_step1=False, curriculum_learning_step2=False, curriculum_learning_step3=False
                       , start_from_defend=False, original_env=False, start_curriculum_transition=3000,
                       end_curriculum_transition=4000, curriculum_transition=False, **kwargs):
    env = AirHockeyOAC(env, interpolation_order=interpolation_order,
                       simple_reward=simple_reward,
                       high_level_action=high_level_action,
                       agent_id=agent_id,
                       delta_action=delta_action,
                       delta_ratio=delta_ratio,
                       jerk_only=jerk_only,
                       include_joints=include_joints,
                       shaped_reward=shaped_reward,
                       clipped_penalty=clipped_penalty,
                       large_reward=large_reward,
                       large_penalty=large_penalty,
                       min_jerk=min_jerk,
                       max_jerk=max_jerk,
                       alpha_r=alpha_r,
                       c_r=c_r,
                       include_hit=include_hit,
                       history=history,
                       use_atacom=use_atacom,
                       stop_after_hit=stop_after_hit,
                       punish_jerk=punish_jerk,
                       acceleration=acceleration,
                       max_accel=max_accel,
                       include_old_action=include_old_action,
                       use_aqp=use_aqp,
                       aqp_terminates=aqp_terminates,
                       speed_decay=speed_decay,
                       clip_vel=clip_vel,
                       whole_game_reward=whole_game_reward,
                       score_reward=score_reward,
                       fault_penalty=fault_penalty,
                       load_second_agent=load_second_agent,
                       dont_include_timer_in_states=dont_include_timer_in_states,
                       action_persistence=action_persistence,
                       stop_when_puck_otherside=stop_when_puck_otherside,
                       curriculum_learning_step1=curriculum_learning_step1,
                       curriculum_learning_step2=curriculum_learning_step2,
                       curriculum_learning_step3=curriculum_learning_step3,
                       start_from_defend=start_from_defend,
                       original_env=original_env,
                       start_curriculum_transition=start_curriculum_transition,
                       end_curriculum_transition=end_curriculum_transition,
                       curriculum_transition=curriculum_transition,
                       **kwargs)
    env = NormalizedBoxEnv(env)
    env = EnvCompatibility(env)
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
    if env_name.startswith('oac_'):
        # needs an argument named env in the constructor
        env_args['env'] = env_name.split('oac_')[1]
        return lambda: make_airhockey_oac(**env_args)
    if env_name == 'option':
        return lambda: make_option_environment(**env_args)
    raise NotImplementedError
