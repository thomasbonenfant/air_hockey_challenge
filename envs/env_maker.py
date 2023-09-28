import pickle
from envs.airhockeydoublewrapper import AirHockeyDouble
from air_hockey_agent.agents.agents import DefendAgent, HitAgent, PrepareAgent
from air_hockey_agent.agents.rule_based_agent import PolicyAgent
from envs.fixed_options_air_hockey import HierarchicalEnv
from gymnasium.wrappers import FlattenObservation


def make_environment(steps_per_action=100, include_timer=False, include_faults=False,
                     render=False, large_reward=100, fault_penalty=33.33, fault_risk_penalty=0.1,
                     scale_obs=False, alpha_r=1., include_joints=False):
    env = AirHockeyDouble(interpolation_order=3)
    env_info = env.env_info

    # load env_info of hit and defend environment
    #with open("envs/env_info_single_agent/env_infos.pkl", "rb") as fp:
    #    env_info_hit, env_info_defend = pickle.load(fp)

    filter_opponent_ee_obs = lambda state: state[:-3]
    #null_filter = lambda state: state

    #defend_policy_oac = DefendAgent(env_info)
    hit_policy_oac = HitAgent(env_info)
    hit_policy_rb = PolicyAgent(env_info, agent_id=1, task="hit", smash_line=-0.5)
    prepare_policy_rb = PolicyAgent(env_info, agent_id=1, task="prepare")
    defend_rb = PolicyAgent(env_info, agent_id=1, task="defend")

    policy_state_processors = {
        #defend_policy_oac: filter_opponent_ee_obs,
        #hit_policy_oac: null_filter,
        #hit_policy_rb: null_filter,
        #prepare_policy_rb: null_filter,
        #defend_rb: null_filter
    }

    env = HierarchicalEnv(env=env,
                          steps_per_action=steps_per_action,
                          policies=[hit_policy_rb, defend_rb, prepare_policy_rb],
                          policy_state_processors=policy_state_processors,
                          render_flag=render,
                          include_joints=include_joints,
                          include_timer=include_timer,
                          include_faults=include_faults,
                          large_reward=large_reward,
                          fault_penalty=fault_penalty,
                          fault_risk_penalty=fault_risk_penalty,
                          scale_obs=scale_obs,
                          alpha_r=alpha_r)

    env = FlattenObservation(env)

    return env
