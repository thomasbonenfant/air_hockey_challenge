import pickle
from envs.airhockeydoublewrapper import AirHockeyDouble
from air_hockey_agent.agents.agents import DefendAgent, HitAgent, PrepareAgent
from air_hockey_agent.agents.rule_based_agent import PolicyAgent
from envs.fixed_options_air_hockey import HierarchicalEnv


def make_environment(steps_per_action=100, include_timer=False, include_faults=False,
                     render=False, large_reward=100, fault_penalty=33.33, fault_risk_penalty=0.1,
                     scale_obs=False, alpha_r=1.):
    env = AirHockeyDouble(interpolation_order=3)
    # env_info = env.env_info

    # load env_info of hit and defend environment
    with open("envs/env_info_single_agent/env_infos.pkl", "rb") as fp:
        env_info_hit, env_info_defend = pickle.load(fp)

    filter_opponent_ee_obs = lambda state: state[:-3]
    null_filter = lambda state: state

    defend_policy = DefendAgent(env_info_defend)
    hit_policy_oac = HitAgent(env_info_hit)
    hit_policy = PolicyAgent(env_info_hit, agent_id=1, task="hit")
    prepare_policy = PolicyAgent(env_info_defend, agent_id=1, task="prepare")

    policy_state_processors = {
        defend_policy: filter_opponent_ee_obs,
        hit_policy_oac: null_filter,
        hit_policy: null_filter,
        prepare_policy: null_filter
    }

    env = HierarchicalEnv(env=env,
                          steps_per_action=steps_per_action,
                          policies=[hit_policy, defend_policy],
                          policy_state_processors=policy_state_processors,
                          render_flag=render,
                          include_timer=include_timer,
                          include_faults=include_faults,
                          large_reward=large_reward,
                          fault_penalty=fault_penalty,
                          fault_risk_penalty=fault_risk_penalty,
                          scale_obs=scale_obs,
                          alpha_r=1.)
    return env
