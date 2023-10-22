import pickle
from envs.airhockeydoublewrapper import AirHockeyDouble
from air_hockey_agent.agents.agents import DefendAgent, HitAgent, PrepareAgent
from air_hockey_agent.agents.rule_based_agent import PolicyAgent
from envs.fixed_options_air_hockey import HierarchicalEnv
from gymnasium.wrappers import FlattenObservation


def make_environment(steps_per_action=100, include_timer=False, include_faults=False,
                     render=False, large_reward=100, fault_penalty=33.33, fault_risk_penalty=0.1,
                     scale_obs=False, alpha_r=1., include_joints=False, include_ee=False):
    env = AirHockeyDouble(interpolation_order=3)
    env_info = env.env_info

    defend_policy_oac = DefendAgent(env_info, env_label="tournament")
    repel_agent_oac = DefendAgent(env_info, env_label="7dof-defend")
    #hit_policy_oac = HitAgent(env_info)
    hit_policy_rb = PolicyAgent(env_info, agent_id=1, task="hit")
    prepare_policy_rb = PolicyAgent(env_info, agent_id=1, task="prepare")
    home_policy_rb = PolicyAgent(env_info, agent_id=1, task="home")

    policy_state_processors = {}

    env = HierarchicalEnv(env=env,
                          steps_per_action=steps_per_action,
                          policies=[hit_policy_rb, defend_policy_oac, repel_agent_oac, prepare_policy_rb, home_policy_rb],
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
                          include_ee=include_ee)

    env = FlattenObservation(env)



    return env
