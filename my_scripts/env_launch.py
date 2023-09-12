import pickle

from envs.airhockeydoublewrapper import AirHockeyDouble
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from envs.fixed_options_air_hockey import HierarchicalEnv
from air_hockey_agent.agents.agents import DefendAgent, HitAgent, PrepareAgent
from air_hockey_agent.agents.rule_based_agent import PolicyAgent

from stable_baselines3 import PPO

env_args = {
    "env": 'tournament',
    "interpolation_order": 3,
    "simple_reward": False,
    "high_level_action": True,
    "agent_id": 1,
    "delta_action": False,
    "delta_ratio": 0.1,
    "jerk_only": False,
    "include_joints": True,
    "shaped_reward": False,
    "clipped_penalty": 0.5,
    "large_reward": 100,
    "large_penalty": 100,
    "min_jerk": 10000,
    "max_jerk": 10000,
    "alpha_r": 1.0,
    "c_r": 0.0,
    "include_hit": True,
    "history": 0,
    "use_atacom": True,
    "stop_after_hit": False,
    "punish_jerk": False,
    "acceleration": False,
    "max_accel": 0.2,
    "include_old_action": False,
    "use_aqp": False,
    "aqp_terminates": False,
    "speed_decay": 0.5,
    "clip_vel": False,
    "whole_game_reward": True,
    "score_reward": 10,
    "fault_penalty": 5,
    "load_second_agent": False,
    "dont_include_timer_in_states": True,
    "action_persistence": 1,

}


env = AirHockeyDouble(interpolation_order=3)
# env_info = env.env_info

# load env_info of hit and defend environment
with open("../envs/env_info_single_agent/env_infos.pkl", "rb") as fp:
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

load_agent = False

if load_agent:
    agent = PPO.load("../models/ppo/saved/best_model")

env = HierarchicalEnv(env, 100, [hit_policy, defend_policy], policy_state_processors, render_flag=True, include_timer=True, include_faults=True, scale_obs=True, alpha_r=1)
for i in range(10):
    s, info = env.reset()
    done = False

    while not done:
        if load_agent:
            action, _ = agent.predict(observation=s, deterministic=True)
        else:
            action = env.action_space.sample()
        s, r, done, truncated, info = env.step(action)
        #print(s[-2:])
    print('episode done')
