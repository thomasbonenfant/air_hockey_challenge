from envs.air_hockey import AirHockeyEnv

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
    "dont_include_timer_in_states": False,
    "action_persistence": 1,

}

env = AirHockeyEnv(**env_args)
env.reset()
done = False

while not done:
    action = env.action_space.sample()
    s, r, done, i = env.step(action)

    env.render()
