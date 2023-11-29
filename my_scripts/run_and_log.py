from air_hockey_agent.agents.hierarchical_agent import AgentSB3
from baseline.baseline_agent.baseline_agent import BaselineAgent
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_agent.delayed_baseline import DelayedBaseline
from my_scripts.utils.logger import Logger

import numpy as np

env = AirHockeyChallengeWrapper('tournament')
agent1 = AgentSB3(env_info=env.env_info, path="/home/thomas/Downloads/markov/data/hit/sac/fineTunedClip_6dof_withoutOpponent/561982")
#agent1 = BaselineAgent(env_info=env.env_info, agent_id=1)
agent2 = DelayedBaseline(env_info=env.env_info, start_time=50)

logger = Logger(log_path='/home/thomas/Desktop/sb3_logs')

n_steps = 4000
step = 0

action_idx = (np.arange(env.base_env.action_shape[0][0]),
                           np.arange(env.base_env.action_shape[1][0]))

obs = env.reset()
agent1.reset()
agent2.reset()

while step < n_steps:
    obs1, obs2 = np.split(obs, 2)

    action1 = agent1.draw_action(obs1)
    action2 = agent2.draw_action(obs2)

    dual_action = (action1, action2)

    obs, reward, done, info = env.step(np.array((dual_action[0][action_idx[0]],
                                                     dual_action[1][action_idx[1]])))

    if done or step % 1 == 0:
        obs = env.reset()
        agent1.reset()
        agent2.reset()

    env.render()

    step += 1

    logger.store(action1, obs1, reward, done, info)

logger.dump()

