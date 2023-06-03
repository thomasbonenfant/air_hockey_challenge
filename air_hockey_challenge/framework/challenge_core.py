import time

import numpy as np
from mushroom_rl.core import Core


class ChallengeCore(Core):
    def __init__(self, *args, action_idx=None, **kwargs):
        if action_idx:
            self.action_idx = action_idx
        else:
            self.action_idx = [0, 1]
        super().__init__(*args, **kwargs)
        
    def _step(self, render):
        """
        Single step.

        Args:
            render (bool):
                whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the
            agent, the reward obtained, the reached state, the absorbing flag
            of the reached state and the last step flag.

        """
        q, dq = self.agent.get_joint_pos(self._state), self.agent.get_joint_vel(self._state)
        start_time = time.time()
        action = self.agent.draw_action(self._state)
        end_time = time.time()
        #self.mdp.set_q_dq(q, dq)
        next_state, reward, absorbing, step_info = self.mdp.step(action)
#        next_state, reward, absorbing, step_info = self.mdp.step(action)
        step_info["computation_time"] = (end_time - start_time)

        self._episode_steps += 1

        if render:
            self.mdp.render()

        last = not (
                self._episode_steps < self.mdp.info.horizon and not absorbing)


        state = self._state
        next_state = self._preprocess(next_state.copy())
        #print(f"postATACOM: obs: {next_state}")
        # added to compute reward
        #reward = self.mdp.base_env.reward(state, None, next_state, absorbing)
        self._state = next_state

        return (state, action, reward, next_state, absorbing, last), step_info
