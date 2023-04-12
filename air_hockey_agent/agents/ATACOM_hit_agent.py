import numpy as np
from mushroom_rl.algorithms.actor_critic import SAC

from air_hockey_agent.agents.hit_agent import HittingAgent
from air_hockey_challenge.framework.agent_base import AgentBase


class AtacomHittingAgent(HittingAgent):

    def __init__(self, env, **kwargs):
        self.mu = 0
        super().__init__(env, **kwargs)

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        """
        Draw an action from the agent's policy.
        :param observation: The current observation of the environment.
        """
        # Sample policy action αk ∼ π(·|sk).
        action = super().draw_action(observation)
        # Observe the qk, q˙k from sk
        q = observation[self.env_info['joint_pos_ids']]
        dq = observation[self.env_info['joint_vel_ids']]
        # Compute Jc, k = Jc(qk, µk), ψk = ψ(qk, q˙k), ck = c(qk, q˙k, µk)

        # Compute the RCEF of tangent space basis of NcR

        # Compute the tangent space acceleration [q¨k µ˙ k].T ← −J^†_c,k [K_cck + ψ_k] + N^R_c α_k

        # Clip the joint acceleration q¨k ← clip(q¨k, al, au)

        # Integrate the slack variable µk+1 = µk + µ˙ k∆T

        return np.array([action, np.ones(3) * 0.05])

