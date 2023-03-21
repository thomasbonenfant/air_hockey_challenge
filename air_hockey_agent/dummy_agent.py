import numpy as np
from air_hockey_challenge.framework.agent_base import AgentBase

class DummyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

        self.path = kwargs['path'] if 'path' in kwargs else [[-0.8, 0, 0]]

        self.steps_per_action = kwargs['steps_per_action'] if 'steps_per_action' in kwargs else 50

        self.step = 0
        self.path_idx = 0

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        if self.new_start == True:
            self.path_idx = 0
            self.new_start = False
        else:
            self.path_idx = (self.path_idx + 1) \
                if self.step % self.steps_per_action == 0 and self.path_idx < len(self.path) - 1 \
                     else self.path_idx  #updates path_idx every steps_per_action steps
        
        action = self.path[self.path_idx]        

        #print(f'Step: {self.step} \t\t Observation: {observation}')

        self.step += 1

        return np.array(action)
