import os
import pickle


class Logger(object):
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.actions = []
        self.states = []
        self.rewards = []
        self.dones = []
        self.infos = []

    def store(self, action, state, reward, done, info):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)

    def dump(self):
        with open(os.path.join(self.log_path, "data.pkl"), 'wb') as file:
            obj = [self.actions, self.states, self.rewards, self.dones, self.infos]
            pickle.dump(obj, file)

    def save_env_info(self, env_info):
        with open(os.path.join(self.log_path, "env_info.pkl"), 'wb') as file:
            obj = env_info
            pickle.dump(obj, file)

