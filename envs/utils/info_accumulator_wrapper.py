from gymnasium import Wrapper
import numpy as np


class InfoStatsWrapper(Wrapper):
    '''
    When the episode terminates it adds for some specified info keywords
    the average of those particular infos and their max as "sum_{keyword}" and "max_{keyword}"
    '''
    def __init__(self, env, info_keywords):
        super().__init__(env)

        self.info_keywords = info_keywords
        self.info_stats = {k: list() for k in self.info_keywords}

        self.steps = 0

    def step(self, action):
        self.steps += 1
        obs, rew, done, truncated, info = self.env.step(action)

        for k in self.info_stats:
            # self.info_stats[k].append(np.sum(np.abs(info[k])))
            self.info_stats[k].extend(info[k])

        # if episode ends add new info
        if done or truncated:
            for k in self.info_stats:
                info[f'avg_{k}'] = np.mean(self.info_stats[k])
                info[f'std_{k}'] = np.std(self.info_stats[k])
                info[f'max_{k}'] = np.max(self.info_stats[k])

        return obs, rew, done, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.steps = 0

        self.info_stats = {k: list() for k in self.info_keywords}

        return obs, info
