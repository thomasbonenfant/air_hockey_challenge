from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np
from collections import deque, defaultdict


class RolloutInfoLog(BaseCallback):

    def __init__(self, stats_window_size=100, log_freq=1, info_keywords=None):
        super().__init__()

        if info_keywords is None:
            info_keywords = list()
        self.stats_window_size = stats_window_size
        self.info_keywords = info_keywords
        self.log_freq = log_freq

        #self.ep_info_buffer = {k: deque(maxlen=self.stats_window_size) for k in self.info_keywords}

        self.env_infos_buffer = []

    def _on_rollout_start(self) -> None:
        self.env_infos_buffer.clear()

    def _on_step(self) -> bool:
        '''for idx, done in enumerate(self.locals['dones']):
            if done:
                for k in self.info_keywords:
                    self.ep_info_buffer[k].append(self.locals['infos'][idx][k])
        if self.log_freq > 0 and self.n_calls % self.log_freq == 0:
            for k in self.info_keywords:
                self.logger.record(f'rollout/ep_{k}', safe_mean(self.ep_info_buffer[k]))'''

        self.env_infos_buffer.extend(self.locals['infos'])

        if self.log_freq > 0 and self.n_calls % self.log_freq == 0:
            buffer = self.env_infos_buffer
            all_env_infos = {k: [b[k] for b in buffer] for k in self.info_keywords} # list of dict to dict of list

            for k in self.info_keywords:
                self.logger.record(f'rollout/max_{k}', np.max(all_env_infos[k]))
                self.logger.record(f'rollout/avg_{k}', np.mean(all_env_infos[k]))
                self.logger.record(f'rollout/std_{k}', np.std(all_env_infos[k]))

                # Now that we logged the buffer we can clear it
                self.env_infos_buffer.clear()
        return True








