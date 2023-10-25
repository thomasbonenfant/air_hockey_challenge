from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np


class RewardLogsCallback(BaseCallback):

    def _on_training_start(self) -> None:
        # self._log_freq = 1000

        output_formats = self.logger.output_formats

        self.tb_formatter = next(formatter for formatter in output_formats
                                 if isinstance(formatter, TensorBoardOutputFormat))

        self.constr_reward = []

    def _on_rollout_start(self) -> None:
        self.constr_reward = []

    def _on_step(self) -> bool:

        for info in self.locals['infos']:
            self.constr_reward.append(info["constr_reward"])

        return True

    def _on_rollout_end(self) -> None:
        self.tb_formatter.writer.add_scalar("rollout/average_constr_reward", np.average(self.constr_reward), self.num_timesteps)
        self.tb_formatter.writer.add_scalar("rollout/max_constr_reward", np.max(self.constr_reward), self.num_timesteps)
        self.tb_formatter.writer.add_scalar("rollout/min_constr_reward", np.min(self.constr_reward), self.num_timesteps)


