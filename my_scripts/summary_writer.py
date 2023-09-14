from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np


class SummaryWriterCallback(BaseCallback):

    def _on_training_start(self) -> None:
        # self._log_freq = 1000

        output_formats = self.logger.output_formats

        self.tb_formatter = next(formatter for formatter in output_formats
                                 if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:

        large_reward = []
        fault_penalty = []
        fault_risk_penalty = []
        constr_penalty = []

        for info in self.locals['infos']:
            reward = info['reward']
            large_reward.append(reward["log_large_reward"])
            fault_penalty.append(reward["log_fault_penalty"])
            fault_risk_penalty.append(reward["log_fault_risk_penalty"])
            constr_penalty.append(reward["log_constr_penalty"])

        self.tb_formatter.writer.add_scalar("train/large_reward", np.sum(large_reward), self.num_timesteps)
        self.tb_formatter.writer.add_scalar("train/fault_penalty", np.sum(fault_penalty), self.num_timesteps)
        self.tb_formatter.writer.add_scalar("train/fault_risk_penalty", np.sum(fault_risk_penalty), self.num_timesteps)
        self.tb_formatter.writer.add_scalar("train/log_constr_penalty", np.sum(constr_penalty), self.num_timesteps)

        return True