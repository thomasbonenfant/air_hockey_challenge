from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np


class SummaryWriterCallback(BaseCallback):

    def _on_training_start(self) -> None:
        # self._log_freq = 1000

        output_formats = self.logger.output_formats

        self.tb_formatter = next(formatter for formatter in output_formats
                                 if isinstance(formatter, TensorBoardOutputFormat))

        self.large_reward = []
        self.fault_penalty = []
        self.fault_risk_penalty = []
        self.constr_penalty = []

    def _on_rollout_start(self) -> None:
        self.large_reward = []
        self.fault_penalty = []
        self.fault_risk_penalty = []
        self.constr_penalty = []

    def _on_step(self) -> bool:

        for info in self.locals['infos']:
            self.large_reward.append(info["large_reward"])
            self.fault_penalty.append(info["fault_penalty"])
            self.fault_risk_penalty.append(info["fault_risk_penalty"])
            self.constr_penalty.append(info["constr_penalty"])

        print(self.locals)
        quit()


        return True

    def _on_rollout_end(self) -> None:
        self.tb_formatter.writer.add_scalar("rollout/average_large_reward", np.average(self.large_reward), self.num_timesteps)
        self.tb_formatter.writer.add_scalar("rollout/average_fault_penalty", np.average(self.fault_penalty), self.num_timesteps)
        self.tb_formatter.writer.add_scalar("rollout/average_fault_risk_penalty", np.average(self.fault_risk_penalty), self.num_timesteps)
        self.tb_formatter.writer.add_scalar("rollout/average_constr_penalty", np.average(self.constr_penalty), self.num_timesteps)
        self.tb_formatter.writer.add_scalar("rollout/max_constr_penalty", np.max(self.constr_penalty), self.num_timesteps)
        self.tb_formatter.writer.add_scalar("rollout/min_constr_penalty", np.min(self.constr_penalty), self.num_timesteps)


