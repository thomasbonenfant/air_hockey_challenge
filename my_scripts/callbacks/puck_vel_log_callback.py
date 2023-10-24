from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np


class RewardLogsCallback(BaseCallback):

    def _on_training_start(self) -> None:
        # self._log_freq = 1000

        output_formats = self.logger.output_formats

        self.tb_formatter = next(formatter for formatter in output_formats
                                 if isinstance(formatter, TensorBoardOutputFormat))



    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:




        return True

    def _on_rollout_end(self) -> None:
        pass
        #self.tb_formatter.writer.add_scalar("rollout/average_large_reward", np.average(self.large_reward), self.num_timesteps)



