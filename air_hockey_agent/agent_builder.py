from air_hockey_challenge.framework import AgentBase
from air_hockey_agent.agents.hit_agent_SAC import HittingAgent
from air_hockey_agent.agents.ATACOM_hit_agent import AtacomHittingAgent
from air_hockey_challenge.utils.transformations import robot_to_world
import pandas as pd
import numpy as np
import os


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    return DataCollector(env_info, **kwargs)
    # agent = DummyAgent(env_info, **kwargs)
    # if 'agent' in kwargs:
    #     # if kwargs['agent'] == 'dummy-agent':
    #     #    agent = DummyAgent(env_info, **kwargs)
    #     # elif kwargs['agent'] == 'defend-agent':
    #     #    agent = SimpleDefendingAgent(env_info, **kwargs)
    #     if kwargs['agent'] == 'hit-agent':
    #         agent = AtacomHittingAgent(env_info, **kwargs)


class DummyAgent(AgentBase):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        if self.new_start:
            self.new_start = False
            self.hold_position = self.get_joint_pos(observation)

        velocity = np.zeros_like(self.hold_position)
        action = np.vstack([self.hold_position, velocity])
        return action


class DataCollector(DummyAgent):
    # def __init__(self):
    #     super(DataCollector, self).__init__()
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        self.dataset = pd.DataFrame()

    def draw_action(self, observation):
        actions = super(DataCollector, self).draw_action(observation)
        self.update_dataset(observation)
        return actions

    def update_dataset(self, observation):
        puck_pos, puck_vel = self.get_puck_state(observation)
        new_data = pd.DataFrame({'puck current pos X': [puck_pos[0]],
                                 'puck current pos Y': [puck_pos[1]],
                                 'puck current pos Yaw': [puck_pos[2]],
                                 'puck current vel X': [puck_vel[0]],
                                 'puck current vel Y': [puck_vel[1]],
                                 'puck current vel Yaw': [puck_vel[2]]})
        # self.dataset.append(new_data, ignore_index=True)
        self.dataset = pd.concat([self.dataset, new_data], ignore_index=False)

    def __del__(self):
        if len(self.dataset) > 6:
            folder_path = "Dataset/Noisy"
            files = os.listdir(folder_path)

            files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
            if files:
                last_file = files[0]  # Get the last file in the sorted list
                # Extract the file name without the extension
                base_name, _ = os.path.splitext(last_file)
                # Increment the file name by 1

                next_file_name = str(int(base_name) + 1)
                next_file_name += ".csv"
            else:
                next_file_name = "1"  # If there are no files, start with 1
                next_file_name += ".csv"
            # Create the full path for the next file
            next_file_path = os.path.join(folder_path, next_file_name)

            # self.dataset.drop(index=self.dataset.index[-1], axis=0, inplace=True)
            # self.dataset = self.dataset.head(self.dataset.shape[0] - 1)
            self.dataset.to_csv(next_file_path, index=False)




# if __name__ == '__main__':

    # # Assuming you have a dataframe named 'df'
    # file_path = 'D:\Research\\air_hockey_challenge\\Dataset\\Labels\\23.csv'
    # df = pd.read_csv(file_path)
    #
    # # Step 1: Identify rows with all zeros
    # all_zeros_mask = (df == 0).all(axis=1)
    #
    # # Step 2: Find the index of the row(s) to be removed
    # indices_to_remove = all_zeros_mask[all_zeros_mask].index
    #
    # # Step 3: Remove the row(s) and the subsequent row
    # df = df.drop(indices_to_remove).drop(indices_to_remove + 1)
    #
    # df.to_csv(file_path, index=False)
    # print("slm")

    # file_path = 'D:\Research\\air_hockey_challenge\\Dataset\\Labels\\24.csv'
    # df_label = pd.read_csv(file_path)
    # file_path = 'D:\Research\\air_hockey_challenge\\Dataset\\Noisy\\24.csv'
    # df_noisy = pd.read_csv(file_path)
    #
    # print(df_label.equals(df_noisy))
    # different = df_label.compare(df_noisy)
    # print("ajab")
