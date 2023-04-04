from air_hockey_challenge.framework.agent_base import AgentBase

class AgentEEBase(AgentBase):
    '''
    Agent Base Class to use with CustomEnvironmentWrapper

    It redefines get_ee_pose to get the ee_pose from obs. (Because CustomEnvironmentWrapper now includes ee_pos and ee_vel into obs)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_ee_pose(self, obs):
        return obs[self.env_info['ee_pos_ids']]

    def get_ee_vel(self, obs):
        return obs[self.env_info['ee_vel_ids']]

    def get_puck_pos(self, obs):
        return obs[self.env_info['new_puck_pos_ids']]

    def get_puck_vel(self, obs):
        return obs[self.env_info['new_puck_vel_ids']]

    def get_joint_pos(self, obs):
        raise NotImplementedError

    def get_joint_vel(self, obs):
        raise NotImplementedError

    