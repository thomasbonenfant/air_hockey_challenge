"""Data Processing class performing a data transformation of the position
of the agent, by using RBF."""
# Libraries
from MagicRL.data_processors.utils import gauss
from MagicRL.envs.utils import Position
from MagicRL.data_processors.base_processor import BaseProcessor
import numpy as np


# Data Processor class
class GWDataProcessorRBF(BaseProcessor):
    """
    Data Processor Class Mapping a state of GridWorld Continuous environment
    into a feature vector.
    """
    def __init__(self, num_basis: int, grid_size: int, std_dev: float) -> None:
        """
        Args:
            num_basis (int): how many Gaussians to use
            grid_size (int): the dimension of the gridworld
            std_dev (float): the standard deviation that each gaussian needs
            to have
        """
        super().__init__()
        assert num_basis > 0, "[ERROR] The number of basis must be positive."
        assert grid_size > 0, "[ERROR] The grid size must be positive."
        assert std_dev > 0, "[ERROR] The standard deviation must be positive."
    
        self.num_basis = num_basis
        self.step = grid_size / num_basis
        self.means = np.linspace(0, grid_size, self.num_basis)
        self.std_dev = std_dev
        
    def transform(self, state: Position) -> np.array:
        """
        Summary: 
            Compute the mapping from teh Position to the feature vector using 
            the gaussians

        Args:
            state (Position): the current position of the agent in the Grid 
            World Continuous Env

        Returns:
            np.array: feature mapping vector
        """
        feat_x = np.zeros(self.num_basis)
        feat_y = np.zeros(self.num_basis)
        for i, mean in enumerate(self.means):
            feat_x[i] = gauss(state.agent_pos.x, mean, self.std_dev)
            feat_y[i] = gauss(state.agent_pos.y, mean, self.std_dev)
        return np.concatenate((feat_x, feat_y))
