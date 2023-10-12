from dataclasses import dataclass, field
import numpy as np

@dataclass
class State:
    # State Elements wrt WORLD
    w_puck_pos: np.array(3) = field(init=True, default=np.zeros(3))
    w_puck_vel: np.array(3) = field(init=True, default=np.zeros(3))
    w_ee_pos: np.array(3) = field(init=True, default=np.zeros(3))
    w_ee_vel: np.array(3) = field(init=True, default=np.zeros(3))
    w_joint_pos: np.array(7) = field(init=True, default=np.zeros(7))
    w_joint_vel: np.array(7) = field(init=True, default=np.zeros(7))
    w_adv_ee_pos: np.array(3) = field(init=True, default=np.zeros(3))
    
    # State Elements wrt ROBOT
    r_puck_pos: np.array(3) = field(init=True, default=np.zeros(3))
    r_puck_vel: np.array(3) = field(init=True, default=np.zeros(3))
    r_ee_pos: np.array(3) = field(init=True, default=np.zeros(3))
    r_ee_vel: np.array(3) = field(init=True, default=np.zeros(3))
    r_joint_pos: np.array(7) = field(init=True, default=np.zeros(7))
    r_joint_vel: np.array(7) = field(init=True, default=np.zeros(7))
    r_adv_ee_pos: np.array(3) = field(init=True, default=np.zeros(3))
