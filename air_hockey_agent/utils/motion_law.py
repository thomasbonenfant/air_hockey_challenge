import numpy as np


class MotionLaw:
    """
    This class implements a simple motion low to predict the position of the puck
    and reduce the effect of noise
    """

    def __init__(self, dt):
        # todo check if time is 1
        self.time = dt * 1e2  # dt in env_info is expressed in ms
        self.puck_pos = None
        self.puck_vel = None

        self.prev_puck_pos = None
        self.prev_puck_vel = None

    def get_prediction(self, puck_pos, puck_vel):
        """
            Predict the position of the puck using a motion law

            ds = v * dt

            Position and velocity are given in x and y components.

            Assume that the velocity will not change, air friction could be considered as well for a more
            precise prediction.
        """

        predicted_puck_pos = np.zeros((1, 3))

        ds = puck_vel * self.time

        predicted_puck_pos = puck_pos + ds

        return np.array(predicted_puck_pos)
