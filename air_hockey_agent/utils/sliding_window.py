"""
This class implements a moving window to reduce the noise effect in the puck tracking

The class will save the first N steps and than produce a state, for puck_pos and puck_vel, that is
the average of the observed values.
"""
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt
from numpy.polynomial.polynomial import Polynomial
from numpy.linalg import LinAlgError
from sklearn import linear_model

from statsmodels.nonparametric.smoothers_lowess import lowess
#import statsmodels.api as sm

import warnings
warnings.filterwarnings("error")  # Consider warnings as errors

USE_LINEAR_REGRESSION = False


class SlidingWindow:

    def __init__(self, window_size):
        self.window_size = window_size
        self.queue_x = deque()
        self.queue_y = deque()
        self.queue_theta = deque()

        self._puck_still_queue = deque()  # queue with puck pos if the puck is still

        # Puck is still
        self.is_puck_still = False

        # Line coefficients
        self.m = None
        self.q = None

        # Linear model
        self.lin_reg_model_x = linear_model.LinearRegression()
        self.lin_reg_model_y = linear_model.LinearRegression()
        self.model = linear_model.LinearRegression()

    def _reset(self):
        """
            Reset all the queues to initial conditions
        """
        self.queue_x = deque()
        self.queue_y = deque()
        self.queue_theta = deque()

        # Puck is still
        self.is_puck_still = False
        self._puck_still_queue = deque()

        # Line coefficients
        self.m = None
        self.q = None

        self.model = linear_model.LinearRegression()

    def get_queue(self):
        return self.queue_x, self.queue_y, self.queue_theta

    def get_line_coefficients(self):
        return self.m, self.q

    def get_mean(self, obs):
        if len(self.queue_x) > 0:
            mean_x = np.mean(self.queue_x)
            mean_y = np.mean(self.queue_y)
            mean_theta = np.mean(self.queue_theta)

            mean_point = np.array([mean_x, mean_y, mean_theta])

        else:
            mean_point = obs

        return mean_point

    def append_element(self, element):
        """
            Add an element to the queue: if it is already full it removes
            the oldest element and adds the new one

            The queue grows from left to right: leftmost element is the oldest
            rightmost the newest

            In the end it updates the coefficients of the line that better fits the
            observed queues
        """

        '''
        # Do not add element if it is the same as before
        if (len(self.queue_x) > 0) and (self.queue_x[-1] == element[0]) and (self.queue_y[-1] == element[1]):
            print('x-1: ', self.queue_x[-1])
            print('element: ', element[0])

            return
        '''

        # Remove the oldest element and add a new one
        if len(self.queue_x) == self.window_size:
            self.queue_x.popleft()
            self.queue_x.append(element[0])

            self.queue_y.popleft()
            self.queue_y.append(element[1])

            self.queue_theta.popleft()
            self.queue_theta.append(element[2])

            assert len(self.queue_x) == self.window_size
            assert len(self.queue_y) == self.window_size
            assert len(self.queue_theta) == self.window_size

        else:
            # fill the queue
            self.queue_x.append(element[0])
            self.queue_y.append(element[1])
            self.queue_theta.append(element[2])

        if self.is_puck_still:
            self._puck_still_queue.append(element)

        # Update coefficients given the new observation
        self._update_coefficients()

    def predict_new_point(self, obs):
        """
            Predict the new point given the observation
        """

        # Do not update observation if puck is still
        if self.m is None or self.q is None:
            return obs

        '''
        x_obs = obs[0]
        y_obs = obs[1]

        x_pred = (y_obs - self.q) / self.m
        y_pred = self.m * x_obs + self.q

        if not self.is_puck_still:
            new_point = np.array([
                np.mean([x_pred, x_obs]), np.mean([y_pred, y_obs])
            ])
        else:
            self.is_puck_still = False
            return np.mean(self._puck_still_queue, axis=0)
        '''
        '''
        if len(self.queue_x) == self.window_size and USE_LINEAR_REGRESSION:
            # predict new point using linear regression
            print('LIN REG')
            self.lin_reg_model_x.fit(self.queue_x, self.queue_y)
            self.lin_reg_model_y.fit(self.queue_y, self.queue_x)

            new_x = self.lin_reg_model_y.predict(obs[1])
            new_y = self.lin_reg_model_x.predict(obs[0])

            new_point = np.array([new_x, new_y, obs[2]])
            return new_point
        '''

        # Return the new point but with the same theta
        #return np.append(new_point, obs[2])
        return np.array([obs[0], self.model.predict(obs[0].reshape(-1, 1)), obs[2]])

    def filter_predict(self, obs):
        # Filter parameters
        T = 1.0          # sample period
        fs = 20          # sample rate, Hz
        cutoff = 2       # desired cutoff frequency (we don't know the signal frequency
                         # cutoff should be a little higher)
        nyq = 0.5 * fs   # Nyquist frequency
        order = 1        # approx order
        n = int(T * fs)  # total number of samples

        def butter_lowpass_filter(data, cutoff, fs, order):
            normal_cutoff = cutoff / nyq
            # Get filter coefficients
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            print(b, a)
            y = filtfilt(b, a, data)
            return y

        x_filtered = butter_lowpass_filter(self.queue_x, cutoff, fs, order)
        y_filtered = butter_lowpass_filter(self.queue_y, cutoff, fs, order)

        new_point = np.array([x_filtered, y_filtered, obs[2]])

        return new_point

    def _update_coefficients(self):
        """
            Updates the coefficients of the line
        """

        '''
        z = [None, None]

        # Retrieve the best fitting line
        try:
            #z = np.flip(Polynomial.fit(self.queue_x, self.queue_y, 1).convert().coef)
            z = np.polyfit(self.queue_x, self.queue_y, 1)
            if self.is_puck_still:
                self.is_puck_still = False
                self._puck_still_queue = deque()
        except np.RankWarning:
            # Do not update observation if puck is still
            self.is_puck_still = True
            self.m = None
            self.q = None
        '''


        self.model.fit(np.array(self.queue_x).reshape(-1, 1), np.array(self.queue_y))

        # Check if there was a collision
        are_queues_reset = self._check_collision(self.model.coef_[0])

        if not are_queues_reset:
            #self.m = z[0]
            #self.q = z[1]
            self.m = self.model.coef_[0]
            self.q = self.model.intercept_

    def _check_collision(self, new_angular_coeff):
        """
            Checks if a collision happened, in that case resets the queues
        """

        collision_tolerance = 0.1

        # Do not update observation if puck is still
        if self.m is None or self.q is None:
            return False

        if np.abs(self.m - new_angular_coeff) > collision_tolerance:
            self._reset()
            return True
        else:
            return False


class SlidingWindowJoints:
    def __init__(self, window_size=None):
        self.window_size = window_size

        self.joint_queues = dict()
        self.joint_queues["0"] = deque()
        self.joint_queues["1"] = deque()
        self.joint_queues["2"] = deque()
        self.joint_queues["3"] = deque()
        self.joint_queues["4"] = deque()
        self.joint_queues["5"] = deque()
        self.joint_queues["6"] = deque()

    def reset(self):
        self.joint_queues = dict()
        self.joint_queues["0"] = deque()
        self.joint_queues["1"] = deque()
        self.joint_queues["2"] = deque()
        self.joint_queues["3"] = deque()
        self.joint_queues["4"] = deque()
        self.joint_queues["5"] = deque()
        self.joint_queues["6"] = deque()

    def append_element(self, joint_obs):

        assert len(joint_obs) == 7

        for index, element in enumerate(joint_obs):

            dict_index = str(index)

            if len(self.joint_queues[dict_index]) == self.window_size:
                self.joint_queues[dict_index].popleft()
                self.joint_queues[dict_index].append(element)

                assert len(self.joint_queues[dict_index]) == self.window_size

            else:
                self.joint_queues[str(index)].append(element)

    def lowess_smoothing(self):

        filtered_y = []

        if len(self.joint_queues["0"]) < self.window_size:
            for queue in self.joint_queues:
                filtered_y.append(self.joint_queues[queue][-1])  # return the latest element

        else:
            for queue in self.joint_queues:
                x = list(range(0, len(self.joint_queues[queue])))
                y = self.joint_queues[queue]

                filtered_y.append(lowess(y, x, frac=0.8, return_sorted=True)[:, 1])

        return np.array(filtered_y).reshape(1, 7)
