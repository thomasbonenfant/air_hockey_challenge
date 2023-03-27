import numpy as np
from scipy.interpolate import CubicSpline

'''
Plans a trajectory with minimum jerk
the main is just an example but the funcion is called in the defend_agent
'''

def plan_minimum_jerk_trajectory(initial_pos, final_pos, traj_time, delta_t):
    #coef = [0,0,0, 10/(traj_time**3), -15/(traj_time**4), 6/(traj_time**5)]
    #poly = np.polynomial.polynomial.Polynomial(coef)

    tt = np.linspace(1, traj_time, int(traj_time/delta_t)) # can't start from 0 because it will be used as a divider

    traj = []

    displ = np.array(final_pos) - np.array(initial_pos)

    for t in tt:
        coef = [0,0,0, 10/((t/traj_time)**3), -15/((t/traj_time)**4), 6/((t/traj_time)**5)] # regularize the coefficients according to the current timestamp
        poly = np.polynomial.polynomial.Polynomial(coef)
        traj.append(np.append(initial_pos + displ * poly(t), 0)) # append x,y,z with z=0

    return np.array(traj)


import matplotlib.pyplot as plt

if __name__ == '__main__':
    initial_pos = [0,0]
    final_pos = [5,3]

    traj_time = 800
    delta_t = 0.02

    traj = plan_minimum_jerk_trajectory(initial_pos, final_pos, traj_time, delta_t)
    plt.plot(np.linspace(0,5, int(traj_time/delta_t)),traj[:,1])
    plt.show()