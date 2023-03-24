import numpy as np

def plan_minimum_jerk_trajectory(initial_pos, final_pos, traj_time, delta_t):
    coef = [0,0,0, 10/(traj_time**3), -15/(traj_time**4), 6/(traj_time**5)]
    poly = np.polynomial.polynomial.Polynomial(coef)

    tt = np.linspace(0,traj_time, int(traj_time/delta_t))

    
    traj = []

    displ = np.array(final_pos) - np.array(initial_pos)

    for t in tt:
        traj.append(np.append(initial_pos + displ * poly(t), 0))

    #print(f'Final pos: {final_pos}\t\tFinal Trajectory Pose: {traj[-1]}')

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