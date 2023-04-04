import numpy as np
from qpsolvers import solve_qp

'''
Plans a trajectory with minimum jerk
the main is just an example but the funcion is called in the defend_agent
'''

def plan_minimum_jerk_trajectory(initial_pos, final_pos, traj_time, delta_t):
    coef = [0,0,0, 10/(traj_time**3), -15/(traj_time**4), 6/(traj_time**5)]
    poly = np.polynomial.polynomial.Polynomial(coef)

    tt = np.linspace(0, traj_time, int(traj_time/delta_t))
    
    traj = []

    displ = np.array(final_pos) - np.array(initial_pos)

    for t in tt:
        traj.append(np.append(initial_pos + displ * poly(t), 0)) # append x,y,z with z=0

    return np.array(traj)


class Planner(object):
    def __init__(self, n_omega, C1, C2):
        self.n_omega = n_omega
        self.C1 = C1
        self.C2 = C2

    def plan(self, T, vf, df, af, v0, a0, a_min, a_max, v_min, v_max, dt=0.02, slack=False):
        n_params = self.n_omega +2 if slack else self.n_omega

        vf = np.clip(vf, v_min, v_max)
        af = np.clip(af, a_min, a_max)

        P = np.zeros((n_params, n_params))
        q = np.zeros(n_params)
        A = np.zeros((2, n_params))
        b = np.array([df-v0*T - 0.5*a0*T**2, vf-v0-a0*T])#, af-a0])
        G = np.zeros((0,n_params))
        h = np.array([])

        t = dt

        while t < T:
            temp = np.zeros((1, n_params), dtype=np.dtype(float))
            for i in range(0, self.n_omega):
                temp[0, i] = t ** (i + 1)
            G = np.concatenate((G, temp), axis=0)
            G = np.concatenate((G, -temp), axis=0)
            h = np.concatenate((h, np.array([a_max - a0])), axis=0)
            h = np.concatenate((h, np.array([-a_min + a0])), axis=0)

            temp = np.zeros(shape=(1, n_params), dtype=np.dtype(float))
            for j in range(0, self.n_omega):
                temp[0, j] = t ** (j + 2) / (j + 2)
            G = np.concatenate((G, temp), axis=0)
            G = np.concatenate((G, -temp), axis=0)
            h = np.concatenate((h, np.array([v_max - v0 - a0 * t])), axis=0)
            h = np.concatenate((h, np.array([-v_min +v0 + a0 * t])), axis=0)

            t += dt
        
        for i in range(self.n_omega):
            for j in range(self.n_omega):
                P[i,j] = (((i + 1)*(j + 1))/(i + j + 1))*(T**(i+j+1))
        
        if slack:
            P[-2,-2] = self.C1
            P[-1,-1] = self.C2
        
        for i in range(self.n_omega):
            A[0,i] = (T**(i+3))/((i+2)*(i+3))
            A[1,i] = (T**(i+2))/(i+2)
            #A[2,i] = (T**i)
        if slack:
            A[0,-2] = -1
            A[1,-1] = -1
        
        weights = solve_qp(P, q, A=A,b=b,G=G, h=h, solver='osqp')
        return weights
    

def make_a_f(a0, theta):
    num_theta = len(theta)

    def a(t):
        return a0 + np.sum([t ** (j + 1) * theta[j] for j in range(num_theta)])

    return a

def make_v_f(v0, a0, theta):
    num_theta = len(theta)

    def v(t):
        return v0 + a0 * t + np.sum([t ** (j + 2) * theta[j] / (j + 2) for j in range(num_theta)])

    return v


def make_d_f(d0, v0, a0, theta):
    num_theta = len(theta)

    def d(t):
        return d0 + v0 * t + 0.5 * a0 * (t ** 2) + np.sum(
            [t ** (j + 1) * theta[j - 2] / ((j + 1) * j) for j in range(2, num_theta + 2)])

    return d

import matplotlib.pyplot as plt

if __name__ == '__main__':

    traj_time = 1
    delta_t = 0.02

    planner = Planner(3, 1e4, 1e4)
    a0 = 10
    af = 0
    v0 = -5
    vf = 0
    df = 1
    a_min = -10
    a_max = 10
    v_min = -10
    v_max = 10
    weights = planner.plan(traj_time, vf, df, af, v0, a0, a_min, a_max, v_min, v_max, dt=delta_t, slack=False)

    if weights is None:
        weights = planner.plan(traj_time, vf, df, af, v0, a0, a_min, a_max, v_min, v_max, dt=delta_t, slack=True)


    d = make_d_f(0,v0,a0, weights)
    v = make_v_f(v0, a0, weights)
    a = make_a_f(a0, weights)

    tt = np.linspace(0,traj_time)
    aa = np.array([a(t) for t in tt])

    jerk = np.array([])
    for i in range(1, len(aa)):
        jerk = np.hstack((jerk, np.array([(aa[i] - aa[i-1])/delta_t])))

    plt.plot(tt,aa, label='a')
    plt.plot(tt,np.array([d(t) for t in tt]), label='d')
    plt.plot(tt,np.array([v(t) for t in tt]), label='v')
    plt.plot(tt[1:], jerk, label='jerk')
    plt.legend()
    plt.show()
