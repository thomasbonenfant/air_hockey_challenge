import numpy as np
from utils.trajectory_planner import Planner, make_a_f, make_v_f, make_d_f

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
