import numpy as np
import matplotlib.pyplot as plt
import os

def plot(experiment, env_name='3dof-defend', n_joints=3):
    path = os.path.join('.', 'logs', experiment, env_name)
    #computation_time = np.load(os.path.join(path, 'computation_time.npy'))
    #ee_constr = np.load(os.path.join(path, 'ee_constr.npy'))
    jerk = np.load(os.path.join(path, 'jerk.npy'))
    #joint_pos_constr = np.load(os.path.join(path, 'joint_pos_constr.npy'))
    #joint_vel_constr = np.load(os.path.join(path, 'joint_vel_constr.npy'))

    steps = jerk.shape[0]
    x = np.linspace(1, steps, steps)

    fig, axs = plt.subplots(n_joints, figsize=(6,6))
    for i in range(n_joints):
        axs[i].plot(x,jerk[:,i], label=f'Joint: {i+1}')
        axs[i].legend()

    plt.show()

if __name__ == '__main__':
    plot('eval-2023-03-27_15-45-30')



