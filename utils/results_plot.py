import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def plot(experiment, env_name='3dof-defend', n_joints=3):
    path = os.path.join('.', 'logs', experiment, env_name)

    jerk = np.load(os.path.join(path, 'jerk.npy'))/1e7
    dataset = []

    joint_pos = []
    joint_vel = []
    actions = []


    #open dataset.pkl in binary mode and load dataset using pickle
    with open(os.path.join(path, 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)

    prev_state, action, _, _, _, _ = dataset[0]

    #joint_vel.append([prev_state[9], prev_state[10], prev_state[11]])
    #actions.append([[0,0,0],[0,0,0]])

    for i in range(len(dataset)):
        prev_state, action, reward, reached_state, abs_flag, last_flag = dataset[i]
        joint_vel.append([reached_state[9], reached_state[10],reached_state[11]])
        joint_pos.append([reached_state[6], reached_state[7], reached_state[8]])
        actions.append(action)
        print(action)

    steps = jerk.shape[0]
    x = np.linspace(1, len(dataset), len(dataset))

    joint_vel = np.array(joint_vel)
    joint_pos = np.array(joint_pos)
    actions = np.array(actions)

    fig, axs = plt.subplots(n_joints, figsize=(6,6))
    for i in range(n_joints):
        axs[i].plot(x,jerk[:,i], label='Jerk / 1e7')
        axs[i].plot(x, joint_vel[:,i], label='Velocity')
        axs[i].plot(x, joint_pos[:,i], label='Position')
        #axs[i].plot(x, actions[:,0,i], label='Joint Pos Action')
        #axs[i].plot(x, actions[:,1,i], label='Joint Vel Action')

        axs[i].legend()

    plt.show()

if __name__ == '__main__':
    plot('eval-2023-03-28_14-10-58')



