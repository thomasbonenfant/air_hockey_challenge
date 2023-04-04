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

    steps = jerk.shape[0]
    x = np.linspace(1, len(dataset), len(dataset))

    joint_vel = np.array(joint_vel)
    joint_pos = np.array(joint_pos)
    actions = np.array(actions)

    fig, axs = plt.subplots(n_joints, figsize=(6,6), sharex=True, sharey=True)
    for i in range(n_joints):
        axs[i].plot(x,jerk[:,i], label='Jerk / 1e7')
        axs[i].plot(x, joint_vel[:,i], label='Velocity')
        axs[i].plot(x, joint_pos[:,i], label='Position')
        #axs[i].plot(x, actions[:,0,i], label='Joint Pos Action')
        #axs[i].plot(x, actions[:,1,i], label='Joint Vel Action')

        axs[i].legend()

    plt.show()

def custom_dataset_plot(log_dir, n_joints=3):
    with open(os.path.join('.', 'logs', log_dir), 'rb') as f:
        joint_action, ee_action, joint_pos, joint_vel, joint_acc, ee_pos, ee_vel, ee_acc, joint_jerk = pickle.load(f)

    joint_jerk = np.array(joint_jerk)
    joint_pos = np.array(joint_pos)
    joint_vel = np.array(joint_vel)
    joint_acc = np.array(joint_acc)
    joint_action = np.array(joint_action)
    ee_pos = np.array(ee_pos)
    ee_vel = np.array(ee_vel)
    ee_acc = np.array(ee_acc)
    ee_action = np.array(ee_action)


    ee_jerk = np.array([(ee_acc[i] - ee_acc[i - 1])/0.02 for i in range(1, len(ee_action))])
    ee_jerk_norm = np.array([np.linalg.norm((ee_acc[i] - ee_acc[i - 1])/0.02) for i in range(1, len(ee_action))])

    fig0, axs = plt.subplots(n_joints, 2, figsize=(15,9), sharex=True)
    fig0.suptitle('Joints Plot')
    for i in range(n_joints):
        axs[i,0].plot(joint_jerk[:,i], label='Jerk')
        axs[i,0].legend()
        axs[i,1].plot(joint_vel[:,i], label='Velocity')
        axs[i,1].plot(joint_action[:,0,i], label='Position Action')
        axs[i,1].plot(joint_action[:,1,i], label='Velocity Action')
        axs[i,1].plot(joint_pos[:,i], label='Position')
        axs[i,1].plot(joint_acc[:,i], label='Acceleration')
        axs[i,1].legend()
    
    
    fig1, axs = plt.subplots(3, figsize=(15,6), sharey=False, sharex=True)
    fig1.suptitle('End Effector Plot')
    axs[0].plot(ee_pos[:,0], label='x EE')
    axs[0].plot(ee_action[:,0], label='action x EE')
    axs[0].plot(ee_vel[:,0], label='dx/dt EE')
    axs[0].plot(ee_acc[:,0], label='ddx/dt EE')
    #axs[0].plot([i for i in range(1, len(ee_acc) - 1)], ee_jerk[:,0], label='jerk')

    axs[0].legend()
    axs[1].plot(ee_pos[:,1], label='y EE')
    axs[1].plot(ee_action[:,1], label='action y EE')
    axs[1].plot(ee_vel[:,1], label='dy/dt EE')
    axs[1].plot(ee_acc[:,1], label='ddy/dt EE')
    #axs[1].plot([i for i in range(1, len(ee_acc) - 1)], ee_jerk[:,1], label='jerk')
    axs[1].legend()

    axs[2].plot([i for i in range(1, len(ee_acc) - 1)], ee_jerk_norm, label='jerk norm')



    plt.show()
        
        



if __name__ == '__main__':
    custom_dataset_plot('custom_log_2023-04-04_13-19-20.pkl')



