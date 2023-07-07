import pickle

import numpy as np
import torch.nn
from mepol.src.policy import GaussianPolicy
from mepol.src.envs.air_hockey import GymAirHockey, GymAirHockeyAcceleration
from mepol.src.envs.wrappers import ErgodicEnv
from utils.env_utils import NormalizedBoxEnv
import matplotlib.pyplot as plt

spec = {
    'hidden_sizes':[400,300],
    'activation': torch.nn.ReLU,
    'log_std_init': -0.5,
}

def visualize_multiple_acc(acc_param_list):

    data = []

    for acc_param in acc_param_list:
        # print(f'Trying max acceleration: {acc_param}')
        env = NormalizedBoxEnv(ErgodicEnv(
            GymAirHockeyAcceleration(env_name='7dof-hit', max_acceleration=acc_param,
                                     interpolation_order=3)))

        policy = GaussianPolicy(
                num_features=env.num_features,
                hidden_sizes=spec['hidden_sizes'],
                action_dim=env.action_space.shape[0],
                activation=spec['activation'],
                log_std_init=spec['log_std_init'],
                use_tanh=True
            )

        # Simulation
        obs = env.reset()
        done = False
        steps = 0
        ee_z = []

        base_env = env.wrapped_env.unwrapped.wrapped_env

        while True:
            steps += 1
            action = np.array([-1,0])

            obs, reward, done, _, info = env.step(action)

            ee_z.append(base_env.ee_z - base_env.ee_desired_height)

            env.render()

            if done or steps >= 500:
                env.reset()
                steps = 0
                data.append(ee_z)
                break

    with open("z_data.pkl", "wb") as fp:
        pickle.dump(data, fp)
    fig, ax = plt.subplots()

    for idx, acc in enumerate(acc_param_list):
        ax.plot([i for i in range(0, len(data[idx]))], data[idx], label=f'{acc}')

    ax.legend()
    plt.show()

if __name__ == '__main__':
    visualize_multiple_acc([500, 10, 15, 20])



