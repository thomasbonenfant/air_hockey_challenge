import numpy as np
import torch.nn
from mepol.src.policy import GaussianPolicy
from mepol.src.envs.air_hockey import GymAirHockey
from mepol.src.envs.wrappers import ErgodicEnv
from gym.wrappers import ClipAction
from utils.trajectory_logger import TrajectoryLogger

spec = {
    'hidden_sizes':[400,300],
    'activation': torch.nn.ReLU,
    'log_std_init': -1.6,
}

env = ErgodicEnv(GymAirHockey(task_space=True, task_space_vel=False, use_delta_pos=True, use_puck_distance=True,
                              normalize_obs=True, scale_task_space_action=True))
policy = GaussianPolicy(
        num_features=env.num_features,
        hidden_sizes=spec['hidden_sizes'],
        action_dim=env.action_space.shape[0],
        activation=spec['activation'],
        log_std_init=spec['log_std_init'],
        use_tanh=True
    )

policy.load_state_dict(torch.load('/home/thomas/Downloads/800-policy'))

#Simulation
obs = env.reset()
done = False
steps = 0
data = []

while True:
    steps += 1
    action = np.array(policy.predict(obs, deterministic=False), dtype=float)
    obs, reward, done, _, info = env.step(action)


    env.render()

    if done or steps >= 400:
        break

