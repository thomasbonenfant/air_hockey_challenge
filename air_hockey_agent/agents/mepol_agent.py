import numpy as np
import torch.nn
from mepol.src.policy import GaussianPolicy
from mepol.src.envs.air_hockey import GymAirHockey, GymAirHockeyAcceleration
from mepol.src.envs.wrappers import ErgodicEnv
from utils.env_utils import NormalizedBoxEnv

spec = {
    'hidden_sizes':[400,300],
    'activation': torch.nn.ReLU,
    'log_std_init': -0.5,
}

#env = ErgodicEnv(GymAirHockey(env_name='7dof-hit', interpolation_order=3, task_space=True, task_space_vel=True, use_delta_pos=True, use_puck_distance=True,
#                             delta_dim=0.1))
env = ErgodicEnv(GymAirHockeyAcceleration(env_name='7dof-hit', max_acceleration=500, interpolation_order=3))

env = NormalizedBoxEnv(env)

policy = GaussianPolicy(
        num_features=env.num_features,
        hidden_sizes=spec['hidden_sizes'],
        action_dim=env.action_space.shape[0],
        activation=spec['activation'],
        log_std_init=spec['log_std_init'],
        use_tanh=True
    )

#policy.load_state_dict(torch.load('/home/thomas/air_hockey_challenge/mepol/results/exploration/mepol/delta_0.3_on_policy_only_puck/1100-policy'))
#policy.load_state_dict(torch.load('/home/thomas/air_hockey_challenge/mepol/results/exploration/mepol/delta_on_policy_0.4/3000-policy'))
#policy.load_state_dict(torch.load('/home/thomas/Downloads/100-policy'))

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
        env.reset()
        steps = 0

