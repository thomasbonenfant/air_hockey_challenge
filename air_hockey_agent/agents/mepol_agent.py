import torch
import torch.nn
from mepol.src.policy import GaussianPolicy
from mepol.src.envs.wrappers import ErgodicEnv
from air_hockey_challenge.framework.gym_air_hockey import GymAirHockey

spec = {
    'hidden_sizes':[400,300],
    'activation': torch.nn.ReLU,
    'log_std_init': -0.5,
}

env = GymAirHockey()
policy = GaussianPolicy(
        num_features=env.num_features,
        hidden_sizes=spec['hidden_sizes'],
        action_dim=env.action_space.shape[0],
        activation=spec['activation'],
        log_std_init=spec['log_std_init']
    )

policy.load_state_dict(torch.load('ENTER PATH'))

#Simulation
obs = env.reset()
done = False

while not done:
    action = policy.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()



