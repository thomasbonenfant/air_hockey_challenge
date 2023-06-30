import argparse
import torch
import torch.nn as nn
import os
import numpy as np

from datetime import datetime
from mepol.src.envs.air_hockey import GymAirHockey, GymAirHockeyAcceleration
from mepol.src.algorithms.trpo import trpo
from mepol.src.policy import GaussianPolicy
from utils.env_utils import NormalizedBoxEnv


parser = argparse.ArgumentParser(description='Goal-Based Reinforcement Learning - TRPO')

parser.add_argument('--num_workers', type=int, default=1,
                    help='How many parallel workers to use when collecting samples')
parser.add_argument('--env', type=str, required=True,
                    help='The MDP')
parser.add_argument('--policy_init', type=str, default=None,
                    help='Path to the weights for custom policy initialization.')
parser.add_argument('--num_epochs', type=int, required=True,
                    help='The number of training epochs')
parser.add_argument('--batch_size', type=int, required=True,
                    help='The batch size')
parser.add_argument('--traj_len', type=int, required=True,
                    help='The maximum length of a trajectory')
parser.add_argument('--gamma', type=float, default=0.995,
                    help='The discount factor')
parser.add_argument('--lambd', type=float, default=0.98,
                    help='The GAE lambda')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='The optimizer used for the critic, either adam or lbfgs')
parser.add_argument('--critic_lr', type=float, default=1e-2,
                    help='Learning rate for critic optimization')
parser.add_argument('--critic_reg', type=float, default=1e-3,
                    help='Regularization coefficient for critic optimization')
parser.add_argument('--critic_iters', type=int, default=5,
                    help='Number of critic full updates')
parser.add_argument('--critic_batch_size', type=int, default=64,
                    help='Mini batch in case of adam optimizer for critic optimization')
parser.add_argument('--cg_iters', type=int, default=10,
                    help='Conjugate gradient iterations')
parser.add_argument('--cg_damping', type=float, default=0.1,
                    help='Conjugate gradient damping factor')
parser.add_argument('--kl_thresh', type=float, required=True,
                    help='KL threshold')
parser.add_argument('--seed', type=int, default=None,
                    help='The random seed')
parser.add_argument('--tb_dir_name', type=str, default='goal_rl',
                    help='The tensorboard directory under which the directory of this experiment is put')
parser.add_argument('--log_dir', type=str, default='/data/air_hockey/thomas')

# environment parameters
parser.add_argument('--env_name', type=str, required=True, choices=['3dof-hit', '7dof-hit'])
parser.add_argument('--task_space', type=int, required=True, choices=[0, 1], help='Whether use task space actions')
parser.add_argument('--task_space_vel', type=int, required=True, choices=[0, 1], help='Use inv kinematics for velocity')
parser.add_argument('--use_delta_pos', type=int, required=True, choices=[0, 1], help='Use Delta Pos Actions')
parser.add_argument('--delta_dim', type=float, default=0.1)
parser.add_argument('--use_puck_distance', type=int, required=True, choices=[0, 1])
parser.add_argument('--max_acc', type=float, default=1, help='Max Acceleration for GymAirHockeyAcc')


#policy parameters
parser.add_argument('--use_tanh', type=int, required=True, choices=[0, 1])
parser.add_argument('--log_std', type=float, required=False, default=-0.5)

args = parser.parse_args()

env_parameters = {k: vars(args)[k] for k in ('env_name', 'max_acc', 'task_space', 'task_space_vel', 'use_delta_pos', 'delta_dim', 'use_puck_distance')}
policy_parameters = {k: vars(args)[k] for k in ('use_tanh',)}

"""
Sparse reward functions
"""


"""
Experiments specifications

    - env_create : callable that returns the target environment
    - hidden_sizes : hidden layer sizes
    - activation : activation function used in the hidden layers
    - log_std_init : log_std initialization for GaussianPolicy

"""
exp_spec = {
    'AirHockey': {
        'env_create': lambda: NormalizedBoxEnv(GymAirHockey(**env_parameters)),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': args.log_std,
    },
    'AirHockeyAcc': {
        'env_create': lambda: NormalizedBoxEnv(GymAirHockeyAcceleration(**env_parameters)),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': args.log_std,
    }

}

spec = exp_spec.get(args.env)

if spec is None:
    print(f"Experiment name not found. Available ones are: {', '.join(key for key in exp_spec)}.")
    exit()

env = spec['env_create']()

# Create a policy
policy = GaussianPolicy(
    num_features=env.num_features,
    hidden_sizes=spec['hidden_sizes'],
    action_dim=env.action_space.shape[0],
    activation=spec['activation'],
    log_std_init=spec['log_std_init'],
    **policy_parameters
)

# Create a critic
hidden_sizes = [64, 64]
hidden_activation = nn.ReLU
layers = []
for i in range(len(hidden_sizes)):
    if i == 0:
        layers.extend([
            nn.Linear(env.num_features, hidden_sizes[i]),
            hidden_activation()
        ])
    else:
        layers.extend([
            nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
            hidden_activation()
        ])

layers.append(nn.Linear(hidden_sizes[i], 1))
vfunc = nn.Sequential(*layers)

for module in vfunc:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)


if args.policy_init is not None:
    kind = 'MEPOLInit'
    policy.load_state_dict(torch.load(args.policy_init))
else:
    kind = 'RandomInit'


exp_name = f"env={args.env},init={kind}"

out_path = os.path.join(args.log_dir, "goal_rl",
                        args.tb_dir_name, exp_name +
                        "__" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') +
                        "__" + str(os.getpid()))
os.makedirs(out_path, exist_ok=True)

with open(os.path.join(out_path, 'log_info.txt'), 'w') as f:
    f.write("Run info:\n")
    f.write("-"*10 + "\n")

    for key, value in vars(args).items():
        f.write("{}={}\n".format(key, value))

    f.write("-"*10 + "\n")

    f.write(policy.__str__())
    f.write("-"*10 + "\n")
    f.write(vfunc.__str__())

    f.write("\n")

    if args.seed is None:
        args.seed = np.random.randint(2**16-1)
        f.write("Setting random seed {}\n".format(args.seed))

trpo(
    env_maker=spec['env_create'],
    env_name=args.env,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    traj_len=args.traj_len,
    gamma=args.gamma,
    lambd=args.lambd,
    vfunc=vfunc,
    policy=policy,
    optimizer=args.optimizer,
    critic_lr=args.critic_lr,
    critic_reg=args.critic_reg,
    critic_iters=args.critic_iters,
    critic_batch_size=args.critic_batch_size,
    cg_iters=args.cg_iters,
    cg_damping=args.cg_damping,
    kl_thresh=args.kl_thresh,
    num_workers=args.num_workers,
    out_path=out_path,
    seed=args.seed
)