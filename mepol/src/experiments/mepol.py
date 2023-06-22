import argparse
import torch.nn as nn
import torch
import os
import numpy as np

from datetime import datetime
from mepol.src.envs.air_hockey import GymAirHockey
from utils.env_utils import NormalizedBoxEnv
from mepol.src.envs.discretizer import Discretizer
from mepol.src.envs.wrappers import ErgodicEnv
from mepol.src.algorithms.mepol import mepol
from mepol.src.policy import GaussianPolicy, train_supervised

parser = argparse.ArgumentParser(description='MEPOL')

parser.add_argument('--num_workers', type=int, default=1,
                    help='How many parallel workers to use when collecting env trajectories and compute k-nn')
parser.add_argument('--env', type=str, required=True,
                    help='The MDP')
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--zero_mean_start', type=int, default=1, choices=[0, 1],
                    help='Whether to make the policy start from a zero mean output')
parser.add_argument('--k', type=int, required=True,
                    help='The number of neighbors')
parser.add_argument('--kl_threshold', type=float, required=True,
                    help='The threshold after which the behavioral policy is updated')
parser.add_argument('--max_off_iters', type=int, default=20,
                    help='The maximum number of off policy optimization steps')
parser.add_argument('--use_backtracking', type=int, default=1, choices=[0, 1],
                    help='Whether to use backtracking or not')
parser.add_argument('--backtrack_coeff', type=float, default=2,
                    help='Backtrack coefficient')
parser.add_argument('--max_backtrack_try', type=int, default=10,
                    help='Maximum number of backtracking try')
parser.add_argument('--learning_rate', type=float, required=True,
                    help='The learning rate')
parser.add_argument('--num_trajectories', type=int, required=True,
                    help='The batch of trajectories used in off policy optimization')
parser.add_argument('--trajectory_length', type=int, required=True,
                    help='The maximum length of each trajectory in the batch of trajectories used in off policy optimization')
parser.add_argument('--num_epochs', type=int, required=True,
                    help='The number of epochs')
parser.add_argument('--optimizer', type=str, default='adam', choices=['rmsprop', 'adam'],
                    help='The optimizer')
parser.add_argument('--heatmap_every', type=int, default=10,
                    help='How many epochs to save a heatmap(if discretizer is defined).'
                         'Also the frequency at which policy weights are saved'
                         'Also the frequency at which full entropy is computed')
parser.add_argument('--heatmap_episodes', type=int, required=True,
                    help='The number of episodes on which the policy is run to compute the heatmap')
parser.add_argument('--heatmap_num_steps', type=int, required=True,
                    help='The number of steps per episode on which the policy is run to compute the heatmap')
parser.add_argument('--full_entropy_traj_scale', type=int, default=2,
                    help='The scale factor to be applied to the number of trajectories to compute the full entropy.')
parser.add_argument('--full_entropy_k', type=int, default=4,
                    help='The number of neighbors used to compute the full entropy')
parser.add_argument('--seed', type=int, default=None,
                    help='The random seed')
parser.add_argument('--tb_dir_name', type=str, default='mepol',
                    help='The tensorboard directory under which the directory of this experiment is put')
parser.add_argument('--log_dir', type=str, default='/data/air_hockey/thomas',
                    help='Where to store results')
parser.add_argument('-s', action='append', type=int, help='Append State index to State filter list')
parser.add_argument('-f', action='append', type=int, help='Append State index to Fallback State filter list')
parser.add_argument('--flag_heatmap', type=int, required=True, choices=[0, 1])

# environment parameters
parser.add_argument('--task_space', type=int, required=True, choices=[0, 1], help='Whether use task space actions')
parser.add_argument('--task_space_vel', type=int, required=True, choices=[0, 1], help='Use inv kinematics for velocity')
parser.add_argument('--use_delta_pos', type=int, required=True, choices=[0, 1], help='Use Delta Pos Actions')
parser.add_argument('--delta_dim', type=float, default=0.1)
parser.add_argument('--use_puck_distance', type=int, required=True, choices=[0, 1])
parser.add_argument('--normalize_obs', type=int, required=True, choices=[0, 1])
parser.add_argument('--scale_task_space_action', type=int, required=True, choices=[0, 1])

#policy parameters
parser.add_argument('--use_tanh', type=int, required=True, choices=[0, 1])
parser.add_argument('--policy_init', type=str, required=False, default=None, help='Path of the policy')
parser.add_argument('--log_std', type=float, required=False, default=-0.5)

args = parser.parse_args()

env_parameters = {k: vars(args)[k] for k in ('task_space', 'task_space_vel', 'use_delta_pos', 'delta_dim', 'use_puck_distance')}
policy_parameters = {k: vars(args)[k] for k in ('use_tanh',)}


def make_discretizer(env):
    if args.flag_heatmap:
        return Discretizer([[-0.974, 0.974],[-0.519,0.519]], [75,40], lambda s: [s[0],s[1]])
    return None


"""
Experiments specifications

    - env_create : callable that returns the target environment
    - discretizer_create : callable that returns a discretizer for the environment
    - hidden_sizes : hidden layer sizes
    - activation : activation function used in the hidden layers
    - log_std_init : log_std initialization for GaussianPolicy
    - state_filter : list of indices representing the set of features over which entropy is maximized
    - eps : epsilon factor to deal with knn aliasing
    - heatmap_inter : kind of interpolation used to draw the heatmap
    - heatmap_cmap : heatmap color map
    - heatmap_labels : names of the discretized features

"""
exp_spec = {
    'AirHockey': {
        'env_create': lambda: NormalizedBoxEnv(ErgodicEnv(GymAirHockey(**env_parameters))),
        'discretizer_create': make_discretizer,
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': args.log_std,
        'eps': 0,
        'state_filter': args.s,
        'fallback_state_filter': args.f,
        'heatmap_interp': 'spline16',
        'heatmap_cmap': 'Blues',
        'heatmap_labels': ('Puck_X', 'Puck_Y')
    },
}

spec = exp_spec.get(args.env)

if spec is None:
    print(f"Experiment name not found. Available ones are: {', '.join(key for key in exp_spec)}.")
    exit()

env = spec['env_create']()
discretizer = spec['discretizer_create'](env)
state_filter = spec.get('state_filter')
fallback_state_filter = spec.get('fallback_state_filter')
eps = spec['eps']


def create_policy(is_behavioral=False):
    policy = GaussianPolicy(
        num_features=env.num_features,
        hidden_sizes=spec['hidden_sizes'],
        action_dim=env.action_space.shape[0],
        activation=spec['activation'],
        log_std_init=spec['log_std_init'],
        **policy_parameters
    )

    if args.policy_init is not None:
        policy.load_state_dict(torch.load(args.policy_init))
        return policy

    if is_behavioral and args.zero_mean_start:
        policy = train_supervised(env, policy, train_steps=100, batch_size=5000)

    return policy


exp_name = f"exp_name={args.exp_name},env={args.env},z_mu_start={args.zero_mean_start},k={args.k},kl_thresh={args.kl_threshold}," \
           f"max_off_iters={args.max_off_iters},num_traj={args.num_trajectories},traj_len={args.trajectory_length}," \
           f"lr={args.learning_rate},opt={args.optimizer},fe_traj_sc={args.full_entropy_traj_scale},fe_k={args.full_entropy_k}," \
           f"use_bt={args.use_backtracking},bt_coeff={args.backtrack_coeff},max_bt_try={args.max_backtrack_try}"

out_path = os.path.join(args.log_dir, "exploration",
                        args.tb_dir_name, exp_name +
                        "__" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') +
                        "__" + str(os.getpid()))
os.makedirs(out_path, exist_ok=True)

with open(os.path.join(out_path, 'log_info.txt'), 'w') as f:
    f.write(f"Run info:\n")
    f.write("-" * 10 + "\n")

    for key, value in vars(args).items():
        f.write(f"{key}={value}\n")

    f.write("-" * 10 + "\n")

    tmp_policy = create_policy()
    f.write(tmp_policy.__str__())
    f.write("\n")

    if args.seed is None:
        args.seed = np.random.randint(2 ** 16 - 1)
        f.write(f"Setting random seed {args.seed}\n")

mepol(
    env=env,
    env_name=args.env,
    env_maker=spec['env_create'],
    state_filter=state_filter,
    state_filter_fallback=fallback_state_filter,
    create_policy=create_policy,
    k=args.k,
    kl_threshold=args.kl_threshold,
    max_off_iters=args.max_off_iters,
    use_backtracking=args.use_backtracking,
    backtrack_coeff=args.backtrack_coeff,
    max_backtrack_try=args.max_backtrack_try,
    eps=eps,
    learning_rate=args.learning_rate,
    num_traj=args.num_trajectories,
    traj_len=args.trajectory_length,
    num_epochs=args.num_epochs,
    optimizer=args.optimizer,
    full_entropy_traj_scale=args.full_entropy_traj_scale,
    full_entropy_k=args.full_entropy_k,
    heatmap_every=args.heatmap_every,
    heatmap_discretizer=discretizer,
    heatmap_episodes=args.heatmap_episodes,
    heatmap_num_steps=args.heatmap_num_steps,
    heatmap_cmap=spec.get('heatmap_cmap'),
    heatmap_labels=spec.get('heatmap_labels'),
    heatmap_interp=spec.get('heatmap_interp'),
    seed=args.seed,
    out_path=out_path,
    num_workers=args.num_workers
)