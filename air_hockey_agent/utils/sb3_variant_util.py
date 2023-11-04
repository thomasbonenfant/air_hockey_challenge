from argparse import ArgumentParser, Namespace
import random
import json
import os


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="algorithm", required=True)
    env_group = parser.add_argument_group('env')

    # log args

    parser.add_argument("--save_model_dir", type=str, default="../models")
    parser.add_argument("--experiment_label", type=str, default="")
    #parser.add_argument("--alg", type=str, default="ppo")

    # eval args

    parser.add_argument("--eval_freq", type=int, default=20480)
    parser.add_argument("--n_eval_episodes", type=int, default=80)

    # env arguments
    env_group.add_argument("--env", choices=['hrl', 'hit'], default='hrl')
    env_group.add_argument("--steps_per_action", type=int, default=100)
    env_group.add_argument("--render", action="store_true")
    env_group.add_argument("--include_timer", action="store_true")
    env_group.add_argument("--include_ee", action="store_true")
    env_group.add_argument("--include_faults", action="store_true")
    env_group.add_argument("--include_joints", action="store_true")
    env_group.add_argument("--include_puck", action="store_true")
    env_group.add_argument("--remove_last_joint", action="store_true")
    env_group.add_argument("--large_reward", type=float, default=100)
    env_group.add_argument("--fault_penalty", type=float, default=33.33)
    env_group.add_argument("--fault_risk_penalty", type=float, default=0.1)
    env_group.add_argument("--scale_obs", action="store_true")
    env_group.add_argument("--alpha_r", type=float, default=1.)
    env_group.add_argument("--parallel", type=int, default=1)

    #parser_hit.add_argument("--include_joints", action="store_true")
    #parser_hit.add_argument('--include_ee', action="store_true")
    env_group.add_argument('--include_ee_vel', action="store_true")
    env_group.add_argument('--scale_action', action='store_true')
    env_group.add_argument('--hit_coeff', type=float, default=1000.0)
    env_group.add_argument('--max_path_len', type=int, default=400)
    #parser_hit.add_argument("--scale_obs", action="store_true")
    #parser_hit.add_argument("--alpha_r", type=float, default=1.0)
    #parser_hit.set_defaults(env="hit")

    # learning args

    parser.add_argument("--total_timesteps", type=int, required=True)
    # parser.add_argument("--tb_log_name", type=str)

    # ppo arguments
    parser_ppo = subparsers.add_parser("ppo")
    parser_ppo.add_argument("--policy", type=str, default="MlpPolicy")
    parser_ppo.add_argument("--lr", type=float, default=3e-4)
    parser_ppo.add_argument("--steps_per_update", type=int, default=32)
    parser_ppo.add_argument("--batch_size", type=int, default=64)
    parser_ppo.add_argument("--n_epochs", type=int, default=10)
    parser_ppo.add_argument("--gamma", type=float, default=0.999)
    parser_ppo.add_argument("--gae_lambda", type=float, default=0.95)
    parser_ppo.add_argument("--clip_range", type=float, default=0.2)
    parser_ppo.add_argument("--clip_range_vf", type=float)
    parser_ppo.add_argument("--normalize_advantage", action="store_true", default=True)
    parser_ppo.add_argument("--ent_coef", type=float, default=0.0)
    parser_ppo.add_argument("--vf_coef", type=float, default=0.5)
    parser_ppo.add_argument("--max_grad_norm", type=float, default=0.5)
    parser_ppo.add_argument("--use_sde", action="store_true", default=False)
    parser_ppo.add_argument("--sde_sample_freq", type=int, default=-1)
    parser_ppo.add_argument("--target_kl", type=float)
    parser_ppo.add_argument("--stats_window_size", type=int, default=100)
    parser_ppo.add_argument("--tensorboard_log", type=str)
    parser_ppo.add_argument("--verbose", type=int, default=1)
    parser_ppo.add_argument("--seed", type=int)
    parser_ppo.set_defaults(alg='ppo')

    #sac arguments
    parser_sac = subparsers.add_parser("sac")
    parser_sac.add_argument("--policy", type=str, default="MlpPolicy")
    parser_sac.add_argument("--learning_rate", type=float, default=3e-4)
    parser_sac.add_argument("--buffer_size", type=int, default=1000000)
    parser_sac.add_argument("--learning_starts", type=int, default=1000)
    parser_sac.add_argument("--batch_size", type=int, default=256)
    parser_sac.add_argument("--tau", type=float, default=0.005)
    parser_sac.add_argument("--gamma", type=float, default=0.99)
    parser_sac.add_argument("--train_freq", type=int, default=1)
    parser_sac.add_argument("--gradient_steps", type=int, default=1)
    parser_sac.add_argument("--action_noise", type=str, default=None)
    parser_sac.add_argument("--replay_buffer_class", type=str, default=None)
    parser_sac.add_argument("--replay_buffer_kwargs", type=json.loads, default=None)
    parser_sac.add_argument("--optimize_memory_usage", action="store_true")
    parser_sac.add_argument("--ent_coef", type=str, default='auto')
    parser_sac.add_argument("--target_update_interval", type=int, default=1)
    parser_sac.add_argument("--target_entropy", type=str, default=None)
    parser_sac.add_argument("--use_sde", action="store_true")
    parser_sac.add_argument("--sde_sample_freq", type=int, default=-1)
    parser_sac.add_argument("--use_sde_at_warmup", action="store_true")
    parser_sac.add_argument("--stats_window_size", type=int, default=100)
    parser_sac.add_argument("--tensorboard_log", type=str, default=None)
    parser_sac.add_argument("--policy_kwargs", type=json.loads, default=None)
    parser_sac.add_argument("--verbose", type=int, default=1)
    parser_sac.add_argument("--seed", type=int, default=None)
    parser_sac.add_argument("--device", type=str, default="auto")
    parser_sac.add_argument("--_init_setup_model", action="store_true")
    parser_sac.set_defaults(alg='sac')

    #dqn arguments
    parser_dqn = subparsers.add_parser("dqn")
    parser_dqn.add_argument("--policy", type=str, default="MlpPolicy")
    parser_dqn.add_argument("--learning_rate", type=float, default=0.001)
    parser_dqn.add_argument("--buffer_size", type=int, default=10000)
    parser_dqn.add_argument("--learning_starts", type=int, default=1000)
    parser_dqn.add_argument("--batch_size", type=int, default=32)
    parser_dqn.add_argument("--tau", type=float, default=1.0)
    parser_dqn.add_argument("--gamma", type=float, default=0.99)
    parser_dqn.add_argument("--train_freq", type=int, default=1)
    parser_dqn.add_argument("--gradient_steps", type=int, default=1)
    parser_dqn.add_argument("--replay_buffer_class", type=str, default=None)
    parser_dqn.add_argument("--replay_buffer_kwargs", type=json.loads, default=None)
    parser_dqn.add_argument("--optimize_memory_usage", action="store_true")
    parser_dqn.add_argument("--target_update_interval", type=int, default=100)
    parser_dqn.add_argument("--exploration_fraction", type=float, default=0.1)
    parser_dqn.add_argument("--exploration_initial_eps", type=float, default=1.0)
    parser_dqn.add_argument("--exploration_final_eps", type=float, default=0.02)
    parser_dqn.add_argument("--max_grad_norm", type=float, default=10.0)
    parser_dqn.add_argument("--stats_window_size", type=int, default=100)
    parser_dqn.add_argument("--tensorboard_log", type=str, default=None)
    parser_dqn.add_argument("--policy_kwargs", type=json.loads, default=None)
    parser_dqn.add_argument("--verbose", type=int, default=1)
    parser_dqn.add_argument("--seed", type=int, default=None)
    parser_dqn.add_argument("--device", type=str, default="auto")
    parser_dqn.add_argument("--_init_setup_model", action="store_true", default=True)
    parser_dqn.set_defaults(alg='dqn')


    variant = parser.parse_args()

    if not variant.seed:
        variant.seed = random.randint(0, 999999)

    return variant


def variant_util(variant):
    env_args = {}
    alg_args = {}
    learn_args = {}
    log_args = {}

    log_args['save_model_dir'] = variant.save_model_dir
    log_args['experiment_label'] = variant.experiment_label
    log_args['alg'] = variant.alg

    env_args['env'] = variant.env

    if variant.env == "hrl":
        env_args['steps_per_action'] = variant.steps_per_action
        env_args['render'] = variant.render
        env_args['include_timer'] = variant.include_timer
        env_args['include_ee'] = variant.include_ee
        env_args['include_faults'] = variant.include_faults
        env_args['large_reward'] = variant.large_reward
        env_args['fault_penalty'] = variant.fault_penalty
        env_args['fault_risk_penalty'] = variant.fault_risk_penalty
        env_args['scale_obs'] = variant.scale_obs
        env_args['alpha_r'] = variant.alpha_r
        env_args['include_joints'] = variant.include_joints
    elif variant.env == "hit":
        env_args['include_ee'] = variant.include_ee
        env_args['include_ee_vel'] = variant.include_ee_vel
        env_args['include_joints'] = variant.include_joints
        try:
            env_args['include_puck'] = variant.include_puck
            env_args['remove_last_joint'] = variant.remove_last_joint
        except:
            env_args['include_puck'] = True
            env_args['remove_last_joint'] = True
        env_args['hit_coeff'] = variant.hit_coeff
        env_args['scale_obs'] = variant.scale_obs
        env_args['scale_action'] = variant.scale_action
        env_args['alpha_r'] = variant.alpha_r
        env_args['max_path_len'] = variant.max_path_len

    if log_args['alg'] == 'ppo':
        alg_args['learning_rate'] = variant.lr
        try:
            alg_args['policy'] = variant.policy
        except Exception:
            alg_args['policy'] = 'MlpPolicy'
        alg_args['n_steps'] = variant.steps_per_update
        alg_args['batch_size'] = variant.batch_size
        alg_args['n_epochs'] = variant.n_epochs
        alg_args['gamma'] = variant.gamma
        alg_args['gae_lambda'] = variant.gae_lambda
        alg_args['clip_range'] = variant.clip_range
        alg_args['clip_range_vf'] = variant.clip_range_vf
        alg_args['normalize_advantage'] = variant.normalize_advantage
        alg_args['ent_coef'] = variant.ent_coef
        alg_args['vf_coef'] = variant.vf_coef
        alg_args['max_grad_norm'] = variant.max_grad_norm
        alg_args['use_sde'] = variant.use_sde
        alg_args['sde_sample_freq'] = variant.sde_sample_freq
        alg_args['target_kl'] = variant.target_kl
        alg_args['stats_window_size'] = variant.stats_window_size
        alg_args['tensorboard_log'] = variant.tensorboard_log
        alg_args['verbose'] = variant.verbose
        alg_args['seed'] = variant.seed
    elif log_args['alg'] == 'sac':
        alg_args['policy'] = variant.policy
        alg_args['learning_rate'] = variant.learning_rate
        alg_args['buffer_size'] = variant.buffer_size
        alg_args['learning_starts'] = variant.learning_starts
        alg_args['batch_size'] = variant.batch_size
        alg_args['tau'] = variant.tau
        alg_args['gamma'] = variant.gamma
        alg_args['train_freq'] = variant.train_freq
        alg_args['gradient_steps'] = variant.gradient_steps
        alg_args['action_noise'] = variant.action_noise
        alg_args['replay_buffer_class'] = variant.replay_buffer_class
        alg_args['replay_buffer_kwargs'] = variant.replay_buffer_kwargs
        alg_args['optimize_memory_usage'] = variant.optimize_memory_usage
        alg_args['ent_coef'] = variant.ent_coef
        alg_args['target_update_interval'] = variant.target_update_interval
        alg_args['target_entropy'] = variant.target_entropy
        alg_args['use_sde'] = variant.use_sde
        alg_args['sde_sample_freq'] = variant.sde_sample_freq
        alg_args['use_sde_at_warmup'] = variant.use_sde_at_warmup
        alg_args['stats_window_size'] = variant.stats_window_size
        alg_args['tensorboard_log'] = variant.tensorboard_log
        alg_args['policy_kwargs'] = variant.policy_kwargs
        alg_args['verbose'] = variant.verbose
        alg_args['seed'] = variant.seed
        alg_args['device'] = variant.device
        alg_args['_init_setup_model'] = variant._init_setup_model
    elif log_args['alg'] == 'dqn':
        alg_args['policy'] = variant.policy
        alg_args['learning_rate'] = variant.learning_rate
        alg_args['buffer_size'] = variant.buffer_size
        alg_args['learning_starts'] = variant.learning_starts
        alg_args['batch_size'] = variant.batch_size
        alg_args['tau'] = variant.tau
        alg_args['gamma'] = variant.gamma
        alg_args['train_freq'] = variant.train_freq
        alg_args['gradient_steps'] = variant.gradient_steps
        alg_args['replay_buffer_class'] = variant.replay_buffer_class
        alg_args['replay_buffer_kwargs'] = variant.replay_buffer_kwargs
        alg_args['optimize_memory_usage'] = variant.optimize_memory_usage
        alg_args['target_update_interval'] = variant.target_update_interval
        alg_args['exploration_fraction'] = variant.exploration_fraction
        alg_args['exploration_initial_eps'] = variant.exploration_initial_eps
        alg_args['exploration_final_eps'] = variant.exploration_final_eps
        alg_args['max_grad_norm'] = variant.max_grad_norm
        alg_args['stats_window_size'] = variant.stats_window_size
        alg_args['tensorboard_log'] = variant.tensorboard_log
        alg_args['policy_kwargs'] = variant.policy_kwargs
        alg_args['verbose'] = variant.verbose
        alg_args['seed'] = variant.seed
        alg_args['device'] = variant.device
        alg_args['_init_setup_model'] = variant._init_setup_model


    learn_args['total_timesteps'] = variant.total_timesteps
    # learn_args['tb_log_name'] = variant.tb_log_name

    return env_args, alg_args, learn_args, log_args, variant


def load_variant(path):
    with open(os.path.join(path, 'variant.json'), 'r') as fp:
        variant = json.load(fp)

    return Namespace(**variant)
