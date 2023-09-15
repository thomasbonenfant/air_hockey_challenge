from argparse import ArgumentParser
import random

def parse_args():
    parser = ArgumentParser()

    # log args

    parser.add_argument("--save_model_dir", type=str, default="../models")
    parser.add_argument("--experiment_label", type=str, default="")
    parser.add_argument("--alg", type=str, default="ppo")

    # eval args

    parser.add_argument("--eval_freq", type=int, default=20480)
    parser.add_argument("--n_eval_episodes", type=int, default=80)

    # env arguments
    parser.add_argument("--steps_per_action", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--include_timer", action="store_true")
    parser.add_argument("--include_faults", action="store_true")
    parser.add_argument("--include_joints", action="store_true")
    parser.add_argument("--large_reward", type=float, default=100)
    parser.add_argument("--fault_penalty", type=float, default=33.33)
    parser.add_argument("--fault_risk_penalty", type=float, default=0.1)
    parser.add_argument("--scale_obs", action="store_true")
    parser.add_argument("--alpha_r", type=float, default=1.)
    parser.add_argument("--parallel", type=int, default=1)

    # ppo arguments
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps_per_update", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--clip_range_vf", type=float)
    parser.add_argument("--normalize_advantage", action="store_true", default=True)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--use_sde", action="store_true", default=False)
    parser.add_argument("--sde_sample_freq", type=int, default=-1)
    parser.add_argument("--target_kl", type=float)
    parser.add_argument("--stats_window_size", type=int, default=100)
    parser.add_argument("--tensorboard_log", type=str)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int)

    # learning args

    parser.add_argument("--total_timesteps", type=int, required=True)
    #parser.add_argument("--tb_log_name", type=str)

    variant = parser.parse_args()
    env_args = {}
    alg_args = {}
    learn_args = {}
    log_args = {}

    log_args['save_model_dir'] = variant.save_model_dir
    log_args['experiment_label'] = variant.experiment_label
    log_args['alg'] = variant.alg

    env_args['steps_per_action'] = variant.steps_per_action
    env_args['render'] = variant.render
    env_args['include_timer'] = variant.include_timer
    env_args['include_faults'] = variant.include_faults
    env_args['large_reward'] = variant.large_reward
    env_args['fault_penalty'] = variant.fault_penalty
    env_args['fault_risk_penalty'] = variant.fault_risk_penalty
    env_args['scale_obs'] = variant.scale_obs
    env_args['alpha_r'] = variant.alpha_r
    env_args['include_joints'] = variant.include_joints

    alg_args['learning_rate'] = variant.lr
    alg_args['policy'] = "MlpPolicy"
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

    if not variant.seed:
        variant.seed = random.randint(0, 999999)

    alg_args['seed'] = variant.seed

    learn_args['total_timesteps'] = variant.total_timesteps
    #learn_args['tb_log_name'] = variant.tb_log_name

    return env_args, alg_args, learn_args, log_args, variant