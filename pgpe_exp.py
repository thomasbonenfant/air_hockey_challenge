# Libraries
import argparse
from MagicRL.algorithms import *
from MagicRL.data_processors import *
from MagicRL.envs import *
from MagicRL.policies import *
from art import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--server",
    help="Location.",
    type=int,
    default=1
)
parser.add_argument(
    "--ite",
    help="How many iterations the algorithm must do.",
    type=int,
    default=100
)
parser.add_argument(
    "--alg",
    help="The algorithm to use.",
    type=str,
    default="pgpe",
    choices=["pgpe", "pg"]
)
parser.add_argument(
    "--var",
    help="The exploration amount.",
    type=float,
    default=1
)
parser.add_argument(
    "--pol",
    help="The policy used.",
    type=str,
    default="switcher",
    choices=["linear", "nn", "switcher"]
)
parser.add_argument(
    "--env",
    help="The environment.",
    type=str,
    default="airhockey",
    choices=["swimmer", "half_cheetah", "reacher", "airhockey"]
)
parser.add_argument(
    "--horizon",
    help="The horizon amount.",
    type=int,
    default=500
)
parser.add_argument(
    "--gamma",
    help="The gamma amount.",
    type=float,
    default=0.999
)
parser.add_argument(
    "--lr",
    help="The lr amount.",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--lr_strategy",
    help="The strategy employed for the lr.",
    type=str,
    default="adam",
    choices=["adam", "constant"]
)
parser.add_argument(
    "--n_workers",
    help="How many parallel cores.",
    type=int,
    default=1
)
parser.add_argument(
    "--batch",
    help="The batch size.",
    type=int,
    default=100
)
parser.add_argument(
    "--clip",
    help="Whether to clip the action in the environment.",
    type=int,
    default=0
)

args = parser.parse_args()

# Preprocess Arguments
np.random.seed()

if args.alg == "pg":
    if args.pol == "linear":
        args.pol = "gaussian"
    elif args.pol == "nn":
        args.pol = "deep_gaussian"

if args.var < 1:
    string_var = str(args.var).replace(".", "")
else:
    string_var = str(int(args.var))

# Build
if args.server:
    dir_name = f"/data/air_hockey/thomas/data/{args.alg}/"
else:
    dir_name = f"/tmp/pgpe/{args.alg}/"
dir_name += f"{args.alg}_{args.ite}_{args.env}_{args.horizon}_{args.lr_strategy}_"
dir_name += f"{str(args.lr).replace('.', '')}_{args.pol}_batch_{args.batch}_"

if args.clip:
    dir_name += "clip_"
else:
    dir_name += "noclip_"

"""Environment"""
MULTI_LINEAR = False
if args.env == "airhockey":
    wrapped_env = AirHockeyDouble(opponent_delay=0)
    env_info = wrapped_env.env_info
    env_maker = AirHockey
    env_args = {
        "horizon": 500,
        "gamma": 0.997,
        "render": False
    }

else:
    raise ValueError(f"Invalid env name.")

"""Data Processor"""
dp = IdentityDataProcessor()

"""Policy"""
if args.pol == "switcher":
    policy_maker = SwitcherPolicy
    policy_args = {"env_info": env_info}
    tot_params = 3
else:
    raise ValueError(f"Invalid policy name.")
dir_name += f"{tot_params}_var_{string_var}"

"""Algorithm"""
if args.alg == "pgpe":
    if args.var == 1:
        var_term = 1.001
    else:
        var_term = args.var
    hp = np.zeros((2, tot_params))
    if args.pol == "linear":
        hp[0] = [0] * tot_params
    else:
        hp[0] = np.random.normal(0, 1, tot_params)
    hp[1] = [np.log(np.sqrt(var_term))] * tot_params
    alg_parameters = dict(
        lr=[args.lr],
        initial_rho=hp,
        ite=args.ite,
        batch_size=args.batch,
        episodes_per_theta=1,
        env_maker=env_maker,
        env_args=env_args,
        policy_maker=policy_maker,
        policy_args=policy_args,
        data_processor=dp,
        directory=dir_name,
        verbose=False,
        natural=False,
        checkpoint_freq=100,
        lr_strategy=args.lr_strategy,
        learn_std=False,
        std_decay=0,
        std_min=1e-6,
        n_jobs_param=args.n_workers,
        n_jobs_traj=1
    )
    alg = PGPE(**alg_parameters)
else:
    raise ValueError("Invalid algorithm name.")

if __name__ == "__main__":
    print(text2art(f"== {args.alg} on {args.env} =="))
    print(args)
    alg.learn()
    alg.save_results()
    print(alg.performance_idx)
