from MagicRL.algorithms.pgpe import PGPE
import numpy as np
from MagicRL.envs import AirHockey, AirHockeyDouble
from MagicRL.policies.switcher_policy import SwitcherPolicy
from MagicRL.data_processors import IdentityDataProcessor

dir = '/tmp/test/pgpe/'

# algorithm
DEBUG = False
NATURAL = False
LR_STRATEGY = "adam"
LEARN_STD = False
ITE = 1
BATCH_SIZE = 1
EPISODES_PER_THETA = 10


class PolicyMaker:
    def __init__(self, env_info):
        self.thetas = None
        self.env_info = env_info

    def set_parameters(self, thetas):
        self.thetas = thetas

    def __call__(self, *args):
        pol = SwitcherPolicy(env_info=self.env_info)
        if self.thetas is not None:
            pol.set_parameters(self.thetas)
        pol.reset()
        return pol


if __name__ == "__main__":
    """Environment"""
    env_wrapped = AirHockeyDouble(opponent_delay=0)
    env_info = env_wrapped.env_info
    #
    env_maker = AirHockey
    env_args = dict(horizon=500,
                    gamma=0.997,
                    render=False)



    """Data Processor"""
    dp = IdentityDataProcessor()

    """Policy"""
    policy_maker = SwitcherPolicy
    policy_args = dict(env_info=env_info)

    hp_mean = np.array([0.3, 4.0, 0.2])
    hp_std = np.array([1e-3, 1e-3, 1e-3])
    hp = np.vstack((hp_mean, hp_std))

    alg_parameters = dict(
        lr=[1e-1],
        initial_rho=hp,
        ite=ITE,
        batch_size=BATCH_SIZE,
        episodes_per_theta=EPISODES_PER_THETA,
        env_maker=env_maker,
        env_args=env_args,
        policy_maker=policy_maker,
        policy_args=policy_args,
        data_processor=dp,
        directory=dir,
        verbose=DEBUG,
        natural=NATURAL,
        checkpoint_freq=100,
        lr_strategy=LR_STRATEGY,
        learn_std=LEARN_STD,
        std_decay=1e-5,
        std_min=1e-4,
        n_jobs_param=1,
        n_jobs_traj=1)
    alg = PGPE(**alg_parameters)

    # Learn phase
    alg.learn()
    alg.save_results()
    print(alg.performance_idx)
