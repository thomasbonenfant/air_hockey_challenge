# AirHockey Tournament
This branch contains a hierarchical environment used to learn a high level policy that selects which lower level policy to play.

The Opponent agent is instantiated in `envs/airhockeydoublewrapper.py` and for now it is the baseline agent.

## Usage
Install all the required dependencies
`pip install -r requirements.txt`

The HierarchicalEnv (which is a Gymnasium environment) is located in the `envs/fixed_options_air_hockey.py` file and the function used to instantiate it
during an experiment is in the `envs/env_maker.py` file. There you can also modify which lower policies the HierarchicalEnv
will use

Additionally to start making experiments you need first to generate and save the `env_info` files used by the lower level policies.
To do that simply run: `python envs/save_env_infos.py`

To test the environment you can use `my_scripts/env_launch.py` where you can choose between using a random high level agent
or load a trained one.

## Running an experiment
Right now the experiments stable_baselines3 for the algorithms and tensorboard for logging.
The available algorithms are: 

- PPO
- SAC 
- DQN

An example script for launching a training with ppo is `hrl_ppo.sh`:
```shell 
python -m my_scripts.experiment \
\
--total_timesteps=1000000 \
--save_model_dir="/tmp/" \
--experiment_label="label" \
\
--steps_per_action=15 \
--include_timer \
--scale_obs \
--alpha_r=0.0 \
--large_reward=1000 \
--fault_penalty=1000 \
--fault_risk_penalty=0 \
--parallel=15 \
\
--eval_freq=2048 \
--n_eval_episodes=10 \
\
ppo \
--lr=3e-4 \
--steps_per_update=256 \
--batch_size=256 \
--n_epochs=20 \
--gamma=0.997 \
--gae_lambda=0.95 \
--clip_range=0.2 \
--ent_coef=0.05 \
--vf_coef=0.5 \
--max_grad_norm=0.5 \
--sde_sample_freq=-1 \
--stats_window_size=100 \
--tensorboard_log="./" \
--verbose=1 \
```
