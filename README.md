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
Right now the experiments use the PPO implementation of stable_baselines3 and tensorboard for logging.

An example script for launching a training is `hrl_exp.sh`:
```shell 
python -m my_scripts.experiment \
--seed=666666 \
--save_model_dir="models" \
--experiment_label="label" \
--alg="ppo" \                   #this is just another label for now
--steps_per_action=10 \         # How many step the low level policy will act before the high level agent is called for a new action
--include_timer \               # Include Fault Timer in the Observation 
--include_faults \              # Include Fault Count in the observation
--scale_obs \                   # Scale Obs between -1 and 1
--alpha_r=1 \                   # Constraint Penalty Coefficient
--large_reward=100 \            
--fault_penalty=33.33 \
--fault_risk_penalty=0.1 \      # Constant penalty when the puck is in our table side
--parallel=8 \                  # Number of environments instantiated
--lr=3e-4 \
--steps_per_update=256 \        
--batch_size=64 \
--n_epochs=10 \
--gamma=0.999 \
--gae_lambda=0.95 \
--clip_range=0.2 \
--ent_coef=0.0 \
--vf_coef=0.5 \
--max_grad_norm=0.5 \
--sde_sample_freq=-1 \
--stats_window_size=100 \
--tensorboard_log="./" \
--verbose=1 \
--total_timesteps=1000000 \
--eval_freq=20480 \
--n_eval_episodes=50 \
```
