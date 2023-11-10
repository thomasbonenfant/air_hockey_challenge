# AirHockey Tournament
This branch contains a hierarchical environment used to learn a high level policy that selects which lower level policy to play.

The Opponent agent is instantiated in `envs/airhockeydoublewrapper.py` and for now it is the baseline agent.

## Usage
Install all the required dependencies
`pip install -r requirements.txt`

To include additional custom environments modify the `envs/env_maker.py` file
The training configuration is located in the conf directory and consists in multiple YAML config files


## Example
Right now the experiments stable_baselines3 for the algorithms and tensorboard for logging.
```shell
python -m my_scripts.experiment2 +environment=hit algorithm=sac
```

### Main Configuration file

Found in `conf/config.yaml`

```yaml 
hydra.output_subdir: null
learn:
  total_timesteps: 5000000
seed: null
log_dir: 'logdir'
log_interval: 10 # log statistics every n episodes
parallel: 20
label: label
defaults:
  - callbacks:
      - custom_eval
      - checkpoint
      - info_log
  - algorithm:
      sac
```
