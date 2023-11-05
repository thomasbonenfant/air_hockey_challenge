import os
import json


def create_log_directory(cfg, seed):
    env = cfg['environment']['env']
    log_dir = cfg['log_dir']
    alg = cfg['algorithm']['alg']
    label = cfg['label']

    # Generate a unique directory name based on experiment label and seed
    log_dir = os.path.join(log_dir, env, alg, label, str(seed))

    try:
        # Create the directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=False)

    except OSError:
        print('An experiment already exists in this directory')

    return log_dir
