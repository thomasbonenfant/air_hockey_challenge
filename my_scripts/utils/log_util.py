import os
import json

def create_log_directory(log_args, variant):
    # Generate a unique directory name based on experiment label and seed
    log_dir = os.path.join(log_args['save_model_dir'], f"{log_args['alg']}",f"{log_args['experiment_label']}",f"{variant.seed}")

    try:
        # Create the directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=False)

        # Save the variant dictionary as a JSON file in the log directory
        variant_json_path = os.path.join(log_dir, "variant.json")
        with open(variant_json_path, 'w') as json_file:
            json.dump(vars(variant), json_file, indent=4)
    except OSError:
        print('An experiment already exists in this directory')

    return log_dir