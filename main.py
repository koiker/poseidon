import argparse
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration for pose estimation project.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file.')
    return parser.parse_args()

def setup_experiment_folders(config):
    # Base directory for all experiments
    base_dir = config['output']['output_dir']
    exp_name = config['output']['exp_name']
    
    # Full path for the current experiment
    exp_path = os.path.join(base_dir, exp_name)
    
    # Subdirectories for models, results, and logs
    folders = []
    if config['output']['save_model']:
        folders.append('models')
    if config['output']['save_results']:
        folders.append('results')
    if config['output']['save_logs']:
        folders.append('logs')
    if config['output']['save_best_model']:
        folders.append('best_model')
    
    # Create the directory structure
    for folder in folders:
        os.makedirs(os.path.join(exp_path, folder), exist_ok=True)
    
    return exp_path  # Return the path to the experiment director

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Setup the experiment folders
    # exp_path = setup_experiment_folders(cfg)


if __name__ == "__main__":
    main()
