import configparser
import sys
import os
import subprocess


def load_in_venv(venv_dir, load_script, env_id, model_path, model_file, num_steps):
    env_bin = os.path.join(venv_dir, 'bin', 'python')

    if not os.path.exists(env_bin):
        print(f"Error: Python binary not found in the virtual environment at {venv_dir}")
        sys.exit(1)
    
    command = [
        env_bin, load_script,
        '--env_id', env_id,
        '--model_path', model_path,
        '--model_file', model_file,
        '--num_steps', str(num_steps),
        '--algorithm', algorithm
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during subprocess execution: {e}")
    finally:
        print("Subprocess completed and cleaned up.")
 
 
def load_config(config_path='TrainingConfig.conf'):
    config = configparser.ConfigParser()
    config.read(config_path)

    try:
        return {
            "algorithm": config.get('training', 'algorithm'),
            "venv_dir": config.get('environment', 'venv_dir'),
            "env_id": config.get('training', 'env_id'),
            "model_path": config.get('model', 'model_path'),
            "model_file": config.get('model', 'model_file'),
            "num_steps": config.getint('model', 'num_steps'),
            "load_script": config.get('script', 'load_script')
        }
    except configparser.NoOptionError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)



if __name__ == '__main__':
    
    config_values = load_config()
    algorithm = config_values['algorithm']
    venv_dir = config_values['venv_dir']
    env_id = config_values['env_id']
    model_path = config_values['model_path']
    model_file = config_values['model_file']
    num_steps = config_values['num_steps']
    load_script = config_values['load_script']
     
    
    print(f"Loaded Configuration:")
    print(f"Algorithm: {config_values['algorithm']}")
    print(f"venv_dir: {config_values['venv_dir']}")
    print(f"env_id: {config_values['env_id']}")
    print(f"model_path: {config_values['model_path']}")
    print(f"model_file: {config_values['model_file']}")
    print(f"num_steps: {config_values['num_steps']}")
    print(f"load_script: {config_values['load_script']}")

    load_in_venv(venv_dir,load_script, env_id, model_path, model_file, num_steps)

    