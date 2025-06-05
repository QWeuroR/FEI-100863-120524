import subprocess
import sys
import os
import configparser

def run_training(venv_dir, algorithm, runs, total_timesteps, env_id, script):
    print(f"Running training with Algorithm: {algorithm}, Runs: {runs}, Total Timesteps: {total_timesteps}")
    
  
    env_bin = os.path.join(venv_dir, 'bin', 'python')

    if not os.path.exists(env_bin):
        print(f"Error: Python binary not found in the virtual environment at {venv_dir}")
        sys.exit(1)

    command = [
        env_bin, script, 
        '--algorithm', algorithm,
        '--runs', str(runs),
        '--total_timesteps', str(total_timesteps),
        '--venv_dir', venv_dir,
        '--env_id', env_id

    ]
    #  Pridanie potlačenia Mesa výstupov do prostredia subprocessu
    env = os.environ.copy()
    env["MESA_DEBUG"] = "none"
    env["LIBGL_DEBUG"] = "quiet"

    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during subprocess execution: {e}")
    finally:
        print("Subprocess completed and cleaned up.")

def load_config(config_path='TrainingConfig.conf'):
    config = configparser.ConfigParser()
    config.read(config_path)

    try:
        return {
            "script": config.get('script', 'name'),
            "algorithm": config.get('training', 'algorithm'),
            "runs": config.getint('training', 'runs'),
            "total_timesteps": config.getint('training', 'total_timesteps'),
            "venv_dir": config.get('environment', 'venv_dir'),
            "env_id": config.get('training', 'env_id')
        }
    except configparser.NoOptionError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    config_values = load_config()

    script = config_values['script']
    algorithm = config_values['algorithm']
    runs = config_values['runs']
    total_timesteps = config_values['total_timesteps']
    venv_dir = config_values['venv_dir']
    env_id = config_values['env_id']

    print(f"Loaded Configuration:")
    print(f"Algorithm: {config_values['algorithm']}")
    print(f"Runs: {config_values['runs']}")
    print(f"Total Timesteps: {config_values['total_timesteps']}")
    print(f"Virtual Environment Directory: {config_values['venv_dir']}")
    print(f"Environment ID: {config_values['env_id']}")
    print(f"Script: {config_values['script']}")

    # Check if the virtual environment directory exists
    if not os.path.exists(config_values["venv_dir"]):
        print(f"Error: Virtual environment directory {config_values['venv_dir']} does not exist.")
        sys.exit(1)


    if not os.path.exists(venv_dir):
        print(f"Error: Virtual environment directory {venv_dir} does not exist.")
        sys.exit(1)

    run_training(venv_dir, algorithm, runs, total_timesteps, env_id, script)
