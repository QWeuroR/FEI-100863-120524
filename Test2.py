import gymnasium as gym
import gymnasium_robotics
import os
from stable_baselines3 import PPO, TD3, SAC, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from gymnasium.wrappers import RecordVideo
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gc
import imageio
import argparse
from Callbacks_sc import TensorboardCallback, EvalCallback

import warnings
# warnings.filterwarnings("ignore", category=UserWarning)



env = 0
#ENV_ID = 'FetchPickAndPlaceDense-v4' #FetchPickAndPlaceDense-v3
DEVICES = "cuda:0"

#TOTAL_TIMESTEPS = int(6e6)
#RUNS = 1
NUMENVS = 8 #32
RUN = 1
TESTSEEDS = [7, 33, 777]
LOG_DIR = "logs/"
LOG_INTERVAL = 15000 #15000

ALGORITHMS = {
    'PPO': PPO,
    'TD3': TD3,
    'SAC': SAC,
    'A2C': A2C
}


if __name__=="__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Train reinforcement learning models.")
    parser.add_argument('--algorithm', type=str, help="Algorithm to use (PPO, TD3, SAC, A2C)")
    parser.add_argument('--runs', type=int, help="Number of training runs", default=1)
    parser.add_argument('--total_timesteps', type=int, help="Total timesteps per training run")
    parser.add_argument('--env_id', type=str, help="Environment ID")
    parser.add_argument('--venv_dir', type=str, help="Virtual environment directory")

    # Parse arguments
    args = parser.parse_args()

    print(f"Starting training with algorithm {args.algorithm} for {args.runs} runs and {args.total_timesteps} timesteps each.")

    ENV_ID = args.env_id
    # Create environment
    env = make_vec_env(ENV_ID, n_envs=NUMENVS, vec_env_cls=SubprocVecEnv)

    # Select the algorithm based on the argument
    if args.algorithm not in ALGORITHMS:
        raise ValueError(f"Invalid algorithm: {args.algorithm}. Available algorithms are {list(ALGORITHMS.keys())}.")

    model_class = ALGORITHMS[args.algorithm]

    for _ in range(args.runs):        
        env = make_vec_env(ENV_ID, n_envs=NUMENVS,vec_env_cls=SubprocVecEnv)
        try:
            model_class = ALGORITHMS[args.algorithm]
            model = model_class("MultiInputPolicy", env, verbose=0, tensorboard_log=LOG_DIR, device=DEVICES)

            evalcallback = EvalCallback(model, ENV_ID, LOG_DIR, TESTSEEDS)
            tensorboardcallback = TensorboardCallback(log_interval=LOG_INTERVAL)
            callbacks = [evalcallback, tensorboardcallback]

            print(f"\nTraining {args.algorithm} on {ENV_ID} for {args.total_timesteps} timesteps...")
            model.learn(total_timesteps=args.total_timesteps, progress_bar=True, log_interval=float('inf'), callback=callbacks)
        finally:
            env.close()
            print("Environment closed.")
        model.save(f"./models/{model.logger.get_dir().split('/')[-1]}")

        del env
        del model
        RUN+=1
