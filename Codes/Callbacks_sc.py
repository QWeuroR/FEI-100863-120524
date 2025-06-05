
import gymnasium as gym
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
from tqdm import tqdm
import numpy as np
import gc
import sys


from contextlib import contextmanager

# @contextmanager
# def suppress_stderr():
#     with open(os.devnull, 'w') as devnull:
#         old_stderr = sys.stderr
#         sys.stderr = devnull
#         try:
#             yield
#         finally:
#             sys.stderr = old_stderr
class EvalCallback(BaseCallback):
    def __init__(self, alg, env_id, log_dir, seed):
        super(EvalCallback, self).__init__()
        self.alg = alg  
        self.log_dir = log_dir  
        self.seed = seed
        self.env_id = env_id

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self):
        output_format=self.logger.output_formats
        self.tb_formatter = next((x for x in output_format if isinstance(x, TensorBoardOutputFormat)))

    def _on_training_end(self):
        self.eval_agent()

    def eval_agent(self):
        crw = [0] * len(self.seed)

        for i in range (len(self.seed)):
            RWD=0
            env = gym.make(self.env_id, render_mode="rgb_array",max_episode_steps=15) #render_mode="rgb_array"
            _,_= env.reset()
            actrun = self.model.logger.get_dir().split("/")[-1]
            # tqdm.write(f"actrun = {actrun}")
            print("actrun = ",actrun)

            writer = self.tb_formatter.writer
            base_video_folder=self.log_dir+f"/videos/{actrun}_{self.seed[i]}"
            video_folder = base_video_folder

            os.makedirs(video_folder)

            env = gym.wrappers.RecordVideo(env, video_folder=video_folder,disable_logger=True )

            obs,_=env.reset(seed=self.seed[i])
            truncation,terminate=False,False
            actStep=0
            info=0

            while not (terminate or truncation):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminate,truncation, info = env.step(action)
                writer.add_scalar(f"eval_succeess/success_rate ({self.seed[i]})",float(info['is_success']) ,actStep )
                RWD+=reward
                actStep+=1
            env.close()
            actStep=0
            crw[i] = RWD
            # print(f"Seed {self.seed[i]}: {crw[i]}")
            # tqdm.write(f"Seed {self.seed[i]}: {crw[i]}")

            writer.add_scalar(f"eval/seed_reward({self.seed[i]})", crw[i])
            del env

            gc.collect()
            torch.cuda.empty_cache()

class TensorboardCallback(BaseCallback):
    def __init__(self, log_interval):
        super().__init__()
        self.log_interval = log_interval

    def _on_step(self):
        if self.model.num_timesteps == 1 or self.model.num_timesteps % self.log_interval == 0: # pridať logovanie na nule #  if self.model.num_timesteps == 0 or self.model.num_timesteps % self.log_interval == 0:
            self.logger.dump(step=self.model.num_timesteps)

            if isinstance(self.model, OnPolicyAlgorithm):
                self.model._dump_logs(iteration=self.model.num_timesteps)
                #print(f"\nONPOLICY ? :   {isinstance(self.model, OnPolicyAlgorithm)}")
            else:
                self.model._dump_logs()
                #print(f"\nONPOLICY ? :   {isinstance(self.model, OnPolicyAlgorithm)}")

        return True

    def _on_training_start(self):
        self.logger.dump(step=0)
        if isinstance(self.model, OnPolicyAlgorithm):
            self.model._dump_logs(iteration=0)
            print("Logging at step 0 completed. On Policy")
        else:
            self.model._dump_logs()
            print("Logging at step 0 completed. Off Policy")


    def _on_training_end(self):
        if isinstance(self.model, OnPolicyAlgorithm):
            self.model._dump_logs(iteration=self.model.num_timesteps) # for PPO print(f"\nONPOLICY ? :   {isinstance(self.model, OnPolicyAlgorithm)}")
        else:
            self.model._dump_logs()
        pass

