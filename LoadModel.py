import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import TD3, SAC, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import argparse


def load_model(model_path, model_file, env, num_steps, algorithm_name):

    algorithms = {
        "SAC": SAC,
        "TD3": TD3,
        "A2C": A2C,
        "PPO": PPO
    }
    algorithm = algorithms.get(algorithm_name)
    if algorithm is None:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    
    model = algorithm.load(model_path+ "/" +model_file, env=env)
    print(f"Model loaded successfully")
    
    
    vec_env = model.get_env()
    obs = vec_env.reset()

    try:
        for i in range(num_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render("human")
            if i % 100 == 0:
                print(i)
    finally:
        vec_env.close()
        print("Vectorized environment closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--algorithm', type=str)
    args = parser.parse_args()

    env = gym.make(args.env_id, render_mode="human")

    load_model(args.model_path, args.model_file, env, args.num_steps, args.algorithm)
    


             