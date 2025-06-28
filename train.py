# train.py
import gymnasium as gym
import torch
import wandb
import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from miltronic_elements import PPO_Miltronic, MiltronicLoggingCallback

def main(args):
    # --- Configuration ---
    config = {
        "total_timesteps": 5_000_000,
        "n_envs": 8,
        "project_name": "miltronic-k-prime-release",
        "run_name": f"{args.mode}_{args.env}_seed_{args.seed}",
        "env_name": args.env,
        "mode": args.mode,
        "seed": args.seed,
        
        "policy_type": "CnnPolicy",
        "n_steps": 256,
        "batch_size": 512,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_epochs": 4,
        "clip_range": 0.1,
        "initial_ent_coef": 0.01,
        "learning_rate": 2.5e-4,

        "reward_shaping_lambda": 0.05,
        "k_prime_ema_alpha": 0.01,
    }

    run = wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config,
        sync_tensorboard=True,
    )
    
    vec_env = make_atari_env(config["env_name"], n_envs=config["n_envs"], seed=config["seed"])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model_class = PPO
    callback = None
    if args.mode == "miltronic_k_prime":
        model_class = PPO_Miltronic
        callback = MiltronicLoggingCallback(
            reward_shaping_lambda=config["reward_shaping_lambda"],
            k_prime_ema_alpha=config["k_prime_ema_alpha"]
        )
    
    model = model_class(
        config["policy_type"],
        vec_env,
        verbose=1,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        n_epochs=config["n_epochs"],
        clip_range=config["clip_range"],
        ent_coef=config["initial_ent_coef"], # Set the initial value
        learning_rate=config["learning_rate"],
        tensorboard_log=f"runs/{run.id}",
        seed=config["seed"],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"--- Starting Training: {config['run_name']} ---")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        progress_bar=True
    )
    
    model_path = f"models/{run.name}.zip"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ALE/MsPacman-v5")
    parser.add_argument("--mode", type=str, default="miltronic_k_prime", choices=["miltronic_k_prime", "baseline"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)