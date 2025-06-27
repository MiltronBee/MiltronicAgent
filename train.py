# train.py - Meta-Adaptive Gating (MAG) Experimental Harness
import argparse
import ale_py
import gymnasium as gym
import torch
import wandb
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env, make_vec_env

from miltronic_elements import (
    PPO_Miltronic, 
    MiltronicActorCriticPolicy, 
    MiltronicMlpPolicy,
    MiltronicCNN, 
    MiltronicLoggingCallback
)

def get_env_specific_configs(env_id):
    """Returns environment-specific configuration parameters."""
    if "BipedalWalker" in env_id:
        return {
            "policy_class": MiltronicMlpPolicy,
            "n_envs": 16,
            "n_steps": 2048,
            "batch_size": 1024,
            "learning_rate": 3e-4,
            "is_atari": False
        }
    elif "Pacman" in env_id or "ALE/" in env_id:
        return {
            "policy_class": MiltronicActorCriticPolicy,
            "n_envs": 8,
            "n_steps": 256,
            "batch_size": 512,
            "learning_rate": 2.5e-4,
            "is_atari": True
        }
    elif "LunarLander" in env_id or "CartPole" in env_id:
        return {
            "policy_class": MiltronicMlpPolicy,
            "n_envs": 8,
            "n_steps": 2048,
            "batch_size": 512,
            "learning_rate": 3e-4,
            "is_atari": False
        }
    else:
        # Default configuration
        return {
            "policy_class": MiltronicMlpPolicy,
            "n_envs": 8,
            "n_steps": 2048,
            "batch_size": 512,
            "learning_rate": 3e-4,
            "is_atari": False
        }

def train_mag_agent(args):
    """Train a Miltronic agent with Meta-Adaptive Gating."""
    env_config = get_env_specific_configs(args.env)
    
    # Central configuration
    config = {
        "total_timesteps": args.timesteps,
        "project_name": "miltronic-mag-experiments",
        "run_name": f"{args.mode}_{args.env.replace('/', '_')}_seed_{args.seed}",
        "mode": args.mode,
        "env_id": args.env,
        "seed": args.seed,
        **env_config,
        
        # MAG Initial Hyperparameters (RECALIBRATED - less sensitive)
        "initial_patience_limit": 100000,  # Increased significantly
        "initial_kl_stability_threshold": 0.015,  # Increased (more lenient)
        "initial_ent_coef": 0.01,
        
        # Miltronic Hyperparameters (RECALIBRATED)
        "harmonic_epsilon": 0.1,
        "expanded_phi_band": 1.05,
        "collapse_warmup_limit": 200000,  # Reduced warmup for faster engagement
        "collapse_trial_length": 20000,   # Increased trial length
        "stability_eval_warmup": 50,      # Increased warmup
        "kl_ema_alpha": 0.001,
        
        # NEW: Trend Confirmation Parameters
        "trend_confirmation_threshold": 0.01,  # Minimal slope for trend validation
        
        # Standard PPO Hyperparameters
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
    }
    
    run = wandb.init(
        project=config["project_name"], 
        name=config["run_name"], 
        config=config, 
        sync_tensorboard=True
    )
    
    # Create environment
    if config["is_atari"]:
        vec_env = make_atari_env(args.env, n_envs=config["n_envs"], seed=config["seed"], vec_env_cls=SubprocVecEnv)
    else:
        vec_env = make_vec_env(args.env, n_envs=config["n_envs"], seed=config["seed"])
    
    # Setup MAG callback with trend confirmation
    initial_mag_params = {
        "patience_limit": config["initial_patience_limit"],
        "kl_stability_threshold": config["initial_kl_stability_threshold"],
        "ent_coef": config["initial_ent_coef"],
    }
    callback = MiltronicLoggingCallback(
        initial_mag_params=initial_mag_params,
        warmup_limit=config["collapse_warmup_limit"],
        trial_length=config["collapse_trial_length"],
        trend_confirmation_threshold=config["trend_confirmation_threshold"]
    )
    
    # Policy kwargs
    policy_kwargs = dict(
        harmonic_epsilon=config["harmonic_epsilon"],
        expanded_phi_band=config["expanded_phi_band"]
    )
    
    # Create the Miltronic agent
    model = PPO_Miltronic(
        policy=config["policy_class"],
        env=vec_env,
        kl_stability_threshold=config["initial_kl_stability_threshold"],
        kl_ema_alpha=config["kl_ema_alpha"],
        stability_eval_warmup=config["stability_eval_warmup"],
        verbose=1,
        n_steps=config["n_steps"], 
        batch_size=config["batch_size"],
        n_epochs=4, 
        gamma=config["gamma"], 
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"], 
        ent_coef=config["initial_ent_coef"],
        learning_rate=config["learning_rate"], 
        tensorboard_log=f"runs/{run.id}",
        seed=config["seed"], 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        policy_kwargs=policy_kwargs
    )
    
    model.learn(total_timesteps=config["total_timesteps"], callback=callback, progress_bar=True)
    
    # Save model
    model_path = f"models/{run.name}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    run.finish()
    vec_env.close()

def train_baseline_agent(args):
    """Train a baseline PPO agent."""
    env_config = get_env_specific_configs(args.env)
    
    config = {
        "total_timesteps": args.timesteps,
        "project_name": "miltronic-mag-experiments",
        "run_name": f"baseline_{args.env.replace('/', '_')}_seed_{args.seed}",
        "env_id": args.env,
        "seed": args.seed,
        **env_config
    }
    
    run = wandb.init(
        project=config["project_name"], 
        name=config["run_name"], 
        config=config, 
        sync_tensorboard=True
    )
    
    # Create environment
    if config["is_atari"]:
        vec_env = make_atari_env(args.env, n_envs=config["n_envs"], seed=config["seed"], vec_env_cls=SubprocVecEnv)
        policy_name = "CnnPolicy"
    else:
        vec_env = make_vec_env(args.env, n_envs=config["n_envs"], seed=config["seed"])
        policy_name = "MlpPolicy"
    
    model = PPO(
        policy=policy_name,
        env=vec_env,
        verbose=1,
        n_steps=config["n_steps"], 
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        tensorboard_log=f"runs/{run.id}",
        seed=config["seed"], 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model.learn(total_timesteps=config["total_timesteps"], progress_bar=True)
    
    # Save model
    model_path = f"models/{run.name}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    run.finish()
    vec_env.close()

def main():
    parser = argparse.ArgumentParser(description="Miltronic MAG Experimental Harness")
    parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", 
                       help="Environment ID (e.g., LunarLander-v2, BipedalWalker-v3, ALE/MsPacman-v5)")
    parser.add_argument("--mode", type=str, default="both", 
                       choices=["miltronic_mag", "baseline", "both"],
                       help="Training mode: miltronic_mag, baseline, or both")
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                       help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    if args.mode == "both":
        print(f"Starting comparison training on {args.env} for {args.timesteps:,} timesteps each (seed={args.seed})")
        
        # Run MAG training
        print("\n=== Phase 1: Training Miltronic MAG Agent ===")
        mag_args = argparse.Namespace(**vars(args))
        mag_args.mode = "miltronic_mag"
        train_mag_agent(mag_args)
        
        # Run baseline training  
        print("\n=== Phase 2: Training Baseline PPO Agent ===")
        baseline_args = argparse.Namespace(**vars(args))
        baseline_args.mode = "baseline"
        train_baseline_agent(baseline_args)
        
        print("Both training runs completed!")
    else:
        print(f"Starting {args.mode} training on {args.env} for {args.timesteps:,} timesteps (seed={args.seed})")
        
        if args.mode == "miltronic_mag":
            train_mag_agent(args)
        else:
            train_baseline_agent(args)
        
        print("Training completed!")

if __name__ == '__main__':
    main()