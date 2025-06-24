# train.py
import ale_py
import gymnasium as gym
import torch
import wandb
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_atari_env

from miltronic_elements import PPO_Miltronic, MiltronicActorCriticPolicy, MiltronicCNN, MiltronicLoggingCallback

# --- Configuration ---
CONFIG = {
    "total_timesteps": 5_000_000,
    "n_envs": 8,
    "n_steps": 256,
    "batch_size": 512,
    "project_name": "miltronic-pacman-v5-stable-warmup",
    "env_name": "ALE/MsPacman-v5",
    "seed": 2028,
    
    # --- Miltronic Hyperparameters ---
    "harmonic_epsilon": 0.1,
    "kl_stability_threshold": 0.005,
    "kl_ema_alpha": 0.001,
    "expanded_phi_band": 1.05,         # Increased to be > |k - φ| ≈ 1.02
    "collapse_warmup_limit": 500000,
    "collapse_patience_limit": 20000,
    "collapse_trial_length": 10000,
    
    # --- NEW V5 STABILITY EVAL WARMUP ---
    "stability_eval_warmup": 20, # Ignore first 20 train() calls for stability checks
    
    # --- Standard Hyperparameters ---
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "learning_rate": 2.5e-4,
}

def train_miltronic_agent():
    run_name = f"miltronic_stable_warmup_{CONFIG['stability_eval_warmup']}"
    
    run = wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=run_name,
        sync_tensorboard=True,
    )
    
    vec_env = make_atari_env(CONFIG["env_name"], n_envs=CONFIG["n_envs"], seed=CONFIG["seed"], vec_env_cls=SubprocVecEnv)
    
    policy_kwargs = dict(
        harmonic_epsilon=CONFIG["harmonic_epsilon"],
        expanded_phi_band=CONFIG["expanded_phi_band"]
    )
    
    model = PPO_Miltronic(
        policy=MiltronicActorCriticPolicy,
        env=vec_env,
        kl_stability_threshold=CONFIG["kl_stability_threshold"],
        kl_ema_alpha=CONFIG["kl_ema_alpha"],
        stability_eval_warmup=CONFIG["stability_eval_warmup"], # Pass new param
        verbose=1, n_steps=CONFIG["n_steps"], batch_size=CONFIG["batch_size"],
        n_epochs=4, gamma=CONFIG["gamma"], gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"], ent_coef=CONFIG["ent_coef"],
        learning_rate=CONFIG["learning_rate"], tensorboard_log=f"runs/{run.id}",
        seed=CONFIG["seed"], device='cuda' if torch.cuda.is_available() else 'cpu',
        policy_kwargs=policy_kwargs
    )
    
    callback = MiltronicLoggingCallback(
        warmup_limit=CONFIG["collapse_warmup_limit"],
        patience_limit=CONFIG["collapse_patience_limit"],
        trial_length=CONFIG["collapse_trial_length"]
    )
    
    model.learn(total_timesteps=CONFIG["total_timesteps"], callback=callback, progress_bar=True)
    
    model_path = f"models/{run.name}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    run.finish()
    vec_env.close()

def train_vanilla_ppo():
    run_name = f"baseline_ppo_seed_{CONFIG['seed']}"
    
    run = wandb.init(
        project=CONFIG["project_name"],
        config=CONFIG,
        name=run_name,
        sync_tensorboard=True,
        reinit=True
    )
    
    vec_env = make_atari_env(CONFIG["env_name"], n_envs=CONFIG["n_envs"], seed=CONFIG["seed"], vec_env_cls=SubprocVecEnv)
    
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1, n_steps=CONFIG["n_steps"], batch_size=CONFIG["batch_size"],
        n_epochs=4, gamma=CONFIG["gamma"], gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"], ent_coef=CONFIG["ent_coef"],
        learning_rate=CONFIG["learning_rate"], tensorboard_log=f"runs/{run.id}",
        seed=CONFIG["seed"], device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model.learn(total_timesteps=CONFIG["total_timesteps"], progress_bar=True)
    
    model_path = f"models/{run.name}.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    run.finish()
    vec_env.close()

def train_agent():
    print("Training Miltronic PPO agent...")
    train_miltronic_agent()
    
    print("Training vanilla PPO agent...")
    train_vanilla_ppo()

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    train_agent()