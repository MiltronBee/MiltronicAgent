# miltronic_elements.py
import ale_py
import torch
import torch.nn as nn
import numpy as np
import wandb
import json
import os
import gymnasium as gym
from collections import deque
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces

from knf import compute_knf, w_lambda_gate_torch, PHI

# --- NEW: Meta-Adaptive Gating (MAG) Component ---
class ConstraintModulator:
    """
    Manages the dynamic adjustment of Miltronic hyperparameters based on performance.
    This is the core of the Meta-Adaptive Gating system.
    """
    def __init__(self, initial_params: dict, trend_threshold=0.01):
        self.trend_threshold = trend_threshold
        
        # Define the parameter tiers based on performance trend
        self.regimes = {
            "PERFORMANCE": { # Reward is trending up
                "patience_limit": initial_params["patience_limit"] * 2,
                "kl_stability_threshold": initial_params["kl_stability_threshold"] * 0.5,
                "ent_coef": initial_params["ent_coef"] * 0.5,
                "regime_id": 1,
                "regime_name": "PERFORMANCE"
            },
            "STAGNATION": { # Reward is flat
                "patience_limit": initial_params["patience_limit"],
                "kl_stability_threshold": initial_params["kl_stability_threshold"],
                "ent_coef": initial_params["ent_coef"],
                "regime_id": 0,
                "regime_name": "STAGNATION"
            },
            "RECOVERY": { # Reward is trending down
                "patience_limit": initial_params["patience_limit"] // 4,
                "kl_stability_threshold": initial_params["kl_stability_threshold"] * 2.0,
                "ent_coef": initial_params["ent_coef"] * 2.0,
                "regime_id": -1,
                "regime_name": "RECOVERY"
            }
        }
        self.current_params = self.regimes["STAGNATION"]

    def update(self, reward_ema_trend: float):
        """Update the current parameter regime based on the reward trend."""
        if reward_ema_trend > self.trend_threshold:
            self.current_params = self.regimes["PERFORMANCE"]
        elif reward_ema_trend < -self.trend_threshold:
            self.current_params = self.regimes["RECOVERY"]
        else:
            self.current_params = self.regimes["STAGNATION"]
            
    def get_params(self) -> dict:
        """Return the current set of dynamic hyperparameters."""
        return self.current_params

# --- 1. Custom CNN Encoder ---
class MiltronicCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# --- 2. Custom CNN Policy (UPDATED for Dynamic Epsilon) ---
class MiltronicActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, harmonic_epsilon=0.1, expanded_phi_band=0.3, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=MiltronicCNN,
                         features_extractor_kwargs=dict(features_dim=128))
        self.k_value = 0.0
        self.harmonic_epsilon = harmonic_epsilon
        self.expanded_phi_band = expanded_phi_band
        self.is_policy_stable = False
        # Flag to be set by the callback
        self.attempt_forced_collapse = False

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, latent_sde=None):
        mean_actions = self.action_net(latent_pi)
        
        # Use expanded band if a forced collapse is being attempted
        current_epsilon = self.expanded_phi_band if self.attempt_forced_collapse else self.harmonic_epsilon
        
        is_harmonic = abs(self.k_value - PHI) <= current_epsilon
        should_collapse = self.training and is_harmonic and self.is_policy_stable
        
        if should_collapse:
            gated_logits = w_lambda_gate_torch(mean_actions)
            return self.action_dist.proba_distribution(action_logits=gated_logits)
        
        return self.action_dist.proba_distribution(action_logits=mean_actions)

# --- NEW: MLP Policy for Non-Atari Environments ---
class MiltronicMlpPolicy(ActorCriticPolicy):
    """
    MLP-based Miltronic policy for continuous control and non-Atari discrete environments.
    Supports the same harmonic collapse mechanics as the CNN version.
    """
    def __init__(self, *args, harmonic_epsilon=0.1, expanded_phi_band=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_value = 0.0
        self.harmonic_epsilon = harmonic_epsilon
        self.expanded_phi_band = expanded_phi_band
        self.is_policy_stable = False
        self.attempt_forced_collapse = False

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, latent_sde=None):
        mean_actions = self.action_net(latent_pi)
        
        # Use expanded band if a forced collapse is being attempted
        current_epsilon = self.expanded_phi_band if self.attempt_forced_collapse else self.harmonic_epsilon
        
        is_harmonic = abs(self.k_value - PHI) <= current_epsilon
        should_collapse = self.training and is_harmonic and self.is_policy_stable
        
        if should_collapse and len(mean_actions.shape) > 1 and mean_actions.shape[-1] > 2:
            # Only apply gating for discrete action spaces with more than 2 actions
            if hasattr(self.action_space, 'n'):  # Discrete action space
                gated_logits = w_lambda_gate_torch(mean_actions)
                return self.action_dist.proba_distribution(action_logits=gated_logits)
        
        return self.action_dist.proba_distribution(action_logits=mean_actions)

# --- 3. Modified PPO Agent (Tracks Policy Stability) ---
class PPO_Miltronic(PPO):
    def __init__(self, policy, env, kl_stability_threshold=0.01, kl_ema_alpha=0.001, stability_eval_warmup=10, **kwargs):
        n_actions = env.action_space.n
        f_dim = 128
        self.k = compute_knf(n_actions, f_dim)
        
        # New parameters for stability evaluation warmup
        self.stability_eval_warmup = stability_eval_warmup
        self.kl_update_counter = 0

        super().__init__(policy, env, **kwargs)
        
        self.kl_stability_threshold = kl_stability_threshold
        self.kl_ema_alpha = kl_ema_alpha
        self.kl_ema = None
        
        print(f"Miltronic Agent Initialized. k={self.k:.4f}, φ≈{PHI:.3f}, KL_thresh<{kl_stability_threshold}, StabilityWarmup={self.stability_eval_warmup} updates")

    def _setup_model(self) -> None:
        super()._setup_model()
        # Pass k-value to policy after initialization
        self.policy.k_value = self.k

    def train(self) -> None:
        super().train()
        
        self.kl_update_counter += 1
        approx_kl = self.logger.name_to_value.get('train/approx_kl')
        
        if approx_kl is not None and not np.isnan(approx_kl):
            if self.kl_ema is None: self.kl_ema = approx_kl
            else: self.kl_ema = ((1 - self.kl_ema_alpha) * self.kl_ema) + (self.kl_ema_alpha * approx_kl)
            
            # --- THIS IS THE FIX ---
            # Gate the stability check until after a warmup period of train() calls
            if self.kl_update_counter < self.stability_eval_warmup:
                is_stable = False
            else:
                is_stable = self.kl_ema < self.kl_stability_threshold
            # ---------------------
            
            self.policy.is_policy_stable = is_stable
            
            self.logger.record("miltronic/kl_ema", self.kl_ema)
            self.logger.record("miltronic/kl_update_counter", self.kl_update_counter)
            self.logger.record("miltronic/is_policy_stable", float(is_stable))

# --- 4. Custom Logging Callback (UPDATED with MAG Integration) ---
class MiltronicLoggingCallback(BaseCallback):
    def __init__(self, initial_mag_params: dict = None, warmup_limit=500000, trial_length=10000, 
                 ema_alpha=0.05, verbose=0):
        super().__init__(verbose)
        self.warmup_limit = warmup_limit
        self.trial_length = trial_length
        
        # MAG Integration
        if initial_mag_params:
            self.modulator = ConstraintModulator(initial_mag_params)
            self.patience_limit = initial_mag_params["patience_limit"]
        else:
            # Fallback to original behavior
            self.modulator = None
            self.patience_limit = 20000
        
        # Reward EMA tracking for MAG
        self.ema_alpha = ema_alpha
        self.reward_ema = None
        self.reward_ema_slow = None  # For trend calculation
        
        # Original state variables
        self.stable_step_counter = 0
        self.force_collapse_cooldown = 0
        self.collapse_events_rollout = 0

    def _on_rollout_start(self) -> None:
        self.collapse_events_rollout = 0

    def _on_step(self) -> bool:
        # --- NEW: WARMUP GATE ---
        # Do not engage any patience/forcing logic until the agent has had time to learn.
        if self.num_timesteps < self.warmup_limit:
            if hasattr(self.model.policy, 'attempt_forced_collapse'):
                self.model.policy.attempt_forced_collapse = False
            # Log that we are in the warmup period
            self.logger.record("miltronic_step/is_in_warmup", 1)
            return True
        # ------------------------

        self.logger.record("miltronic_step/is_in_warmup", 0)
        policy = self.model.policy

        # Decrement cooldown timer
        if self.force_collapse_cooldown > 0:
            self.force_collapse_cooldown -= 1

        # Track continuous stability
        if hasattr(policy, 'is_policy_stable') and policy.is_policy_stable:
            self.stable_step_counter += 1
        else:
            self.stable_step_counter = 0 # Reset if not stable

        # Determine if we should attempt a forced collapse
        should_force = (self.stable_step_counter > self.patience_limit) and (self.force_collapse_cooldown == 0)
        if hasattr(policy, 'attempt_forced_collapse'):
            policy.attempt_forced_collapse = should_force
        
        # Check if a collapse actually occurred this step (for Atari environments)
        if hasattr(policy, 'k_value') and hasattr(policy, 'expanded_phi_band') and hasattr(policy, 'harmonic_epsilon'):
            current_epsilon = policy.expanded_phi_band if should_force else policy.harmonic_epsilon
            is_harmonic_now = abs(policy.k_value - PHI) <= current_epsilon
            
            if hasattr(policy, 'is_policy_stable') and policy.is_policy_stable and is_harmonic_now:
                self.collapse_events_rollout += 1
                # If this was a forced collapse, reset counters and start cooldown
                if should_force:
                    self.stable_step_counter = 0
                    self.force_collapse_cooldown = self.trial_length
                    self.logger.record("miltronic_event/forced_collapse_triggered", 1)

        self.logger.record("miltronic_step/stable_step_counter", self.stable_step_counter)
        self.logger.record("miltronic_step/is_attempting_force_collapse", float(should_force))
        return True

    def _on_rollout_end(self) -> None:
        # MAG Update Logic
        if self.modulator:
            # 1. Update Reward EMA (R̄_t) Monitor
            current_reward = 0
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                # Extract mean reward from episode info buffer
                rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer if 'r' in ep_info]
                if rewards:
                    current_reward = np.mean(rewards)
            elif hasattr(self.locals, 'infos') and self.locals.get('infos'):
                # Fallback: try to get from infos
                rewards = [info.get('episode', {}).get('r', 0) for info in self.locals['infos'] 
                          if info.get('episode', {}).get('r') is not None]
                if rewards:
                    current_reward = np.mean(rewards)
            else:
                current_reward = self.reward_ema or 0
            
            if self.reward_ema is None:
                self.reward_ema = current_reward
                self.reward_ema_slow = current_reward
            else:
                self.reward_ema = (1 - self.ema_alpha) * self.reward_ema + self.ema_alpha * current_reward
                # Use a much slower alpha for the trend anchor
                self.reward_ema_slow = (1 - 0.005) * self.reward_ema_slow + 0.005 * current_reward
                
            reward_trend = self.reward_ema - self.reward_ema_slow

            # 2. Update the Constraint Modulator
            self.modulator.update(reward_trend)
            
            # 3. Inject New Parameters into the Agent and Callback
            new_params = self.modulator.get_params()
            self.patience_limit = new_params["patience_limit"]
            
            # Update model parameters if they exist
            if hasattr(self.model, 'ent_coef'):
                self.model.ent_coef = new_params["ent_coef"]
            if hasattr(self.model, 'kl_stability_threshold'):
                self.model.kl_stability_threshold = new_params["kl_stability_threshold"]
            
            # 4. Log everything to W&B for analysis
            wandb.log({
                "mag/reward_ema": self.reward_ema,
                "mag/reward_trend": reward_trend,
                "mag/current_regime": new_params["regime_id"],
                "mag/regime_name": new_params["regime_name"],
                "mag/dynamic_patience_limit": self.patience_limit,
                "mag/dynamic_ent_coef": new_params["ent_coef"],
                "mag/dynamic_kl_threshold": new_params["kl_stability_threshold"],
                "global_step": self.num_timesteps
            })
        
        # Original logging
        wandb.log({
            "miltronic_rollout/total_collapse_events": self.collapse_events_rollout,
            "global_step": self.num_timesteps
        })
