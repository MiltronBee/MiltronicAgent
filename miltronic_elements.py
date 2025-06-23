# miltronic_elements.py
import ale_py
import torch
import torch.nn as nn
import numpy as np
import wandb
import json
import os
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces

from knf import compute_knf, w_lambda_gate_torch, PHI

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

# --- 2. Custom Policy (UPDATED for Dynamic Epsilon) ---
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

# --- 4. Custom Logging Callback (UPDATED with Warmup Gate) ---
class MiltronicLoggingCallback(BaseCallback):
    def __init__(self, warmup_limit=500000, patience_limit=20000, trial_length=10000, verbose=0):
        super().__init__(verbose)
        self.warmup_limit = warmup_limit
        self.patience_limit = patience_limit
        self.trial_length = trial_length
        
        self.stable_step_counter = 0
        self.force_collapse_cooldown = 0
        self.collapse_events_rollout = 0

    def _on_rollout_start(self) -> None:
        self.collapse_events_rollout = 0

    def _on_step(self) -> bool:
        # --- NEW: WARMUP GATE ---
        # Do not engage any patience/forcing logic until the agent has had time to learn.
        if self.num_timesteps < self.warmup_limit:
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
        if policy.is_policy_stable:
            self.stable_step_counter += 1
        else:
            self.stable_step_counter = 0 # Reset if not stable

        # Determine if we should attempt a forced collapse
        should_force = (self.stable_step_counter > self.patience_limit) and (self.force_collapse_cooldown == 0)
        policy.attempt_forced_collapse = should_force
        
        # Check if a collapse actually occurred this step
        current_epsilon = policy.expanded_phi_band if should_force else policy.harmonic_epsilon
        is_harmonic_now = abs(policy.k_value - PHI) <= current_epsilon
        
        if policy.is_policy_stable and is_harmonic_now:
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
        wandb.log({
            "miltronic_rollout/total_collapse_events": self.collapse_events_rollout,
            "global_step": self.num_timesteps
        })
