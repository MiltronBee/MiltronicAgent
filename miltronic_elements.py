# miltronic_elements.py
import torch
import torch.nn as nn
import numpy as np
import wandb
from collections import deque
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# k' target value
K_PRIME_TARGET = 1.0

# --- 1. k' Calculation Utility ---
def compute_k_prime(action_probs: np.ndarray, f_base=None) -> float:
    """
    Computes the dynamic, entropy-normalized harmonic weight k'.
    Uses golden ratio as the natural harmonic base to target k' ≈ φ as the resonant state.
    """
    # Epsilon for numerical stability
    epsilon = 1e-10
    action_probs = action_probs + epsilon
    
    # H(P) = -Σ p_i * log(p_i)
    entropy = -np.sum(action_probs * np.log(action_probs))
    
    if f_base is None:
        # Use k' = 1 as the target
        # This provides meaningful variance while targeting 1 as the resonance point
        log_f = np.log(np.e)  # ln(e) = 1, so k' = H(P) / 1 = H(P)
    else:
        log_f = np.log(f_base)
        
    # If entropy is effectively zero, the system has minimal information flow
    if abs(entropy) < epsilon:
        return 0.0
        
    return entropy / log_f

# --- 2. Minimal PPO Agent Wrapper ---
class PPO_Miltronic(PPO):
    """
    A simple wrapper around the SB3 PPO class. All custom logic is deferred to the callback.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Miltronic Agent (k'-centric) Initialized. All logic handled by callback.")

# --- 3. The Central Nervous System Callback ---
class MiltronicLoggingCallback(BaseCallback):
    def __init__(self, reward_shaping_lambda=0.05, k_prime_ema_alpha=0.01, history_len=50, verbose=0):
        super().__init__(verbose)
        self.reward_shaping_lambda = reward_shaping_lambda
        self.k_prime_ema_alpha = k_prime_ema_alpha
        
        # EMA and history trackers for the HIW vector
        self.k_prime_ema = K_PRIME_TARGET  # Start at k' = 1 target
        self.entropy_history = deque(maxlen=history_len)
        self.reward_history = deque(maxlen=history_len)
        
    def _on_rollout_end(self) -> None:
        # --- Step 1: Calculate the HIW Vector ---
        # A. Get a representative policy distribution P
        # We sample the buffer to see how the policy behaved on recent, real observations.
        if self.model.rollout_buffer.size() == 0: 
            return
        
        obs_sample, _ = self.model.rollout_buffer.sample(batch_size=min(256, self.model.rollout_buffer.size()))
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_sample)
            action_probs = dist.distribution.probs.cpu().numpy()
            mean_policy_probs = np.mean(action_probs, axis=0)

        # B. Compute current k' and policy entropy H
        k_prime_current = compute_k_prime(mean_policy_probs)
        current_entropy = -np.sum(mean_policy_probs * np.log(mean_policy_probs + 1e-10))

        # C. Update trackers and calculate trends (dH/dt, dReward/dt)
        self.k_prime_ema = (1 - self.k_prime_ema_alpha) * self.k_prime_ema + self.k_prime_ema_alpha * k_prime_current
        self.entropy_history.append(current_entropy)
        current_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
        self.reward_history.append(current_reward)

        # --- Step 2: Implement k'-driven Logic ---
        # A. Reward Shaping: Harmonic alignment bonus targeting k' = 1
        harmonic_reward = self.reward_shaping_lambda * (1 - abs(self.k_prime_ema - K_PRIME_TARGET))
        
        # B. Constraint Modulation: Dynamically adjust the entropy coefficient around k' = 1
        if self.k_prime_ema > K_PRIME_TARGET + 0.3:  # Too chaotic (above k' + threshold)
            self.model.ent_coef = max(self.model.ent_coef * 0.99, 0.001) # Decrease bonus
        elif self.k_prime_ema < K_PRIME_TARGET - 0.3: # Too rigid (below k' - threshold)
            self.model.ent_coef = min(self.model.ent_coef * 1.01, 0.05) # Increase bonus

        # C. Collapse Detection Gate Evaluation
        dH_dt = np.polyfit(range(len(self.entropy_history)), self.entropy_history, 1)[0] if len(self.entropy_history) > 1 else 0
        dReward_dt = np.polyfit(range(len(self.reward_history)), self.reward_history, 1)[0] if len(self.reward_history) > 1 else 0
        
        is_resonant = abs(self.k_prime_ema - K_PRIME_TARGET) < 0.1  # Condition 1: k' ≈ 1 resonance
        is_converging = dH_dt < -0.001                   # Condition 2: Entropy decreasing
        is_saturating = abs(dReward_dt) < 0.1            # Condition 3: Reward plateaued
        
        collapse_gate_active = is_resonant and is_converging and is_saturating

        # --- Step 3: Log All Metrics to W&B ---
        wandb.log({
            "hiw/k_prime_ema": self.k_prime_ema,
            "hiw/k_prime_current": k_prime_current,
            "hiw/policy_entropy": current_entropy,
            "hiw/harmonic_reward": harmonic_reward,
            "hiw/collapse_gate_active": float(collapse_gate_active),
            "hiw/is_resonant": float(is_resonant),
            "hiw/is_converging": float(is_converging),
            "hiw/is_saturating": float(is_saturating),
            "trends/dH_dt": dH_dt,
            "trends/dReward_dt": dReward_dt,
            "params/dynamic_ent_coef": self.model.ent_coef,
            "global_step": self.num_timesteps
        }, commit=False) # Commit with SB3 logs