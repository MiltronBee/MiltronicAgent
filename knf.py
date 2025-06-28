# knf.py
import numpy as np
import torch
from mpmath import log, factorial

# k' target value
K_PRIME_TARGET = 1.0

def compute_knf(n, f):
    """Computes the harmonic informational weight metric k(n, f)."""
    if f <= 1:
        return float('inf')
    return float(log(factorial(n)) / log(f))

def compute_k_prime(action_probs: np.ndarray, f_base=None) -> float:
    """
    Computes the dynamic, entropy-normalized harmonic weight k'.
    
    k'(P, f) = H(P) / ln(f)
    
    Args:
        action_probs: A numpy array of action probabilities from the policy.
        f_base: The harmonic resolution base. If None, uses the dynamic method.
    
    Returns:
        The calculated k' value.
    """
    # Add a small epsilon to prevent log(0)
    action_probs = action_probs + 1e-10
    
    # H(P) = -Σ p_i * log(p_i)
    entropy = -np.sum(action_probs * np.log(action_probs))
    
    if f_base is None:
        # Dynamic base method: Use k' = 1 as the target
        # This creates meaningful variance while targeting 1 as the resonance point
        log_f = np.log(np.e)  # ln(e) = 1, so k' = H(P) / 1 = H(P)
    else:
        log_f = np.log(f_base)
        
    # Handle the edge case where entropy is near zero
    if log_f < 1e-9:
        return 0.0  # A system with zero entropy has no information flow
        
    k_prime = entropy / log_f
    return k_prime

def w_lambda_gate_torch(action_logits):
    """
    Applies the volitional collapse gate.
    Note: The conditional logic is now handled in the policy,
    so this function just performs the collapse.
    """
    action_probs = torch.softmax(action_logits, dim=-1)
    num_actions = action_probs.shape[-1]
    num_to_mask = num_actions // 2
    
    _, mask_indices = torch.topk(action_probs, num_to_mask, largest=False, dim=-1)
    
    mask = torch.full_like(action_logits, 0)
    mask.scatter_(-1, mask_indices, -float('inf'))
    
    gated_logits = action_logits + mask
    return gated_logits

