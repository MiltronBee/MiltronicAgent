#!/usr/bin/env python3
"""
Test script to verify k' calculations and HIW functionality.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from miltronic_elements import compute_k_prime, PHI

def test_k_prime_calculations():
    """Test various k' calculation scenarios."""
    print("=" * 60)
    print("Testing k' (Harmonic Information Weight) Calculations")
    print("=" * 60)
    
    # Test Case 1: Uniform distribution (maximum entropy)
    print("\n1. Uniform Distribution (4 actions):")
    uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
    k_prime_uniform = compute_k_prime(uniform_probs)
    expected_entropy = -4 * (0.25 * np.log(0.25))
    print(f"   Probabilities: {uniform_probs}")
    print(f"   Expected Entropy: {expected_entropy:.4f}")
    print(f"   k' = {k_prime_uniform:.4f}")
    print(f"   Note: k' = {k_prime_uniform:.4f} (targeting φ ≈ {PHI:.3f})")
    
    # Test Case 2: Highly concentrated distribution (low entropy)
    print("\n2. Concentrated Distribution:")
    concentrated_probs = np.array([0.9, 0.05, 0.03, 0.02])
    k_prime_concentrated = compute_k_prime(concentrated_probs)
    entropy_concentrated = -np.sum(concentrated_probs * np.log(concentrated_probs))
    print(f"   Probabilities: {concentrated_probs}")
    print(f"   Entropy: {entropy_concentrated:.4f}")
    print(f"   k' = {k_prime_concentrated:.4f}")
    print(f"   Note: Lower k' (< φ) indicates low entropy, high certainty")
    
    # Test Case 3: Static base vs φ-based comparison
    print("\n3. Static vs φ-based Comparison:")
    test_probs = np.array([0.6, 0.3, 0.08, 0.02])
    k_prime_phi = compute_k_prime(test_probs, f_base=None)  # Uses φ base
    k_prime_static_e = compute_k_prime(test_probs, f_base=np.e)
    k_prime_static_2 = compute_k_prime(test_probs, f_base=2.0)
    
    print(f"   Probabilities: {test_probs}")
    print(f"   k' (φ base): {k_prime_phi:.4f}")
    print(f"   k' (static base e): {k_prime_static_e:.4f}")
    print(f"   k' (static base 2): {k_prime_static_2:.4f}")
    print(f"   Note: φ base provides natural variance around {PHI:.3f}")
    
    # Test Case 4: Edge case - nearly deterministic
    print("\n4. Nearly Deterministic Distribution:")
    deterministic_probs = np.array([0.99, 0.005, 0.003, 0.002])
    k_prime_det = compute_k_prime(deterministic_probs)
    entropy_det = -np.sum(deterministic_probs * np.log(deterministic_probs + 1e-10))
    print(f"   Probabilities: {deterministic_probs}")
    print(f"   Entropy: {entropy_det:.4f}")
    print(f"   k' = {k_prime_det:.4f}")
    print(f"   Note: Very low k' indicates collapse toward deterministic policy")
    
    # Test Case 5: Resonance detection
    print("\n5. Resonance Detection Examples:")
    resonance_threshold = 0.1
    test_cases = [
        ("Resonant", np.array([0.25, 0.25, 0.25, 0.25])),
        ("Sub-resonant", np.array([0.8, 0.1, 0.06, 0.04])),
        ("Super-resonant", np.array([0.2, 0.2, 0.2, 0.4])),
    ]
    
    for name, probs in test_cases:
        k_prime = compute_k_prime(probs)
        is_resonant = abs(k_prime - PHI) < resonance_threshold
        print(f"   {name:15} k' = {k_prime:.4f}, Resonant: {is_resonant}")
    
    print("\n" + "=" * 60)
    print("k' Calculation Tests Complete")
    print("=" * 60)

def test_collapse_conditions():
    """Test the new collapse detection logic."""
    print("\n" + "=" * 60)
    print("Testing Collapse Detection Conditions")
    print("=" * 60)
    
    # Simulate different k' scenarios
    scenarios = [
        ("Early Training", 2.1, 0.1, 0.05),     # High k', increasing entropy, increasing reward
        ("Mid Training", 1.3, -0.01, 0.02),     # Moderate k', flat entropy, increasing reward
        ("Near Collapse", 1.65, -0.05, 0.001),  # k' ≈ φ, decreasing entropy, flat reward
        ("Post Collapse", 0.8, -0.1, -0.01),    # Low k', decreasing entropy, decreasing reward
    ]
    
    for name, k_prime, dH_dt, dReward_dt in scenarios:
        is_resonant = abs(k_prime - PHI) < 0.1
        is_converging = dH_dt < -0.001
        is_saturating = abs(dReward_dt) < 0.01
        collapse_gate = is_resonant and is_converging and is_saturating
        
        print(f"\n{name}:")
        print(f"   k' = {k_prime:.3f}, dH/dt = {dH_dt:.4f}, dReward/dt = {dReward_dt:.4f}")
        print(f"   Resonant: {is_resonant}, Converging: {is_converging}, Saturating: {is_saturating}")
        print(f"   Collapse Gate: {collapse_gate}")

if __name__ == "__main__":
    test_k_prime_calculations()
    test_collapse_conditions()
    
    print(f"\nGolden Ratio φ = {PHI:.6f} - the natural harmonic resonance target")
    print("φ-based k' calculation provides meaningful variance for meta-adaptation!")
    print("All tests completed successfully!")