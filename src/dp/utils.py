"""
Utility functions for differential privacy.

This module provides helper functions and utilities for working with 
differential privacy mechanisms and evaluating privacy guarantees.
"""

import math
from typing import Tuple, Dict, Any, List, Optional, Union

import numpy as np


def compute_sensitivity_l2(
    model_weights: Dict[str, np.ndarray],
    clip_norm: float = 1.0
) -> float:
    """
    Compute L2 sensitivity for model weights when clipping is applied.
    
    In federated learning with clipping, the L2 sensitivity is bounded by 
    2 * clip_norm / num_participants when using averaging.
    
    Args:
        model_weights: Dictionary of model weights.
        clip_norm: Maximum L2 norm for clipping gradients/weights.
        
    Returns:
        float: The L2 sensitivity.
    """
    # Just return the clip_norm as sensitivity
    # This assumes worst-case where a participant's data can change 
    # the aggregated model by at most clip_norm
    return clip_norm


def compute_privacy_spent(
    epsilon: float,
    delta: float,
    num_iterations: int,
    noise_multiplier: float,
    num_participants: int
) -> Tuple[float, float]:
    """
    Compute the privacy spent across training iterations.
    
    Simple implementation of privacy accounting. For more precise accounting,
    consider using techniques like Rényi DP or the moments accountant.
    
    Args:
        epsilon: Target privacy parameter.
        delta: Target probability bound.
        num_iterations: Number of training iterations.
        noise_multiplier: The ratio of noise standard deviation to sensitivity.
        num_participants: Number of participants in each round.
        
    Returns:
        Tuple[float, float]: The effective (epsilon, delta) pair.
    """
    # This is a simplified privacy accounting. 
    # For production use, consider libraries like TensorFlow Privacy or Opacus.
    
    # Simple composition (worst-case)
    effective_epsilon = epsilon * math.sqrt(2 * num_iterations * math.log(1/delta))
    effective_delta = delta * num_iterations
    
    return effective_epsilon, effective_delta


def calibrate_noise_to_privacy(
    target_epsilon: float,
    target_delta: float,
    sensitivity: float,
    num_iterations: int
) -> float:
    """
    Calibrate noise scale to achieve target privacy parameters.
    
    Args:
        target_epsilon: Target privacy parameter.
        target_delta: Target probability bound.
        sensitivity: Sensitivity of the query.
        num_iterations: Number of training iterations.
        
    Returns:
        float: The sigma value to use for Gaussian noise.
    """
    # For advanced implementations, consider the analytical Gaussian mechanism or
    # Renyi DP accounting to get tighter bounds.
    
    # Scale epsilon for composition across iterations
    epsilon_per_iteration = target_epsilon / math.sqrt(2 * num_iterations * math.log(1/target_delta))
    
    # Calculate sigma based on the per-iteration privacy parameters
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / target_delta)) / epsilon_per_iteration
    
    return sigma


def assess_utility_loss(
    original_weights: Dict[str, np.ndarray],
    noisy_weights: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Assess utility loss from adding noise.
    
    Args:
        original_weights: Original model weights.
        noisy_weights: Noisy model weights.
        
    Returns:
        Dict: Metrics about utility loss.
    """
    metrics = {}
    
    # Calculate average absolute error
    total_abs_error = 0
    total_elements = 0
    
    # Calculate relative error and RMSE
    total_squared_error = 0
    
    for name in original_weights:
        orig = original_weights[name]
        noisy = noisy_weights[name]
        
        # Absolute error
        abs_error = np.abs(orig - noisy).mean()
        
        # Squared error
        mse = np.mean((orig - noisy) ** 2)
        
        # Store per-weight metrics
        metrics[f"{name}_abs_error"] = float(abs_error)
        metrics[f"{name}_mse"] = float(mse)
        
        # Accumulate for global metrics
        total_abs_error += np.sum(np.abs(orig - noisy))
        total_squared_error += np.sum((orig - noisy) ** 2)
        total_elements += orig.size
    
    # Global metrics
    metrics["global_mean_absolute_error"] = float(total_abs_error / total_elements)
    metrics["global_rmse"] = float(math.sqrt(total_squared_error / total_elements))
    
    return metrics
