"""
Gaussian Mechanism for Differential Privacy.

This module implements the Gaussian noise mechanism for differential privacy,
which adds calibrated Gaussian noise to query outputs based on privacy parameters.
"""

import math
from typing import Union, Dict, Any, Optional, List, Tuple

import numpy as np


class GaussianMechanism:
    """
    Implements the Gaussian Noise Mechanism for Differential Privacy.
    
    This class provides methods to add Gaussian noise to aggregated model weights
    based on the specified privacy parameters (epsilon, delta) and sensitivity.
    
    Attributes:
        epsilon (float): Privacy parameter that controls the privacy budget.
        delta (float): Probability of privacy loss exceeding epsilon.
        sensitivity (float): Maximum difference in the query output when one record changes.
        sigma (float): Standard deviation of the Gaussian noise.
    """
    
    def __init__(
        self, 
        epsilon: float = 1.0, 
        delta: float = 1e-5, 
        sensitivity: float = 1.0
    ):
        """
        Initialize the Gaussian Mechanism with privacy parameters.
        
        Args:
            epsilon (float, optional): Privacy parameter. Defaults to 1.0.
            delta (float, optional): Probability of privacy loss exceeding epsilon. Defaults to 1e-5.
            sensitivity (float, optional): Maximum change in output when one record changes. Defaults to 1.0.
            
        Raises:
            ValueError: If any parameter has an invalid value.
        """
        self._validate_parameters(epsilon, delta, sensitivity)
        
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # Calculate the sigma (noise scale) based on the privacy parameters
        self.sigma = self._calculate_sigma()
        
    def _validate_parameters(self, epsilon: float, delta: float, sensitivity: float) -> None:
        """
        Validate privacy parameters.
        
        Args:
            epsilon (float): Privacy parameter.
            delta (float): Probability of privacy loss exceeding epsilon.
            sensitivity (float): Maximum change in output when one record changes.
            
        Raises:
            ValueError: If any parameter has an invalid value.
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        
        if delta <= 0 or delta >= 1:
            raise ValueError(f"Delta must be in range (0, 1), got {delta}")
        
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
    
    def _calculate_sigma(self) -> float:
        """
        Calculate the scale (standard deviation) of Gaussian noise based on privacy parameters.
        
        Uses the analytic Gaussian mechanism formula to determine the appropriate noise scale.
        
        Returns:
            float: The standard deviation (sigma) for the Gaussian noise.
        """
        # Implementation based on the standard Gaussian mechanism formula
        # σ ≥ sqrt(2 * ln(1.25/δ)) * Δf / ε
        # where Δf is the sensitivity
        
        # This is a simplified version - a more sophisticated implementation might use
        # the analytic Gaussian mechanism from https://arxiv.org/abs/1805.06530
        return math.sqrt(2 * math.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
    
    def add_noise(self, values: Union[np.ndarray, List[float], float]) -> Union[np.ndarray, List[float], float]:
        """
        Add Gaussian noise to input values based on the privacy parameters.
        
        Args:
            values: Array, list, or single value to add noise to.
            
        Returns:
            Union[np.ndarray, List[float], float]: The noisy values with same shape as input.
            
        Raises:
            TypeError: If values is not a supported type.
        """
        if isinstance(values, (int, float)):
            # Handle scalar case
            return values + np.random.normal(0, self.sigma)
        
        elif isinstance(values, list):
            # Convert list to array, add noise, convert back to list
            noisy_array = np.array(values) + np.random.normal(0, self.sigma, size=len(values))
            return noisy_array.tolist()
        
        elif isinstance(values, np.ndarray):
            # Add noise to the numpy array
            return values + np.random.normal(0, self.sigma, size=values.shape)
        
        else:
            raise TypeError(f"Unsupported type for adding noise: {type(values)}")
    
    def add_noise_to_weights(
        self, 
        model_weights: Dict[str, np.ndarray], 
        weight_sensitivity: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Add calibrated Gaussian noise to model weights.
        
        Args:
            model_weights: Dictionary mapping weight names to their values.
            weight_sensitivity: Optional dictionary of per-weight sensitivities.
                               If None, uses the global sensitivity for all weights.
                               
        Returns:
            Dict[str, np.ndarray]: Model weights with added noise.
        """
        noisy_weights = {}
        
        for name, weights in model_weights.items():
            # Use per-parameter sensitivity if provided, otherwise use global
            sensitivity = weight_sensitivity.get(name, self.sensitivity) if weight_sensitivity else self.sensitivity
            
            # Calculate sigma for this specific weight tensor if needed
            if sensitivity != self.sensitivity:
                local_sigma = self.sigma * (sensitivity / self.sensitivity)
            else:
                local_sigma = self.sigma
                
            # Add appropriately scaled noise
            noise = np.random.normal(0, local_sigma, size=weights.shape)
            noisy_weights[name] = weights + noise
            
        return noisy_weights
    
    def get_privacy_parameters(self) -> Dict[str, float]:
        """
        Get the current privacy parameters.
        
        Returns:
            Dict[str, float]: Dictionary containing epsilon, delta, sensitivity and sigma.
        """
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "sigma": self.sigma
        }
    
    def update_parameters(
        self, 
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        sensitivity: Optional[float] = None
    ) -> None:
        """
        Update privacy parameters and recalculate sigma.
        
        Args:
            epsilon: New epsilon value (optional).
            delta: New delta value (optional).
            sensitivity: New sensitivity value (optional).
            
        Raises:
            ValueError: If any parameter has an invalid value.
        """
        # Only update parameters that are provided
        new_epsilon = epsilon if epsilon is not None else self.epsilon
        new_delta = delta if delta is not None else self.delta
        new_sensitivity = sensitivity if sensitivity is not None else self.sensitivity
        
        # Validate and update
        self._validate_parameters(new_epsilon, new_delta, new_sensitivity)
        
        self.epsilon = new_epsilon
        self.delta = new_delta
        self.sensitivity = new_sensitivity
        
        # Recalculate sigma with the new parameters
        self.sigma = self._calculate_sigma()
