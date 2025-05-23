"""
Example of integrating Differential Privacy with the Aggregator.

This module demonstrates how to use the Gaussian Mechanism to add differential
privacy guarantees to the federated learning aggregation process.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from src.dp import GaussianMechanism, compute_sensitivity_l2


class DPAggregator:
    """
    Aggregator with Differential Privacy support.
    
    This class shows how to integrate DP mechanisms with the aggregation pipeline.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        mechanism: str = 'gaussian'
    ):
        """
        Initialize the DP Aggregator.
        
        Args:
            epsilon: Privacy parameter (smaller = more privacy).
            delta: Probability of privacy breach.
            clip_norm: Maximum L2 norm for clipping model updates.
            mechanism: Type of DP mechanism ('gaussian' is currently the only option).
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.mechanism_type = mechanism
        self.dp_mechanism = None
        
    def _clip_updates(
        self, 
        model_updates: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Clip model updates to bound their L2 norm.
        
        Args:
            model_updates: List of model updates from clients.
            
        Returns:
            List of clipped model updates.
        """
        clipped_updates = []
        
        for update in model_updates:
            # Calculate the L2 norm of the entire update
            squared_sum = 0
            for layer_name, layer_update in update.items():
                squared_sum += np.sum(layer_update ** 2)
            
            update_norm = np.sqrt(squared_sum)
            
            # Apply clipping if the norm exceeds the threshold
            if update_norm > self.clip_norm:
                scaling_factor = self.clip_norm / update_norm
                clipped_update = {
                    name: values * scaling_factor 
                    for name, values in update.items()
                }
                clipped_updates.append(clipped_update)
            else:
                # No clipping needed
                clipped_updates.append(update)
        
        return clipped_updates
    
    def aggregate_with_dp(
        self, 
        model_updates: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate model updates with differential privacy.
        
        This method:
        1. Clips updates to bound sensitivity
        2. Aggregates the updates (averaging)
        3. Adds calibrated noise for privacy
        
        Args:
            model_updates: List of model updates from clients.
            
        Returns:
            Aggregated model update with differential privacy.
        """
        num_clients = len(model_updates)
        if num_clients == 0:
            raise ValueError("No model updates provided")
        
        # Step 1: Clip updates to bound sensitivity
        clipped_updates = self._clip_updates(model_updates)
        
        # Step 2: Aggregate updates (simple averaging)
        aggregated_update = {}
        
        # Initialize with the structure of the first update
        for layer_name in clipped_updates[0]:
            # Stack all client updates for this layer
            stacked_updates = np.stack([
                client_update[layer_name] 
                for client_update in clipped_updates
            ])
            
            # Average the updates
            aggregated_update[layer_name] = np.mean(stacked_updates, axis=0)
        
        # Step 3: Add noise for differential privacy
        
        # Compute sensitivity (depends on clipping and number of clients)
        # For average aggregation, sensitivity = clip_norm / num_clients
        sensitivity = self.clip_norm / num_clients
        
        # Create or update the DP mechanism
        if self.dp_mechanism is None:
            self.dp_mechanism = GaussianMechanism(
                epsilon=self.epsilon,
                delta=self.delta,
                sensitivity=sensitivity
            )
        else:
            self.dp_mechanism.update_parameters(sensitivity=sensitivity)
        
        # Apply DP noise to the aggregated update
        private_update = self.dp_mechanism.add_noise_to_weights(aggregated_update)
        
        return private_update
    
    def get_privacy_budget_spent(self, num_rounds: int) -> Dict[str, float]:
        """
        Calculate the cumulative privacy budget spent over multiple rounds.
        
        Args:
            num_rounds: Number of aggregation rounds performed.
            
        Returns:
            Dictionary with privacy parameters (effective epsilon and delta).
        """
        if self.dp_mechanism is None:
            raise ValueError("DP mechanism not initialized yet")
        
        # Simple composition (this is a conservative upper bound)
        # In practice, more sophisticated composition theorems should be used
        effective_epsilon = self.epsilon * np.sqrt(2 * num_rounds * np.log(1/self.delta))
        effective_delta = self.delta * num_rounds
        
        return {
            "rounds": num_rounds,
            "effective_epsilon": effective_epsilon,
            "effective_delta": effective_delta,
            "noise_scale": self.dp_mechanism.sigma
        }
