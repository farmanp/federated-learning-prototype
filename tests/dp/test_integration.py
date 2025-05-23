"""
Integration tests for the Differential Privacy mechanism.

This module provides tests that demonstrate how to use the DP mechanism
in the context of federated learning.
"""
import unittest
import numpy as np
from src.dp.gaussian_mechanism import GaussianMechanism
from src.dp.utils import compute_sensitivity_l2, assess_utility_loss


class TestDPIntegration(unittest.TestCase):
    """Integration tests for DP mechanisms in federated learning context."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model with random weights to simulate aggregated model
        self.model_weights = {
            'conv1.weight': np.random.rand(16, 3, 3, 3),
            'conv1.bias': np.random.rand(16),
            'fc1.weight': np.random.rand(128, 256),
            'fc1.bias': np.random.rand(128),
            'fc2.weight': np.random.rand(10, 128),
            'fc2.bias': np.random.rand(10)
        }
        
        # Privacy parameters
        self.epsilon = 1.0
        self.delta = 1e-5
        self.clip_norm = 1.0
        
        # Create the mechanism
        self.sensitivity = compute_sensitivity_l2(self.model_weights, self.clip_norm)
        self.dp_mechanism = GaussianMechanism(
            epsilon=self.epsilon,
            delta=self.delta,
            sensitivity=self.sensitivity
        )
    
    def test_model_weight_privacy(self):
        """Test adding DP noise to model weights."""
        # Get a copy of the original weights
        original_weights = {k: v.copy() for k, v in self.model_weights.items()}
        
        # Apply differential privacy
        noisy_weights = self.dp_mechanism.add_noise_to_weights(self.model_weights)
        
        # Check that privacy was applied (weights changed)
        for name in original_weights:
            self.assertFalse(np.allclose(original_weights[name], noisy_weights[name]))
        
        # Assess utility loss
        metrics = assess_utility_loss(original_weights, noisy_weights)
        
        # Log some metrics
        print(f"Privacy parameters: epsilon={self.epsilon}, delta={self.delta}")
        print(f"Noise scale (sigma): {self.dp_mechanism.sigma}")
        print(f"Global mean absolute error: {metrics['global_mean_absolute_error']}")
        print(f"Global RMSE: {metrics['global_rmse']}")
        
        # Basic sanity checks on error metrics
        self.assertGreater(metrics['global_mean_absolute_error'], 0)
        self.assertGreater(metrics['global_rmse'], 0)
    
    def test_privacy_utility_tradeoff(self):
        """Test privacy-utility tradeoff with different privacy parameters."""
        # List of epsilon values to test (stronger to weaker privacy)
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        errors = []
        
        for eps in epsilons:
            # Create mechanism with current epsilon
            mechanism = GaussianMechanism(
                epsilon=eps,
                delta=self.delta,
                sensitivity=self.sensitivity
            )
            
            # Apply DP noise
            noisy_weights = mechanism.add_noise_to_weights(self.model_weights)
            
            # Assess utility loss
            metrics = assess_utility_loss(self.model_weights, noisy_weights)
            errors.append(metrics['global_rmse'])
            
            print(f"Epsilon: {eps}, Sigma: {mechanism.sigma}, RMSE: {metrics['global_rmse']}")
        
        # Verify privacy-utility tradeoff: higher epsilon should give lower error
        for i in range(1, len(epsilons)):
            # Each subsequent epsilon is larger (less privacy), so error should be smaller
            self.assertLess(errors[i], errors[i-1])
    
    def test_federated_learning_simulation(self):
        """
        Simulate a federated learning round with DP.
        
        This test simulates:
        1. Multiple parties computing local updates
        2. Server aggregating the updates
        3. Applying DP noise to the aggregated model
        4. Assessing the impact on model accuracy
        """
        # Simulate local updates from 5 parties
        num_parties = 5
        local_updates = []
        
        for i in range(num_parties):
            # Each party has slightly different weights (simulating local training)
            party_weights = {
                name: values + np.random.normal(0, 0.1, size=values.shape)
                for name, values in self.model_weights.items()
            }
            local_updates.append(party_weights)
        
        # Server aggregation (simple averaging)
        aggregated_weights = {}
        for name in self.model_weights:
            # Collect all parties' weights for this layer
            all_weights = [party[name] for party in local_updates]
            # Average them
            aggregated_weights[name] = np.mean(all_weights, axis=0)
        
        # Apply DP noise to the aggregated model
        sensitivity = compute_sensitivity_l2(aggregated_weights, self.clip_norm) / num_parties
        dp_mechanism = GaussianMechanism(
            epsilon=self.epsilon,
            delta=self.delta,
            sensitivity=sensitivity  # Sensitivity is lower for aggregated values
        )
        
        noisy_aggregated_weights = dp_mechanism.add_noise_to_weights(aggregated_weights)
        
        # Assess utility loss from adding noise
        metrics = assess_utility_loss(aggregated_weights, noisy_aggregated_weights)
        
        print(f"Federated learning with {num_parties} parties:")
        print(f"Sensitivity: {sensitivity}")
        print(f"Noise scale (sigma): {dp_mechanism.sigma}")
        print(f"Global mean absolute error: {metrics['global_mean_absolute_error']}")
        print(f"Global RMSE: {metrics['global_rmse']}")
        
        # With more parties, noise should be less impactful
        self.assertLess(
            metrics['global_rmse'], 
            self.dp_mechanism.sigma,  # Original mechanism had higher sigma
            msg="With multiple parties, error should be lower than for a single party"
        )


if __name__ == '__main__':
    unittest.main()
