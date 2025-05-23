"""
Unit tests for the Differential Privacy utility functions.
"""
import unittest
import math
import numpy as np
from src.dp.utils import (
    compute_sensitivity_l2,
    compute_privacy_spent,
    calibrate_noise_to_privacy,
    assess_utility_loss
)


class TestDPUtils(unittest.TestCase):
    """Tests for DP utility functions."""
    
    def test_compute_sensitivity_l2(self):
        """Test L2 sensitivity computation."""
        weights = {
            'layer1': np.ones((3, 3)),
            'layer2': np.ones(5)
        }
        
        # Test with default clip_norm
        sensitivity = compute_sensitivity_l2(weights)
        self.assertEqual(sensitivity, 1.0)
        
        # Test with custom clip_norm
        sensitivity = compute_sensitivity_l2(weights, clip_norm=2.5)
        self.assertEqual(sensitivity, 2.5)
    
    def test_compute_privacy_spent(self):
        """Test privacy budget accounting."""
        epsilon = 1.0
        delta = 1e-5
        num_iterations = 10
        noise_multiplier = 1.0
        num_participants = 100
        
        effective_epsilon, effective_delta = compute_privacy_spent(
            epsilon, delta, num_iterations, noise_multiplier, num_participants
        )
        
        # Basic sanity checks
        self.assertGreater(effective_epsilon, epsilon)  # Should increase with composition
        self.assertGreater(effective_delta, delta)  # Should increase with composition
        
        # Test scaling with iterations
        epsilon2, delta2 = compute_privacy_spent(
            epsilon, delta, num_iterations * 2, noise_multiplier, num_participants
        )
        
        # More iterations should result in higher privacy cost
        self.assertGreater(epsilon2, effective_epsilon)
        self.assertGreater(delta2, effective_delta)
    
    def test_calibrate_noise_to_privacy(self):
        """Test noise calibration for privacy parameters."""
        target_epsilon = 1.0
        target_delta = 1e-5
        sensitivity = 2.0
        num_iterations = 10
        
        sigma = calibrate_noise_to_privacy(
            target_epsilon, target_delta, sensitivity, num_iterations
        )
        
        # Basic sanity checks
        self.assertGreater(sigma, 0)
        
        # More privacy (lower epsilon) should require more noise
        sigma2 = calibrate_noise_to_privacy(
            target_epsilon / 2, target_delta, sensitivity, num_iterations
        )
        self.assertGreater(sigma2, sigma)
        
        # Higher sensitivity should require more noise
        sigma3 = calibrate_noise_to_privacy(
            target_epsilon, target_delta, sensitivity * 2, num_iterations
        )
        self.assertGreater(sigma3, sigma)
        
        # More iterations should require more noise per iteration
        sigma4 = calibrate_noise_to_privacy(
            target_epsilon, target_delta, sensitivity, num_iterations * 2
        )
        self.assertGreater(sigma4, sigma)
    
    def test_assess_utility_loss(self):
        """Test utility loss assessment."""
        # Create original and noisy weights with controlled differences
        original_weights = {
            'layer1': np.ones((3, 3)),
            'layer2': np.ones(5)
        }
        
        # Add known noise (1.0 to all weights)
        noisy_weights = {
            'layer1': np.ones((3, 3)) + 1.0,
            'layer2': np.ones(5) + 1.0
        }
        
        metrics = assess_utility_loss(original_weights, noisy_weights)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'layer1_abs_error', 'layer1_mse',
            'layer2_abs_error', 'layer2_mse',
            'global_mean_absolute_error', 'global_rmse'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Since we added 1.0 to all values, absolute error should be 1.0
        self.assertAlmostEqual(metrics['layer1_abs_error'], 1.0)
        self.assertAlmostEqual(metrics['layer2_abs_error'], 1.0)
        self.assertAlmostEqual(metrics['global_mean_absolute_error'], 1.0)
        
        # MSE should be 1.0 as well (1.0^2)
        self.assertAlmostEqual(metrics['layer1_mse'], 1.0)
        self.assertAlmostEqual(metrics['layer2_mse'], 1.0)
        self.assertAlmostEqual(metrics['global_rmse'], 1.0)
        
        # Test with different noise levels
        noisy_weights2 = {
            'layer1': np.ones((3, 3)) + 2.0,
            'layer2': np.ones(5) + 0.5
        }
        
        metrics2 = assess_utility_loss(original_weights, noisy_weights2)
        
        # Layer1 should have higher error than before
        self.assertGreater(metrics2['layer1_abs_error'], metrics['layer1_abs_error'])
        
        # Layer2 should have lower error than before
        self.assertLess(metrics2['layer2_abs_error'], metrics['layer2_abs_error'])


if __name__ == '__main__':
    unittest.main()
