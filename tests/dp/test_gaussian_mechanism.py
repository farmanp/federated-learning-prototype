"""
Unit tests for the Gaussian Mechanism implementation.
"""
import unittest
import numpy as np
from src.dp.gaussian_mechanism import GaussianMechanism


class TestGaussianMechanism(unittest.TestCase):
    """Tests for the Gaussian noise mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.epsilon = 1.0
        self.delta = 1e-5
        self.sensitivity = 2.0
        self.mechanism = GaussianMechanism(
            epsilon=self.epsilon,
            delta=self.delta,
            sensitivity=self.sensitivity
        )
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        mechanism = GaussianMechanism(epsilon=2.0, delta=1e-6, sensitivity=1.5)
        self.assertEqual(mechanism.epsilon, 2.0)
        self.assertEqual(mechanism.delta, 1e-6)
        self.assertEqual(mechanism.sensitivity, 1.5)
        self.assertGreater(mechanism.sigma, 0)
    
    def test_init_invalid_parameters(self):
        """Test initialization with invalid parameters raises ValueError."""
        # Test invalid epsilon
        with self.assertRaises(ValueError):
            GaussianMechanism(epsilon=-1.0, delta=1e-5, sensitivity=1.0)
        
        # Test invalid delta
        with self.assertRaises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=0, sensitivity=1.0)
        
        with self.assertRaises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=1.5, sensitivity=1.0)
        
        # Test invalid sensitivity
        with self.assertRaises(ValueError):
            GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=-1.0)
    
    def test_calculate_sigma(self):
        """Test sigma calculation based on privacy parameters."""
        # Expected sigma for standard Gaussian mechanism formula
        expected_sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
        self.assertAlmostEqual(self.mechanism.sigma, expected_sigma)
    
    def test_add_noise_scalar(self):
        """Test adding noise to a scalar value."""
        value = 10.0
        # Set random seed for reproducibility
        np.random.seed(42)
        noisy_value = self.mechanism.add_noise(value)
        
        # The noise should change the value
        self.assertNotEqual(value, noisy_value)
        
        # Run multiple times to test statistical properties
        np.random.seed(42)
        values = np.array([10.0] * 1000)
        noisy_values = np.array([self.mechanism.add_noise(10.0) for _ in range(1000)])
        
        # Noise should be zero-centered
        noise = noisy_values - values
        self.assertAlmostEqual(np.mean(noise), 0, delta=0.5)
        
        # Standard deviation should be close to sigma
        self.assertAlmostEqual(np.std(noise), self.mechanism.sigma, delta=0.5)
    
    def test_add_noise_list(self):
        """Test adding noise to a list of values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        np.random.seed(42)
        noisy_values = self.mechanism.add_noise(values)
        
        # Output should be the same length
        self.assertEqual(len(values), len(noisy_values))
        
        # Values should be changed
        for original, noisy in zip(values, noisy_values):
            self.assertNotEqual(original, noisy)
        
        # Check type
        self.assertIsInstance(noisy_values, list)
    
    def test_add_noise_array(self):
        """Test adding noise to a numpy array."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        np.random.seed(42)
        noisy_values = self.mechanism.add_noise(values)
        
        # Output should have the same shape
        self.assertEqual(values.shape, noisy_values.shape)
        
        # Check type
        self.assertIsInstance(noisy_values, np.ndarray)
        
        # Test with a different shape
        values_3d = np.ones((3, 4, 5))
        noisy_values_3d = self.mechanism.add_noise(values_3d)
        self.assertEqual(values_3d.shape, noisy_values_3d.shape)
    
    def test_add_noise_to_weights(self):
        """Test adding noise to model weights."""
        model_weights = {
            'layer1/weights': np.ones((5, 5)),
            'layer1/bias': np.ones(5),
            'layer2/weights': np.ones((5, 2)),
            'layer2/bias': np.ones(2)
        }
        
        np.random.seed(42)
        noisy_weights = self.mechanism.add_noise_to_weights(model_weights)
        
        # Check structure is preserved
        self.assertEqual(set(model_weights.keys()), set(noisy_weights.keys()))
        
        # Check shapes are preserved
        for name in model_weights:
            self.assertEqual(model_weights[name].shape, noisy_weights[name].shape)
        
        # Test with per-weight sensitivity
        weight_sensitivity = {
            'layer1/weights': 3.0,  # Different from global sensitivity
            'layer2/bias': 0.5      # Different from global sensitivity
        }
        
        noisy_weights_with_sensitivity = self.mechanism.add_noise_to_weights(
            model_weights, 
            weight_sensitivity=weight_sensitivity
        )
        
        # The structure and shapes should still be preserved
        self.assertEqual(set(model_weights.keys()), set(noisy_weights_with_sensitivity.keys()))
        for name in model_weights:
            self.assertEqual(model_weights[name].shape, noisy_weights_with_sensitivity[name].shape)
    
    def test_get_privacy_parameters(self):
        """Test retrieving privacy parameters."""
        params = self.mechanism.get_privacy_parameters()
        
        self.assertEqual(params['epsilon'], self.epsilon)
        self.assertEqual(params['delta'], self.delta)
        self.assertEqual(params['sensitivity'], self.sensitivity)
        self.assertEqual(params['sigma'], self.mechanism.sigma)
    
    def test_update_parameters(self):
        """Test updating privacy parameters."""
        # Initial parameters
        initial_sigma = self.mechanism.sigma
        
        # Update epsilon only
        self.mechanism.update_parameters(epsilon=2.0)
        self.assertEqual(self.mechanism.epsilon, 2.0)
        self.assertEqual(self.mechanism.delta, self.delta)  # Unchanged
        self.assertEqual(self.mechanism.sensitivity, self.sensitivity)  # Unchanged
        self.assertNotEqual(self.mechanism.sigma, initial_sigma)  # Should be updated
        
        # Update all parameters
        self.mechanism.update_parameters(epsilon=3.0, delta=1e-6, sensitivity=1.0)
        self.assertEqual(self.mechanism.epsilon, 3.0)
        self.assertEqual(self.mechanism.delta, 1e-6)
        self.assertEqual(self.mechanism.sensitivity, 1.0)


if __name__ == '__main__':
    unittest.main()
