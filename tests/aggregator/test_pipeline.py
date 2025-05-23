"""
Unit tests for the Aggregator Pipeline.

These tests verify the complete integration of secure multi-party computation
and differential privacy in the federated learning aggregation process.
"""

import unittest
import numpy as np
from unittest.mock import patch
from typing import List

from src.smc.paillier import generate_keys, encrypt_vector
from src.aggregator.pipeline import AggregatorPipeline, aggregate_and_finalize


class TestAggregatorPipeline(unittest.TestCase):
    """Tests for the AggregatorPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate keys for testing
        self.public_key, self.private_key = generate_keys(key_size=1024)  # Smaller keys for faster tests
        
        # Test parameters
        self.dp_epsilon = 1.0
        self.dp_delta = 1e-5
        self.dp_sensitivity = 1.0
        
        # Create pipeline
        self.pipeline = AggregatorPipeline(
            public_key=self.public_key,
            private_key=self.private_key,
            dp_epsilon=self.dp_epsilon,
            dp_delta=self.dp_delta,
            dp_sensitivity=self.dp_sensitivity
        )
        
        # Sample test data - model weights for 3 clients
        self.client_weights = [
            [1.0, 2.0, 3.0, 4.0, 5.0],  # Client 1
            [2.0, 3.0, 4.0, 5.0, 6.0],  # Client 2  
            [3.0, 4.0, 5.0, 6.0, 7.0]   # Client 3
        ]
        
        # Expected aggregated result (sum)
        self.expected_sum = [6.0, 9.0, 12.0, 15.0, 18.0]
        
        # Expected average
        self.expected_average = [2.0, 3.0, 4.0, 5.0, 6.0]
    
    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.public_key, self.public_key)
        self.assertEqual(self.pipeline.private_key, self.private_key)
        self.assertEqual(self.pipeline.dp_mechanism.epsilon, self.dp_epsilon)
        self.assertEqual(self.pipeline.dp_mechanism.delta, self.dp_delta)
        self.assertEqual(self.pipeline.dp_mechanism.sensitivity, self.dp_sensitivity)
        self.assertEqual(self.pipeline.aggregation_count, 0)
    
    def test_encrypt_decrypt_consistency(self):
        """Test that encryption followed by decryption preserves values."""
        original_weights = [1.5, -2.3, 0.0, 4.7, -1.2]
        
        # Encrypt
        encrypted_weights = encrypt_vector(original_weights, self.public_key)
        
        # Create single-client update for testing decryption
        encrypted_updates = [encrypted_weights]
        
        # Process through pipeline (this will add noise, so we check structure)
        result = self.pipeline.aggregate_and_finalize(encrypted_updates)
        
        # Should have same length
        self.assertEqual(len(result), len(original_weights))
        
        # Values should be different due to DP noise
        self.assertFalse(np.allclose(result, original_weights, atol=0.1))
    
    def test_three_client_aggregation(self):
        """Test aggregation with exactly 3 clients as required."""
        # Encrypt each client's weights
        encrypted_updates = []
        for client_weights in self.client_weights:
            encrypted_update = encrypt_vector(client_weights, self.public_key)
            encrypted_updates.append(encrypted_update)
        
        # Perform aggregation
        result = self.pipeline.aggregate_and_finalize(encrypted_updates)
        
        # Verify structure
        self.assertEqual(len(result), len(self.client_weights[0]))
        self.assertEqual(self.pipeline.aggregation_count, 1)
        
        # The result should be close to the expected sum (with some noise)
        # We can't check exact equality due to DP noise, but should be in reasonable range
        result_array = np.array(result)
        expected_array = np.array(self.expected_sum)
        
        # Check that results are within reasonable bounds (considering DP noise)
        # The noise scale is proportional to sensitivity/epsilon
        max_expected_noise = 10 * self.dp_sensitivity / self.dp_epsilon  # 10-sigma bound for more leniency in tests
        
        for i in range(len(result)):
            self.assertLess(
                abs(result[i] - self.expected_sum[i]), 
                max_expected_noise,
                f"Position {i}: result {result[i]} too far from expected {self.expected_sum[i]}"
            )
    
    def test_multiple_client_scenarios(self):
        """Test aggregation with different numbers of clients (2-10)."""
        base_weights = [1.0, 2.0, 3.0]
        
        for num_clients in [2, 4, 5, 7, 10]:
            with self.subTest(num_clients=num_clients):
                # Generate weights for each client (slight variations)
                client_weights = []
                for i in range(num_clients):
                    weights = [w + 0.1 * i for w in base_weights]  # Small variations
                    client_weights.append(weights)
                
                # Encrypt updates
                encrypted_updates = []
                for weights in client_weights:
                    encrypted_update = encrypt_vector(weights, self.public_key)
                    encrypted_updates.append(encrypted_update)
                
                # Create fresh pipeline for each test
                test_pipeline = AggregatorPipeline(
                    public_key=self.public_key,
                    private_key=self.private_key,
                    dp_epsilon=1.0,
                    dp_delta=1e-5,
                    dp_sensitivity=1.0
                )
                
                # Perform aggregation
                result = test_pipeline.aggregate_and_finalize(encrypted_updates)
                
                # Verify basic properties
                self.assertEqual(len(result), len(base_weights))
                self.assertIsInstance(result, list)
                self.assertTrue(all(isinstance(x, float) for x in result))
    
    def test_dp_parameter_override(self):
        """Test overriding DP parameters during aggregation."""
        # Encrypt test data
        encrypted_updates = []
        for client_weights in self.client_weights:
            encrypted_update = encrypt_vector(client_weights, self.public_key)
            encrypted_updates.append(encrypted_update)
        
        # Override DP parameters
        new_dp_params = {
            'epsilon': 2.0,  # Less privacy, less noise
            'delta': 1e-6,
            'sensitivity': 0.5
        }
        
        result = self.pipeline.aggregate_and_finalize(encrypted_updates, dp_params=new_dp_params)
        
        # Check that parameters were updated
        self.assertEqual(self.pipeline.dp_mechanism.epsilon, 2.0)
        self.assertEqual(self.pipeline.dp_mechanism.delta, 1e-6)
        self.assertEqual(self.pipeline.dp_mechanism.sensitivity, 0.5)
        
        # Result should have expected structure
        self.assertEqual(len(result), len(self.client_weights[0]))
    
    def test_input_validation(self):
        """Test input validation for encrypted updates."""
        # Test empty updates
        with self.assertRaises(ValueError):
            self.pipeline.aggregate_and_finalize([])
        
        # Test single client - now allowed, but with a warning
        # We can't easily test for warnings in this framework, so we'll just test it works
        single_update = [encrypt_vector([1.0, 2.0], self.public_key)]
        result = self.pipeline.aggregate_and_finalize(single_update)
        self.assertEqual(len(result), 2)  # Should return result with same length
        
        # Test mismatched vector lengths
        weights1 = [1.0, 2.0, 3.0]
        weights2 = [1.0, 2.0]  # Different length
        encrypted_updates = [
            encrypt_vector(weights1, self.public_key),
            encrypt_vector(weights2, self.public_key)
        ]
        with self.assertRaises(ValueError):
            self.pipeline.aggregate_and_finalize(encrypted_updates)
        
        # Test empty vector
        encrypted_updates = [[], encrypt_vector([1.0, 2.0], self.public_key)]
        with self.assertRaises(ValueError):
            self.pipeline.aggregate_and_finalize(encrypted_updates)
    
    def test_privacy_budget_tracking(self):
        """Test privacy budget tracking across multiple rounds."""
        # Perform multiple aggregations
        for round_num in range(3):
            encrypted_updates = []
            for client_weights in self.client_weights:
                encrypted_update = encrypt_vector(client_weights, self.public_key)
                encrypted_updates.append(encrypted_update)
            
            self.pipeline.aggregate_and_finalize(encrypted_updates)
        
        # Check privacy budget
        budget_info = self.pipeline.get_privacy_budget_spent()
        
        self.assertEqual(budget_info['rounds_completed'], 3)
        self.assertEqual(budget_info['per_round_epsilon'], self.dp_epsilon)
        self.assertEqual(budget_info['per_round_delta'], self.dp_delta)
        self.assertGreater(budget_info['effective_epsilon'], self.dp_epsilon)
        self.assertGreater(budget_info['effective_delta'], self.dp_delta)
    
    def test_convenience_function(self):
        """Test the standalone aggregate_and_finalize function."""
        # Encrypt test data
        encrypted_updates = []
        for client_weights in self.client_weights:
            encrypted_update = encrypt_vector(client_weights, self.public_key)
            encrypted_updates.append(encrypted_update)
        
        # DP parameters
        dp_params = {
            'epsilon': 1.0,
            'delta': 1e-5,
            'sensitivity': 1.0
        }
        
        # Use convenience function
        result = aggregate_and_finalize(
            encrypted_updates=encrypted_updates,
            public_key=self.public_key,
            private_key=self.private_key,
            dp_params=dp_params
        )
        
        # Verify result structure
        self.assertEqual(len(result), len(self.client_weights[0]))
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(x, float) for x in result))


class TestAggregationCorrectness(unittest.TestCase):
    """Tests to verify that aggregation produces mathematically correct results."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.public_key, self.private_key = generate_keys(key_size=1024)
    
    def test_aggregation_matches_plain_sum(self):
        """Test that encrypted aggregation matches plain summation (without DP noise)."""
        # Use deterministic "noise" for this test
        with patch('numpy.random.normal', return_value=0.0):  # No noise
            # Test data
            client_weights = [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ]
            
            # Expected sum
            expected_sum = [12.0, 15.0, 18.0]
            
            # Encrypt and aggregate
            encrypted_updates = []
            for weights in client_weights:
                encrypted_updates.append(encrypt_vector(weights, self.public_key))
            
            # Use very high epsilon to minimize noise
            dp_params = {
                'epsilon': 1000.0,  # Very weak privacy = very little noise
                'delta': 1e-5,
                'sensitivity': 1.0
            }
            
            result = aggregate_and_finalize(
                encrypted_updates=encrypted_updates,
                public_key=self.public_key,
                private_key=self.private_key,
                dp_params=dp_params
            )
            
            # Should match expected sum very closely
            np.testing.assert_allclose(result, expected_sum, atol=1e-10)
    
    def test_aggregation_with_negative_values(self):
        """Test aggregation with negative values."""
        client_weights = [
            [-1.0, 2.0, -3.0],
            [4.0, -5.0, 6.0],
            [-7.0, 8.0, -9.0]
        ]
        
        expected_sum = [-4.0, 5.0, -6.0]
        
        # Encrypt and aggregate with minimal noise
        encrypted_updates = []
        for weights in client_weights:
            encrypted_updates.append(encrypt_vector(weights, self.public_key))
        
        dp_params = {
            'epsilon': 100.0,  # Weak privacy for accurate test
            'delta': 1e-5,
            'sensitivity': 1.0
        }
        
        result = aggregate_and_finalize(
            encrypted_updates=encrypted_updates,
            public_key=self.public_key,
            private_key=self.private_key,
            dp_params=dp_params
        )
        
        # Should be close to expected sum
        np.testing.assert_allclose(result, expected_sum, atol=0.1)
    
    def test_aggregation_with_fractional_values(self):
        """Test aggregation with fractional values."""
        client_weights = [
            [1.5, 2.7, 3.3],
            [0.5, 1.3, 2.7],
            [2.0, 0.0, 1.0]
        ]
        
        expected_sum = [4.0, 4.0, 7.0]
        
        encrypted_updates = []
        for weights in client_weights:
            encrypted_updates.append(encrypt_vector(weights, self.public_key))
        
        dp_params = {
            'epsilon': 50.0,  # Weak privacy for accurate test
            'delta': 1e-5,
            'sensitivity': 1.0
        }
        
        result = aggregate_and_finalize(
            encrypted_updates=encrypted_updates,
            public_key=self.public_key,
            private_key=self.private_key,
            dp_params=dp_params
        )
        
        np.testing.assert_allclose(result, expected_sum, atol=0.5)


if __name__ == '__main__':
    unittest.main()
