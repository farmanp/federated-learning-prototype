"""
Unit tests for the Paillier homomorphic encryption module.
"""

import unittest
import numpy as np
from src.smc.paillier import (
    PaillierCrypto,
    generate_keys,
    encrypt_vector,
    decrypt_vector,
    aggregate_encrypted_vectors
)

class TestPaillierCrypto(unittest.TestCase):
    """Test cases for the PaillierCrypto class and related functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a small key size for faster testing
        self.crypto = PaillierCrypto(key_size=1024)
        self.public_key, self.private_key = self.crypto.generate_keys()
        
        # Test data - small matrix of weight vectors
        self.test_vectors = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [2.1, 2.2, 2.3, 2.4, 2.5],
        ]
        
        # Expected sum of all vectors
        self.expected_sum = [
            sum(vec[i] for vec in self.test_vectors)
            for i in range(len(self.test_vectors[0]))
        ]
    
    def test_key_generation(self):
        """Test that key generation produces valid keys."""
        self.assertIsNotNone(self.public_key)
        self.assertIsNotNone(self.private_key)
        
        # Create a new instance and generate different keys
        crypto2 = PaillierCrypto(key_size=1024)
        pub2, priv2 = crypto2.generate_keys()
        
        # Verify keys are different
        self.assertNotEqual(self.public_key.n, pub2.n)
        self.assertNotEqual(self.private_key.p, priv2.p)
    
    def test_encrypt_decrypt_value(self):
        """Test encryption and decryption of a single value."""
        value = 3.14159
        encrypted = self.crypto.encrypt_value(value)
        decrypted = self.crypto.decrypt_value(encrypted)
        
        # Due to floating point precision, use almost equal
        self.assertAlmostEqual(value, decrypted, places=10)
    
    def test_encrypt_decrypt_vector(self):
        """Test encryption and decryption of a vector."""
        vector = [1.1, 2.2, 3.3, 4.4, 5.5]
        
        encrypted = self.crypto.encrypt_vector(vector)
        decrypted = self.crypto.decrypt_vector(encrypted)
        
        # Check length matches
        self.assertEqual(len(vector), len(decrypted))
        
        # Check values match within floating point precision
        for original, result in zip(vector, decrypted):
            self.assertAlmostEqual(original, result, places=10)
    
    def test_convenience_functions(self):
        """Test the standalone convenience functions."""
        public_key, private_key = generate_keys(key_size=1024)
        vector = [1.1, 2.2, 3.3]
        
        encrypted = encrypt_vector(vector, public_key)
        decrypted = decrypt_vector(encrypted, private_key)
        
        for original, result in zip(vector, decrypted):
            self.assertAlmostEqual(original, result, places=10)
    
    def test_aggregation(self):
        """Test secure aggregation of encrypted vectors."""
        # Encrypt each test vector
        encrypted_vectors = [
            self.crypto.encrypt_vector(vector)
            for vector in self.test_vectors
        ]
        
        # Aggregate the encrypted vectors
        aggregated = aggregate_encrypted_vectors(encrypted_vectors)
        
        # Decrypt the result
        decrypted_sum = self.crypto.decrypt_vector(aggregated)
        
        # Check the result matches the expected sum
        for expected, actual in zip(self.expected_sum, decrypted_sum):
            self.assertAlmostEqual(expected, actual, places=10)
    
    def test_homomorphic_property(self):
        """Test the homomorphic property explicitly: E(a) + E(b) = E(a+b)."""
        a = 5.5
        b = 7.9
        
        # Method 1: Encrypt individually and add encrypted values
        e_a = self.crypto.encrypt_value(a)
        e_b = self.crypto.encrypt_value(b)
        e_sum = e_a + e_b
        
        # Method 2: Add values and then encrypt
        expected_sum = a + b
        
        # Decrypt the result from Method 1
        decrypted_sum = self.crypto.decrypt_value(e_sum)
        
        # Check both methods yield the same result
        self.assertAlmostEqual(expected_sum, decrypted_sum, places=10)
    
    def test_error_handling_key_missing(self):
        """Test error handling when keys are not set."""
        crypto = PaillierCrypto()  # Fresh instance with no keys
        
        with self.assertRaises(ValueError):
            crypto.encrypt_value(1.0)
        
        with self.assertRaises(ValueError):
            crypto.decrypt_value(self.crypto.encrypt_value(1.0))
    
    def test_error_handling_vector_lengths(self):
        """Test error handling for vectors with different lengths."""
        vector1 = self.crypto.encrypt_vector([1.0, 2.0, 3.0])
        vector2 = self.crypto.encrypt_vector([1.0, 2.0])  # Different length
        
        with self.assertRaises(ValueError):
            aggregate_encrypted_vectors([vector1, vector2])
    
    def test_error_handling_empty_vectors(self):
        """Test error handling for empty vectors."""
        with self.assertRaises(ValueError):
            aggregate_encrypted_vectors([])
    
    def test_encryption_performance(self):
        """Test the performance with a slightly larger vector."""
        # Optional test for larger vectors if needed
        # This is more of a benchmark than a test
        vector = np.random.rand(100).tolist()
        
        encrypted = self.crypto.encrypt_vector(vector)
        decrypted = self.crypto.decrypt_vector(encrypted)
        
        # Just check the first and last elements for simplicity
        self.assertAlmostEqual(vector[0], decrypted[0], places=10)
        self.assertAlmostEqual(vector[-1], decrypted[-1], places=10)

if __name__ == "__main__":
    unittest.main()
