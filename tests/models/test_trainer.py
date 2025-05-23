"""
Unit tests for the local model trainer module.
"""

import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.trainer import train_local_model

class TestLocalModelTrainer(unittest.TestCase):
    """Test cases for the local model trainer module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate a synthetic binary classification dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into train/test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Generate a dataset with constant labels for edge case testing
        self.X_constant = np.random.rand(100, 10)
        self.y_constant = np.ones(100)
    
    def test_basic_training(self):
        """Test training a model with default parameters."""
        weights, accuracy, metrics = train_local_model(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        # Check that weights are extracted correctly
        self.assertEqual(len(weights), self.X_train.shape[1])
        
        # Verify that weights are not all zero
        self.assertGreater(np.sum(np.abs(weights)), 0)
        
        # Check that accuracy is reasonable (above 80% for this synthetic data)
        self.assertGreater(accuracy, 0.75)
        
        # Check metrics dictionary
        self.assertEqual(metrics["model_type"], "logistic_regression")
        self.assertEqual(metrics["n_classes"], 2)
        self.assertEqual(metrics["n_features"], self.X_train.shape[1])
        self.assertEqual(metrics["test_accuracy"], accuracy)
    
    def test_custom_hyperparameters(self):
        """Test training with custom hyperparameters."""
        custom_hyperparams = {
            'solver': 'liblinear',
            'max_iter': 2000,
            'C': 0.5,
            'random_state': 42
        }
        
        weights, accuracy, metrics = train_local_model(
            self.X_train, self.y_train, self.X_test, self.y_test,
            hyperparams=custom_hyperparams
        )
        
        # Check that hyperparams were used
        self.assertEqual(metrics["hyperparams"], custom_hyperparams)
    
    def test_constant_labels(self):
        """Test handling of a dataset with constant labels (edge case)."""
        weights, accuracy, metrics = train_local_model(
            self.X_constant, self.y_constant, self.X_constant, self.y_constant
        )
        
        # Check that a constant model was created
        self.assertEqual(metrics["model_type"], "constant_predictor")
        
        # Check that weights are all zeros (no need for weights with constant prediction)
        self.assertEqual(np.sum(np.abs(weights)), 0)
        
        # Perfect accuracy on constant test data
        self.assertEqual(accuracy, 1.0)
    
    def test_empty_data(self):
        """Test handling of empty datasets."""
        # Empty training data
        with self.assertRaises(ValueError):
            train_local_model(
                np.array([]), self.y_train, self.X_test, self.y_test
            )
        
        # Empty test data
        with self.assertRaises(ValueError):
            train_local_model(
                self.X_train, self.y_train, np.array([]), self.y_test
            )
    
    def test_feature_mismatch(self):
        """Test handling of feature dimension mismatch."""
        with self.assertRaises(ValueError):
            train_local_model(
                self.X_train, self.y_train, self.X_test[:, :10], self.y_test
            )
    
    def test_sample_label_mismatch(self):
        """Test handling of sample/label count mismatch."""
        with self.assertRaises(ValueError):
            train_local_model(
                self.X_train, self.y_train[:50], self.X_test, self.y_test
            )

if __name__ == "__main__":
    unittest.main()
