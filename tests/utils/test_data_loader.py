"""
Unit tests for data_loader utility module.
"""
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.utils.data_loader import load_data, preprocess_data

class TestDataLoader(unittest.TestCase):
    """Test cases for the data_loader module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = self.test_dir.name
        
        # Create a test CSV file
        self.csv_path = os.path.join(self.temp_dir_path, "test_data.csv")
        self.n_samples = 100
        self.n_features = 10
        
        # Generate test data with known properties
        np.random.seed(42)  # For reproducibility
        X = np.random.randn(self.n_samples, self.n_features)
        y = np.random.randint(0, 2, size=self.n_samples)  # Binary classification
        
        # Add some NaN values for testing
        X[0, 0] = np.nan
        X[5, 5] = np.nan
        
        # Create a DataFrame and save to CSV
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(self.n_features)])
        df['target'] = y
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        """Tear down test fixtures."""
        self.test_dir.cleanup()

    def test_load_data_from_csv(self):
        """Test loading data from a CSV file."""
        df = load_data(self.csv_path)
        
        # Check data shape
        self.assertEqual(df.shape, (self.n_samples, self.n_features + 1))
        
        # Check column names
        self.assertTrue('target' in df.columns)
        for i in range(self.n_features):
            self.assertTrue(f'feature_{i}' in df.columns)
            
        # Check data types
        self.assertEqual(df.dtypes['target'], np.int64)

    def test_load_data_synthetic(self):
        """Test generating synthetic data."""
        n_samples = 100
        n_features = 20
        n_classes = 3
        
        df = load_data(file_path=None, n_samples=n_samples, n_features=n_features, n_classes=n_classes)
        
        # Check data shape
        self.assertEqual(df.shape, (n_samples, n_features + 1))
        
        # Check column names
        self.assertTrue('target' in df.columns)
        for i in range(n_features):
            self.assertTrue(f'feature_{i}' in df.columns)
            
        # Check class count
        self.assertEqual(len(np.unique(df['target'])), n_classes)

    def test_load_data_file_not_found(self):
        """Test error handling for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data("nonexistent_file.csv")

    def test_preprocess_data(self):
        """Test data preprocessing function."""
        # Generate test data
        df = load_data(file_path=None, n_samples=100, n_features=10, random_state=42)
        
        # Process the data
        X_train, X_test, y_train, y_test = preprocess_data(df, test_size=0.2, random_state=42)
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 80)  # 80% of data
        self.assertEqual(X_test.shape[0], 20)   # 20% of data
        self.assertEqual(X_train.shape[1], 10)  # 10 features
        
        # Check normalization 
        # Standard scaled data should have mean close to 0 and std close to 1
        self.assertTrue(abs(X_train.mean()) < 0.1)
        self.assertTrue(abs(X_train.std() - 1.0) < 0.1)
        
        # Ensure labels are preserved
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)

    def test_preprocess_data_with_nan(self):
        """Test handling of NaN values during preprocessing."""
        # Load the test CSV with NaN values
        df = load_data(self.csv_path)
        
        # Check that there are NaN values in the loaded data
        self.assertTrue(df.isna().any().any())
        
        # Process the data - should handle NaNs
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Ensure no NaNs in processed data
        self.assertFalse(np.isnan(X_train).any())
        self.assertFalse(np.isnan(X_test).any())

    def test_preprocess_data_missing_target(self):
        """Test error handling for missing target column."""
        # Create DataFrame without target column
        df = pd.DataFrame(np.random.randn(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
        
        with self.assertRaises(ValueError):
            preprocess_data(df, target_column='target')

    def test_preprocess_data_empty_dataframe(self):
        """Test error handling for empty DataFrame."""
        # Create empty DataFrame
        df = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            preprocess_data(df)

if __name__ == "__main__":
    unittest.main()
