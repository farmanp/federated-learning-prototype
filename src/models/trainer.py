"""
Local model trainer module for federated learning.

This module provides functionality for training logistic regression models locally
on each data party's private data and preparing the model parameters for
secure aggregation in the federated learning process.
"""

from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from loguru import logger

def train_local_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hyperparams: Optional[Dict[str, Any]] = None
) -> Tuple[List[float], float, Dict[str, Any]]:
    """
    Train a logistic regression model on local data and return model weights and metrics.
    
    Args:
        X_train: Training feature matrix of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
        X_test: Test feature matrix of shape (n_test_samples, n_features)
        y_test: Test labels of shape (n_test_samples,)
        hyperparams: Optional dictionary of hyperparameters for the model.
                     If None, default parameters will be used.
    
    Returns:
        Tuple containing:
        - List[float]: Flattened model weights (coefficients)
        - float: Test accuracy
        - Dict[str, Any]: Additional metrics and model information
    
    Raises:
        ValueError: If input data has invalid shape or constant labels
    """
    # Validate input data
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logger.error("Empty training or test data provided")
        raise ValueError("Training and test data cannot be empty")
    
    if X_train.shape[1] != X_test.shape[1]:
        logger.error(f"Feature dimension mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}")
        raise ValueError("Training and test data must have the same number of features")
    
    if len(y_train) != X_train.shape[0] or len(y_test) != X_test.shape[0]:
        logger.error("Sample/label count mismatch")
        raise ValueError("Number of samples and labels must match")
    
    # Check for constant labels (all samples have the same class)
    if len(np.unique(y_train)) == 1:
        logger.warning("Training data has only one class. Model will default to predicting this class.")
        # Return constant model (all weights zero except bias)
        constant_class = y_train[0]
        weights = [0.0] * X_train.shape[1]
        # Evaluate "accuracy" on test set if it also has the same constant label
        accuracy = np.mean(y_test == constant_class)
        
        metrics = {
            "model_type": "constant_predictor",
            "constant_class": int(constant_class),
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0],
            "n_test_samples": X_test.shape[0],
            "test_accuracy": accuracy
        }
        
        logger.info(f"Constant model created with test accuracy: {accuracy:.4f}")
        return weights, accuracy, metrics
    
    # Set default hyperparameters if none provided
    if hyperparams is None:
        hyperparams = {
            'solver': 'liblinear',
            'max_iter': 1000,
            'random_state': 42,
            'C': 1.0
        }
    
    # Log the training process
    logger.info(f"Training logistic regression model on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    logger.debug(f"Hyperparameters: {hyperparams}")
    
    try:
        # Initialize and train the model
        model = LogisticRegression(**hyperparams)
        model.fit(X_train, y_train)
        
        # Extract model weights (coefficients)
        if len(model.classes_) == 2:
            # Binary classification: one coefficient vector
            weights = model.coef_.flatten().tolist()
        else:
            # Multiclass: multiple coefficient vectors (one per class)
            # Average the vectors for simplicity in federated setting
            # Note: This is a simplification; in practice, you might need a more sophisticated approach
            weights = np.mean(model.coef_, axis=0).flatten().tolist()
        
        # Evaluate model on test data
        accuracy = model.score(X_test, y_test)
        
        # Extract additional metrics
        metrics = {
            "model_type": "logistic_regression",
            "n_classes": len(model.classes_),
            "classes": model.classes_.tolist(),
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0],
            "n_test_samples": X_test.shape[0],
            "test_accuracy": accuracy,
            "hyperparams": hyperparams,
            "intercept": model.intercept_.tolist() if hasattr(model, 'intercept_') else None,
            "iterations": model.n_iter_[0] if hasattr(model, 'n_iter_') else None
        }
        
        logger.success(f"Model training completed with test accuracy: {accuracy:.4f}")
        return weights, accuracy, metrics
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
