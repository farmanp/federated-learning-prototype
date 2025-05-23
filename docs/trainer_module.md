# Local Model Trainer Module

This document provides detailed information about the local model trainer module 
in the Federated Learning Prototype system.

## Overview

In federated learning, each data party (client) trains a model locally on their private data 
and contributes only the model parameters to the aggregation process. The `trainer.py` module 
provides functionality for training logistic regression models locally and preparing their 
parameters for secure aggregation.

## Module Location

`src/models/trainer.py`

## Features

- **Model Training**
  - Train logistic regression models using sklearn
  - Extract model weights (coefficients) for secure aggregation
  - Evaluate model performance on local test data

- **Parameter Management**
  - Support for custom hyperparameters
  - Handling of multi-class classification
  - Regularization control

- **Error Handling**
  - Validate input data dimensions
  - Handle edge cases like constant labels
  - Provide meaningful error messages through logging

## API Reference

### `train_local_model`

```python
def train_local_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hyperparams: Optional[Dict[str, Any]] = None
) -> Tuple[List[float], float, Dict[str, Any]]:
    """
    Train a logistic regression model on local data and return model weights and metrics.
    """
```

**Parameters:**
- `X_train`: Training feature matrix of shape (n_samples, n_features)
- `y_train`: Training labels of shape (n_samples,)
- `X_test`: Test feature matrix of shape (n_test_samples, n_features)
- `y_test`: Test labels of shape (n_test_samples,)
- `hyperparams`: Optional dictionary of hyperparameters for the model (defaults provided if None)

**Returns:**
- A tuple containing:
  - `List[float]`: Flattened model weights (coefficients)
  - `float`: Test accuracy
  - `Dict[str, Any]`: Additional metrics and model information

**Default Hyperparameters:**
```python
{
    'solver': 'liblinear',
    'max_iter': 1000,
    'random_state': 42,
    'C': 1.0
}
```

## Usage in Federated Learning

In the federated learning context, the trainer module supports the following workflow at each data party:

1. **Data Preparation**: 
   - Load local data from storage
   - Preprocess data (normalize, handle missing values)
   - Split into training and test sets

2. **Local Training**: 
   - Train logistic regression model on local training data
   - Extract model weights (coefficients)
   - Evaluate model on local test data for performance metrics

3. **Parameter Sharing**: 
   - Encrypt the model weights using the public key
   - Send encrypted weights to the aggregator
   - Keep test metrics for local evaluation

## Example Usage

### Basic Model Training

```python
from src.utils.data_loader import load_data, preprocess_data
from src.models.trainer import train_local_model

# Load and preprocess data
data_df = load_data("path/to/local_data.csv")
X_train, X_test, y_train, y_test = preprocess_data(data_df)

# Train the model with default parameters
weights, accuracy, metrics = train_local_model(X_train, y_train, X_test, y_test)

print(f"Model trained with {len(weights)} coefficients")
print(f"Test accuracy: {accuracy:.4f}")
```

### Custom Hyperparameters

```python
# Custom hyperparameters for the logistic regression model
custom_hyperparams = {
    'solver': 'liblinear',
    'max_iter': 2000,
    'C': 0.5,  # Increase regularization strength
    'class_weight': 'balanced',  # Handle class imbalance
    'random_state': 42
}

# Train with custom hyperparameters
weights, accuracy, metrics = train_local_model(
    X_train, y_train, X_test, y_test,
    hyperparams=custom_hyperparams
)
```

### Integration with Secure Aggregation

```python
from src.smc.paillier import encrypt_vector

# Train model and get weights
weights, accuracy, metrics = train_local_model(X_train, y_train, X_test, y_test)

# Encrypt weights using the public key (received from aggregator)
encrypted_weights = encrypt_vector(weights, public_key)

# Send encrypted weights to aggregator
# ... (communication logic)
```

## Error Handling

The module provides comprehensive error handling for:

1. **Empty Datasets**: Raises ValueError if training or test data is empty
2. **Feature Dimension Mismatch**: Ensures training and test data have the same feature dimensions
3. **Sample/Label Mismatch**: Validates that the number of samples matches the number of labels
4. **Constant Labels**: Handles the case where all training samples belong to the same class

## Performance Considerations

- **Computation**: Logistic regression training is generally fast, even for large datasets
- **Memory**: The model parameters are represented as a flat vector of floats, which is memory-efficient
- **Convergence**: The default `max_iter=1000` ensures convergence for most datasets

## Testing

The module includes comprehensive unit tests in `tests/models/test_trainer.py` that verify:

- Basic model training functionality
- Weight extraction and accuracy computation
- Edge case handling (constant labels, empty datasets)
- Custom hyperparameter support

To run the tests:
```bash
cd /path/to/federated-learning-prototype
PYTHONPATH=. pytest -xvs tests/models/test_trainer.py
```

## Demo

A demonstration script is available at `src/models/demo_trainer.py`, which illustrates a simple federated learning scenario with multiple data parties training local models and securely aggregating their weights.

To run the demo:
```bash
cd /path/to/federated-learning-prototype
python -m src.models.demo_trainer
```
