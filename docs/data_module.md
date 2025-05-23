# Data Loader and Preprocessing Module

This document provides detailed information about the data loading and preprocessing module 
in the Federated Learning Prototype system.

## Overview

In federated learning, data remains decentralized across multiple parties. The `data_loader.py` module
provides utilities for each data party to efficiently load and preprocess their local datasets,
ensuring consistent formatting and normalization across the federation.

## Module Location

`src/utils/data_loader.py`

## Features

- **Flexible Data Sources**
  - Load data from local CSV files
  - Generate synthetic classification data when real data is unavailable
  - Handle common data loading errors gracefully

- **Standardized Preprocessing**
  - Normalize features using StandardScaler
  - Impute missing values
  - Split datasets consistently into training (80%) and testing (20%) sets

- **Robust Error Handling**
  - Validate data structures and formats
  - Detailed error messages and logging
  - Graceful handling of edge cases

## API Reference

### `load_data`

```python
def load_data(file_path: Optional[str] = None, n_samples: int = 1000, 
              n_features: int = 20, n_classes: int = 2, 
              random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Loads data from a CSV file or generates synthetic classification data.
    """
```

**Parameters:**
- `file_path`: Path to the CSV file. If `None`, synthetic data is generated.
- `n_samples`: Number of samples for synthetic data (default: 1000).
- `n_features`: Number of features for synthetic data (default: 20).
- `n_classes`: Number of classes for synthetic data (default: 2).
- `random_state`: Random seed for synthetic data generation.

**Returns:**
- A pandas DataFrame containing the features and target variable.

**Example:**
```python
# Load real data
df = load_data("path/to/dataset.csv")

# Generate synthetic data
df = load_data(file_path=None, n_samples=500, n_features=10, n_classes=3)
```

### `preprocess_data`

```python
def preprocess_data(df: pd.DataFrame, target_column: str = 'target', 
                   test_size: float = 0.2, 
                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses the data: separates features and target, normalizes features,
    and splits data into training and testing sets.
    """
```

**Parameters:**
- `df`: Pandas DataFrame containing features and the target column.
- `target_column`: Name of the target variable column (default: 'target').
- `test_size`: Proportion of the dataset to include in the test split (default: 0.2).
- `random_state`: Random seed for data splitting.

**Returns:**
- A tuple `(X_train, X_test, y_train, y_test)` of numpy arrays.

**Example:**
```python
# Load data
df = load_data("path/to/dataset.csv")

# Preprocess with default settings (80/20 split)
X_train, X_test, y_train, y_test = preprocess_data(df)

# Customize preprocessing
X_train, X_test, y_train, y_test = preprocess_data(
    df, 
    target_column='class',  # Custom target column name
    test_size=0.3,          # 70/30 split
    random_state=42         # Fixed random seed
)
```

## Error Handling

The module includes comprehensive error handling for various scenarios:

1. **File Not Found**: When a specified CSV file doesn't exist
2. **Empty Data**: When a CSV file is empty or contains no usable data
3. **Missing Target Column**: When the specified target column is not in the dataset
4. **NaN Values**: Automatically imputes NaN values with column means

## Usage in the Federated Learning Context

In federated learning, each data party will:

1. Load their local dataset using `load_data()`
2. Preprocess the data using `preprocess_data()`
3. Train a local model on `X_train, y_train`
4. Evaluate performance on `X_test, y_test`
5. Share model updates (not data) with the aggregator

Example of a data party workflow:
```python
from src.utils.data_loader import load_data, preprocess_data

# 1. Load local data
local_data = load_data("path/to/local_dataset.csv")

# 2. Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(local_data)

# 3. Train local model (example with sklearn)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Evaluate performance
accuracy = model.score(X_test, y_test)
print(f"Local model accuracy: {accuracy:.4f}")

# 5. Extract model parameters for secure sharing
model_params = model.coef_.flatten().tolist()
```

## Testing

The module includes comprehensive unit tests that verify:

- Data loading from CSV files
- Synthetic data generation
- Preprocessing functionality including normalization
- Error handling for various edge cases
- NaN value imputation

To run the tests:
```bash
cd /path/to/federated-learning-prototype
PYTHONPATH=. pytest -xvs tests/utils/test_data_loader.py
```
