"""
Utility module for loading and preprocessing datasets.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from loguru import logger

def load_data(file_path: Optional[str] = None, n_samples: int = 1000, n_features: int = 20, n_classes: int = 2, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Loads data from a CSV file or generates synthetic classification data.

    Args:
        file_path: Path to the CSV file. If None, synthetic data is generated.
        n_samples: Number of samples for synthetic data.
        n_features: Number of features for synthetic data.
        n_classes: Number of classes for synthetic data.
        random_state: Random seed for synthetic data generation.

    Returns:
        A pandas DataFrame containing the features and target.
    """
    if file_path:
        try:
            logger.info(f"Loading data from CSV: {file_path}")
            df = pd.read_csv(file_path)
            # Assuming the last column is the target variable
            if 'target' not in df.columns and df.shape[1] > 0:
                 df.columns = [*df.columns[:-1], 'target']
            elif df.shape[1] == 0:
                logger.error("Loaded CSV is empty or has no columns.")
                raise ValueError("CSV file is empty or has no columns.")
            logger.success(f"Successfully loaded data from {file_path}")
            return df
        except FileNotFoundError:
            logger.error(f"CSV file not found at {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file at {file_path} is empty.")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV from {file_path}: {e}")
            raise
    else:
        logger.info(f"Generating synthetic classification data: {n_samples} samples, {n_features} features, {n_classes} classes.")
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2), # Ensure at least 2 informative features
            n_redundant=max(0, n_features // 4),
            n_repeated=0,
            n_classes=n_classes,
            n_clusters_per_class=1, # Simplified cluster structure
            weights=None, # Balanced classes
            flip_y=0.01, # Small amount of noise
            random_state=random_state
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y
        logger.success("Successfully generated synthetic data.")
        return df

def preprocess_data(df: pd.DataFrame, target_column: str = 'target', test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses the data: separates features and target, normalizes features,
    and splits data into training and testing sets.

    Args:
        df: Pandas DataFrame containing features and the target column.
        target_column: Name of the target variable column.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for data splitting.

    Returns:
        A tuple (X_train, X_test, y_train, y_test) of numpy arrays.
    """
    try:
        logger.info("Starting data preprocessing...")
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in DataFrame.")
            raise ValueError(f"Target column '{target_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")

        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        if X.shape[0] == 0:
            logger.error("Feature set (X) is empty after dropping target column.")
            raise ValueError("Feature set (X) is empty. This might happen if the DataFrame only contained the target column or was empty.")
        if y.shape[0] == 0:
             logger.error("Target set (y) is empty.")
             raise ValueError("Target set (y) is empty.")


        # Check for NaN values before scaling
        if np.isnan(X).any():
            logger.warning("NaN values found in features. Attempting to fill with mean.")
            # Impute NaN values with the mean of their respective columns
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            if np.isnan(X).any(): # Check again if NaNs persist (e.g. whole column was NaN)
                logger.error("NaN values persist after attempting to fill with mean. Please clean data.")
                raise ValueError("NaN values persist in features after attempting to fill with mean. This can happen if an entire feature column is NaN.")


        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Features normalized using StandardScaler.")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y if np.unique(y).size > 1 else None
        )
        logger.success(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets.")
        
        return X_train, X_test, y_train, y_test
    except ValueError as ve:
        logger.error(f"ValueError during preprocessing: {ve}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during preprocessing: {e}")
        raise

