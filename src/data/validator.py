"""
Data validation module for customer segmentation project.

Provides functions to validate input data for quality, completeness,
and suitability for clustering algorithms.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass


def validate_dataframe(
    df: pd.DataFrame,
    min_rows: int = 10,
    allow_null: bool = False,
    numeric_only: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate pandas DataFrame for clustering.
    
    Args:
        df: Input DataFrame
        min_rows: Minimum number of rows required
        allow_null: Whether to allow null values
        numeric_only: Whether to allow only numeric columns
    
    Returns:
        Tuple of (is_valid, error_messages)
    
    Example:
        >>> df = pd.DataFrame({'age': [25, 30, 35], 'income': [50000, 60000, 70000]})
        >>> is_valid, errors = validate_dataframe(df)
        >>> if not is_valid:
        ...     print(errors)
    """
    errors = []
    
    # Check if DataFrame is empty
    if df is None or df.empty:
        errors.append("DataFrame is None or empty")
        return False, errors
    
    # Check minimum rows
    if df.shape[0] < min_rows:
        errors.append(f"Insufficient rows: {df.shape[0]} < {min_rows}")
    
    # Check for null values
    if not allow_null:
        null_cols = df.columns[df.isnull().any()].tolist()
        if null_cols:
            null_counts = df[null_cols].isnull().sum().to_dict()
            errors.append(f"Null values found in: {null_counts}")
    
    # Check for numeric columns if required
    if numeric_only:
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            errors.append(f"Non-numeric columns found: {non_numeric}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            errors.append(f"Infinite values in column: {col}")
    
    # Check for constant columns
    constant_cols = df.select_dtypes(include=[np.number]).columns[
        df.select_dtypes(include=[np.number]).nunique() == 1
    ].tolist()
    if constant_cols:
        errors.append(f"Constant columns (no variance): {constant_cols}")
    
    return len(errors) == 0, errors


def validate_features(
    X: np.ndarray,
    feature_names: List[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate feature matrix for clustering.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: Optional list of feature names
    
    Returns:
        Tuple of (is_valid, error_messages)
    
    Raises:
        DataValidationError: If validation fails
    
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> is_valid, errors = validate_features(X, ['age', 'income'])
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(error)
    """
    errors = []
    
    # Check if X is valid numpy array
    if not isinstance(X, np.ndarray):
        errors.append(f"X must be numpy array, got {type(X)}")
        return False, errors
    
    # Check dimensionality
    if X.ndim != 2:
        errors.append(f"X must be 2D array, got {X.ndim}D")
        return False, errors
    
    n_samples, n_features = X.shape
    
    # Check for minimum samples
    if n_samples < 2:
        errors.append(f"Too few samples: {n_samples}")
    
    # Check for minimum features
    if n_features == 0:
        errors.append("No features provided")
    
    # Check for null/NaN values
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        errors.append(f"NaN values found: {nan_count}")
    
    # Check for infinite values
    if np.isinf(X).any():
        inf_count = np.isinf(X).sum()
        errors.append(f"Infinite values found: {inf_count}")
    
    # Check feature names if provided
    if feature_names is not None:
        if len(feature_names) != n_features:
            errors.append(
                f"Feature names length {len(feature_names)} != n_features {n_features}"
            )
    
    # Check for zero variance features
    feature_variance = np.var(X, axis=0)
    zero_var_indices = np.where(feature_variance == 0)[0]
    if len(zero_var_indices) > 0:
        feature_indices = zero_var_indices.tolist()
        errors.append(f"Zero variance features at indices: {feature_indices}")
    
    # Check data types
    if not np.issubdtype(X.dtype, np.number):
        errors.append(f"X must contain numeric values, dtype is {X.dtype}")
    
    return len(errors) == 0, errors


def validate_labels(
    y: np.ndarray,
    X: np.ndarray = None
) -> Tuple[bool, List[str]]:
    """
    Validate clustering labels.
    
    Args:
        y: Predicted labels
        X: Optional feature matrix to check alignment
    
    Returns:
        Tuple of (is_valid, error_messages)
    
    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 1, 0])
        >>> is_valid, errors = validate_labels(y, X)
    """
    errors = []
    
    # Check if y is numpy array
    if not isinstance(y, np.ndarray):
        errors.append(f"y must be numpy array, got {type(y)}")
        return False, errors
    
    # Check dimensionality
    if y.ndim != 1:
        errors.append(f"y must be 1D array, got {y.ndim}D")
        return False, errors
    
    # Check for NaN values
    if np.isnan(y).any():
        errors.append("NaN values found in labels")
    
    # Check for non-integer labels
    if not np.issubdtype(y.dtype, np.integer):
        errors.append(f"Labels should be integers, got {y.dtype}")
    
    # Check alignment with X if provided
    if X is not None:
        if X.shape[0] != y.shape[0]:
            errors.append(
                f"Label count {y.shape[0]} != sample count {X.shape[0]}"
            )
    
    # Check for at least 1 cluster
    unique_labels = np.unique(y)
    if len(unique_labels) < 1:
        errors.append("No clusters found in labels")
    
    return len(errors) == 0, errors


def validate_and_prepare_data(
    df: pd.DataFrame,
    numeric_columns: List[str],
    drop_nulls: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Complete data validation and preparation pipeline.
    
    Args:
        df: Input DataFrame
        numeric_columns: Columns to use for clustering
        drop_nulls: Whether to drop rows with null values
    
    Returns:
        Tuple of (validated_array, feature_names)
    
    Raises:
        DataValidationError: If validation fails
    
    Example:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'income': [50000, 60000, 70000]
        ... })
        >>> X, features = validate_and_prepare_data(df, ['age', 'income'])
    """
    logger.info("Starting data validation and preparation")
    
    # Validate DataFrame
    is_valid, errors = validate_dataframe(df)
    if not is_valid:
        error_msg = "; ".join(errors)
        logger.error(f"DataFrame validation failed: {error_msg}")
        raise DataValidationError(f"DataFrame validation failed: {error_msg}")
    
    # Check if numeric columns exist
    missing_cols = [col for col in numeric_columns if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing columns: {missing_cols}")
    
    # Create a copy to avoid modifying original
    df_clean = df[numeric_columns].copy()
    
    # Drop nulls if requested
    if drop_nulls:
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        dropped_rows = initial_rows - len(df_clean)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with null values")
    
    # Convert to numpy array
    X = df_clean.values
    
    # Validate features
    is_valid, errors = validate_features(X, numeric_columns)
    if not is_valid:
        error_msg = "; ".join(errors)
        logger.error(f"Feature validation failed: {error_msg}")
        raise DataValidationError(f"Feature validation failed: {error_msg}")
    
    logger.info(f"Data preparation successful: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, numeric_columns
