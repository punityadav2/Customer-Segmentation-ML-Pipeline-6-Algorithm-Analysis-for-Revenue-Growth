"""Input validation utilities for clustering algorithms."""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class ClusteringError(Exception):
    """Custom exception for clustering-related errors."""
    pass


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_clustering_input(
    X: np.ndarray,
    algorithm: str = "clustering",
    min_samples: int = 2
) -> None:
    """
    Validate input data for clustering operations.
    
    Checks:
    - X is not None
    - X is numpy array
    - Sufficient number of samples
    - Has features
    - No NaN values
    - No infinite values
    
    Args:
        X: Input feature matrix (n_samples, n_features)
        algorithm: Name of algorithm (for logging)
        min_samples: Minimum required samples
    
    Raises:
        ClusteringError: If validation fails
    
    Example:
        >>> X = np.random.randn(100, 5)
        >>> validate_clustering_input(X, "kmeans")
    """
    # Check if None
    if X is None:
        raise ClusteringError("Input data X is None")
    
    # Check type
    if not isinstance(X, np.ndarray):
        raise ClusteringError(
            f"X must be numpy array, got {type(X).__name__}"
        )
    
    # Check dimensions
    if X.ndim != 2:
        raise ClusteringError(
            f"X must be 2D array, got {X.ndim}D"
        )
    
    n_samples, n_features = X.shape
    
    # Check sample count
    if n_samples < min_samples:
        raise ClusteringError(
            f"Need at least {min_samples} samples, got {n_samples}"
        )
    
    # Check feature count
    if n_features == 0:
        raise ClusteringError("Input data has 0 features")
    
    # Check for NaN values
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        raise ClusteringError(
            f"Input data contains {nan_count} NaN values. "
            f"Please handle missing values before clustering."
        )
    
    # Check for infinite values
    if np.isinf(X).any():
        inf_count = np.isinf(X).sum()
        raise ClusteringError(
            f"Input data contains {inf_count} infinite values. "
            f"Please check for overflow or invalid computations."
        )
    
    # Check for empty rows (all zeros)
    if (X == 0).all(axis=1).any():
        logger.warning("Input contains some all-zero samples")
    
    logger.debug(
        f"✓ Validation passed for {algorithm}: "
        f"shape={X.shape}, dtype={X.dtype}"
    )


def validate_labels(
    labels: np.ndarray,
    X: np.ndarray
) -> None:
    """
    Validate cluster labels from clustering algorithm.
    
    Args:
        labels: Cluster labels array
        X: Original feature matrix
    
    Raises:
        ValidationError: If validation fails
    """
    if labels is None:
        raise ValidationError("Labels are None")
    
    if not isinstance(labels, np.ndarray):
        raise ValidationError(f"Labels must be numpy array, got {type(labels)}")
    
    if len(labels) != X.shape[0]:
        raise ValidationError(
            f"Labels length ({len(labels)}) != X samples ({X.shape[0]})"
        )
    
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValidationError(f"Labels must be integers, got {labels.dtype}")
    
    logger.debug(f"✓ Labels validation passed: {len(labels)} labels")


def validate_metrics(
    metrics: dict,
    expected_keys: list = None
) -> None:
    """
    Validate metrics dictionary from clustering.
    
    Args:
        metrics: Metrics dictionary
        expected_keys: List of expected metric keys
    
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(metrics, dict):
        raise ValidationError(f"Metrics must be dict, got {type(metrics)}")
    
    if len(metrics) == 0:
        raise ValidationError("Metrics dictionary is empty")
    
    if expected_keys:
        missing_keys = set(expected_keys) - set(metrics.keys())
        if missing_keys:
            logger.warning(f"Missing expected metrics: {missing_keys}")
    
    logger.debug(f"✓ Metrics validation passed: {len(metrics)} metrics")


def validate_n_clusters(
    n_clusters: int,
    n_samples: int,
    algorithm: str = "clustering"
) -> None:
    """
    Validate number of clusters parameter.
    
    Args:
        n_clusters: Number of clusters
        n_samples: Number of samples
        algorithm: Algorithm name (for logging)
    
    Raises:
        ValidationError: If invalid
    """
    if not isinstance(n_clusters, int):
        raise ValidationError(f"n_clusters must be int, got {type(n_clusters)}")
    
    if n_clusters < 1:
        raise ValidationError(f"n_clusters must be >= 1, got {n_clusters}")
    
    if n_clusters > n_samples:
        raise ValidationError(
            f"n_clusters ({n_clusters}) > n_samples ({n_samples})"
        )
    
    logger.debug(f"✓ n_clusters validation passed: {n_clusters} clusters for {algorithm}")
