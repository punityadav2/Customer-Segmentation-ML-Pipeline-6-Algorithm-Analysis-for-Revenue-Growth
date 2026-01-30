"""
Clustering evaluation metrics module.

Provides functions to compute and compare clustering quality metrics
including silhouette score, Davies-Bouldin index, and Calinski-Harabasz score.
"""

from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score
)
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MetricsError(Exception):
    """Custom exception for metrics calculation failures."""
    pass


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    handle_noise: bool = True
) -> Dict[str, Any]:
    """
    Calculate comprehensive clustering metrics.
    
    Computes silhouette score, Davies-Bouldin index, and Calinski-Harabasz score.
    Handles noise points from DBSCAN/OPTICS algorithms if present.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,). Can contain -1 for noise points
        handle_noise: Whether to exclude noise points when calculating metrics
    
    Returns:
        Dictionary with metrics:
        - 'Clusters': Number of clusters
        - 'Silhouette Score': Silhouette coefficient (-1 to 1, higher is better)
        - 'Davies-Bouldin Score': Davies-Bouldin index (lower is better)
        - 'Calinski-Harabasz Score': Calinski-Harabasz index (higher is better)
        - 'Noise Points': Number of noise points (if present)
    
    Raises:
        MetricsError: If metrics calculation fails
    
    Example:
        >>> X = np.array([[1, 2], [2, 1], [8, 9], [9, 8]])
        >>> labels = np.array([0, 0, 1, 1])
        >>> metrics = evaluate_clustering(X, labels)
        >>> print(f"Silhouette: {metrics['Silhouette Score']:.3f}")
    """
    try:
        # Count clusters and noise points
        unique_labels = np.unique(labels)
        has_noise = -1 in labels
        n_clusters = len(unique_labels) - (1 if has_noise else 0)
        n_noise = (labels == -1).sum() if has_noise else 0
        
        logger.info(f"Evaluating clustering: {n_clusters} clusters, {n_noise} noise points")
        
        # Prepare data for metric calculation
        if handle_noise and has_noise:
            mask = labels != -1
            X_clean = X[mask]
            labels_clean = labels[mask]
        else:
            X_clean = X
            labels_clean = labels
        
        # If less than 2 clusters, metrics are not meaningful
        if n_clusters < 2:
            logger.warning("Less than 2 clusters found, returning NaN for metrics")
            return {
                'Clusters': n_clusters,
                'Silhouette Score': np.nan,
                'Davies-Bouldin Score': np.nan,
                'Calinski-Harabasz Score': np.nan,
                'Noise Points': int(n_noise)
            }
        
        # Calculate metrics
        try:
            silhouette = silhouette_score(X_clean, labels_clean)
        except Exception as e:
            logger.warning(f"Silhouette score calculation failed: {e}")
            silhouette = np.nan
        
        try:
            davies_bouldin = davies_bouldin_score(X_clean, labels_clean)
        except Exception as e:
            logger.warning(f"Davies-Bouldin score calculation failed: {e}")
            davies_bouldin = np.nan
        
        try:
            calinski_harabasz = calinski_harabasz_score(X_clean, labels_clean)
        except Exception as e:
            logger.warning(f"Calinski-Harabasz score calculation failed: {e}")
            calinski_harabasz = np.nan
        
        result = {
            'Clusters': int(n_clusters),
            'Silhouette Score': float(silhouette),
            'Davies-Bouldin Score': float(davies_bouldin),
            'Calinski-Harabasz Score': float(calinski_harabasz),
            'Noise Points': int(n_noise)
        }
        
        logger.info(f"Metrics calculated: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Metrics evaluation failed: {str(e)}")
        raise MetricsError(f"Failed to evaluate clustering: {str(e)}")


def compare_algorithms(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'Silhouette Score'
) -> pd.DataFrame:
    """
    Compare multiple clustering algorithms side-by-side.
    
    Args:
        results: Dictionary mapping algorithm names to metric dictionaries
        metric: Primary metric to sort by
    
    Returns:
        Comparison DataFrame with metrics sorted by primary metric
    
    Example:
        >>> results = {
        ...     'KMeans': {'Clusters': 3, 'Silhouette Score': 0.65},
        ...     'DBSCAN': {'Clusters': 3, 'Silhouette Score': 0.60}
        ... }
        >>> comparison = compare_algorithms(results, 'Silhouette Score')
        >>> print(comparison)
    """
    if not results:
        raise MetricsError("No results to compare")
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    # Sort by specified metric if it exists
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False, na_position='last')
        logger.info(f"Comparison sorted by {metric}")
    
    return df


def get_best_algorithm(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'Silhouette Score'
) -> Tuple[str, float]:
    """
    Identify best performing algorithm based on metric.
    
    Args:
        results: Dictionary mapping algorithm names to metric dictionaries
        metric: Metric to evaluate on
    
    Returns:
        Tuple of (algorithm_name, metric_value)
    
    Raises:
        MetricsError: If metric not found or no valid results
    
    Example:
        >>> results = {
        ...     'KMeans': {'Silhouette Score': 0.65},
        ...     'DBSCAN': {'Silhouette Score': 0.60}
        ... }
        >>> best_algo, score = get_best_algorithm(results, 'Silhouette Score')
        >>> print(f"Best: {best_algo} with score {score}")
    """
    if not results:
        raise MetricsError("No results to evaluate")
    
    valid_results = {
        algo: metrics for algo, metrics in results.items()
        if metric in metrics and not np.isnan(metrics[metric])
    }
    
    if not valid_results:
        raise MetricsError(f"No valid {metric} values found in results")
    
    # Higher is better for Silhouette and Calinski-Harabasz
    # Lower is better for Davies-Bouldin
    ascending = metric == 'Davies-Bouldin Score'
    
    best_algo = min(
        valid_results.items(),
        key=lambda x: x[1][metric] if ascending else -x[1][metric]
    )
    
    logger.info(f"Best algorithm: {best_algo[0]} ({metric}: {best_algo[1][metric]:.3f})")
    return best_algo[0], best_algo[1][metric]
