"""DBSCAN (Density-Based Spatial Clustering) algorithm."""
import logging
import numpy as np
from typing import Tuple, Dict, Any

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

from src.utils.validator import validate_clustering_input, ClusteringError

logger = logging.getLogger(__name__)


def run_dbscan(
    X: np.ndarray,
    eps: float,
    min_samples: int = 5,
    metric: str = "euclidean",
    **kwargs
) -> Tuple[DBSCAN, np.ndarray, Dict[str, Any]]:
    """
    Run DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    
    Parameters explained:
    - eps: Maximum distance between two samples. Low values = fewer clusters,
           high values = more points in clusters. Typical: 0.1 to 1.0
    - min_samples: Minimum samples in eps-neighborhood to form core point.
                   Typical: 5 to 10. Higher = stricter clustering
    - metric: Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
    
    Args:
        X: Input feature matrix (n_samples, n_features)
        eps: Maximum distance between samples in same neighborhood
        min_samples: Minimum samples in eps-neighborhood (default: 5)
        metric: Distance metric (default: 'euclidean')
        **kwargs: Additional DBSCAN parameters
    
    Returns:
        Tuple of (model, cluster_labels, metrics_dict)
        
        Note: Labels include -1 for noise points
        
        Metrics include:
        - n_clusters: Number of clusters found
        - n_noise_points: Number of noise points (-1 label)
        - silhouette: Only calculated for core points (label != -1)
        - davies_bouldin: Only for 2+ clusters
    
    Raises:
        ClusteringError: If validation fails
    
    Example:
        >>> X = np.random.randn(100, 3)
        >>> model, labels, metrics = run_dbscan(X, eps=0.5, min_samples=5)
        >>> print(f"Found {metrics['n_clusters']} clusters, {metrics['n_noise_points']} noise points")
    """
    try:
        validate_clustering_input(X, "dbscan", min_samples=2)
        
        # Validate parameters
        if eps <= 0:
            raise ClusteringError(f"eps must be positive, got {eps}")
        
        if min_samples < 1:
            raise ClusteringError(f"min_samples must be >= 1, got {min_samples}")
        
        logger.info(
            f"Running DBSCAN: eps={eps}, min_samples={min_samples}, "
            f"metric='{metric}'"
        )
        
        # Modern API
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, **kwargs)
        labels = model.fit_predict(X)
        
        # Analyze results
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = (labels == -1).sum()
        n_core_points = (model.core_sample_indices_.size)
        
        metrics = {
            "n_clusters": int(n_clusters),
            "n_noise_points": int(n_noise),
            "n_core_points": int(n_core_points),
            "noise_percentage": float((n_noise / X.shape[0]) * 100),
        }
        
        # Calculate metrics only for core points (exclude noise)
        if n_clusters > 1:
            X_clean = X[labels != -1]
            labels_clean = labels[labels != -1]
            
            if len(X_clean) > 0:
                try:
                    metrics["silhouette"] = float(silhouette_score(X_clean, labels_clean))
                    metrics["davies_bouldin"] = float(davies_bouldin_score(X_clean, labels_clean))
                    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_clean, labels_clean))
                    logger.debug("DBSCAN metrics calculated (excluding noise points)")
                except Exception as e:
                    logger.warning(f"Could not calculate metrics: {e}")
        
        logger.info(
            f"DBSCAN completed: {n_clusters} clusters, "
            f"{n_noise} noise points ({metrics['noise_percentage']:.1f}%)"
        )
        
        return model, labels, metrics
    
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"DBSCAN failed: {str(e)}", exc_info=True)
        raise ClusteringError(f"DBSCAN failed: {str(e)}")


def find_optimal_eps(
    X: np.ndarray,
    min_samples: int = 5,
    percentile: int = 90,
    k: int = None
) -> float:
    """
    Find optimal eps value using k-distance graph method.
    
    Algorithm:
    1. Compute k-nearest neighbor distances
    2. Sort distances
    3. Return distance at specified percentile
    
    Args:
        X: Input feature matrix
        min_samples: Min samples (used as k for neighbors)
        percentile: Percentile of sorted distances (0-100)
        k: Number of neighbors (default: min_samples)
    
    Returns:
        Recommended eps value
    
    Example:
        >>> eps = find_optimal_eps(X, min_samples=5, percentile=90)
        >>> print(f"Recommended eps: {eps:.3f}")
    """
    if k is None:
        k = min_samples
    
    logger.info(f"Finding optimal eps (k={k}, percentile={percentile})")
    
    # Calculate k-distances
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # Sort distances (take max distance in each neighborhood)
    distances = np.sort(distances[:, k - 1], axis=0)
    
    # Get percentile
    optimal_eps = distances[int(len(distances) * percentile / 100)]
    
    logger.info(f"Optimal eps found: {optimal_eps:.4f} (percentile {percentile})")
    
    return optimal_eps
