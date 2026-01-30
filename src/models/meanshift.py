"""Mean Shift clustering algorithm."""
import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.utils.validator import validate_clustering_input, ClusteringError

logger = logging.getLogger(__name__)


def run_meanshift(
    X: np.ndarray,
    bandwidth: Optional[float] = None,
    seeds: Optional[np.ndarray] = None,
    bin_seeding: bool = False,
    max_iter: int = 300,
    **kwargs
) -> Tuple[MeanShift, np.ndarray, Dict[str, Any]]:
    """
    Run Mean Shift clustering - automatic cluster number selection.
    
    Key Feature: Like Affinity Propagation, automatically discovers clusters!
    Uses kernel density estimation and mode-seeking algorithm.
    
    Parameters:
    - bandwidth: Kernel bandwidth. If None, estimated from data.
                 Smaller = more clusters, larger = fewer clusters
    - seeds: Initial cluster centers. If None, derived from samples
    - bin_seeding: If True, faster but less accurate seed generation
    
    Algorithm:
    1. Start at each sample point (or seed)
    2. Iteratively move toward higher density regions
    3. Converge when no significant movement occurs
    4. Group nearby converged points as one cluster
    
    Args:
        X: Input feature matrix (n_samples, n_features)
        bandwidth: Kernel bandwidth (None = auto-estimate)
        seeds: Initial seeds (None = use samples)
        bin_seeding: Fast seed generation (less accurate)
        max_iter: Maximum iterations per sample
        **kwargs: Additional MeanShift parameters
    
    Returns:
        Tuple of (model, cluster_labels, metrics_dict)
        
        Metrics include:
        - n_clusters: Number of clusters found
        - bandwidth_used: Bandwidth value used
        - silhouette: Silhouette coefficient
    
    Raises:
        ClusteringError: If validation fails
    
    Example:
        >>> X = np.random.randn(100, 3)
        >>> model, labels, metrics = run_meanshift(X)
        >>> print(f"Found {metrics['n_clusters']} clusters")
        
        >>> # Control cluster count via bandwidth
        >>> model, labels, metrics = run_meanshift(X, bandwidth=0.5)
    """
    try:
        validate_clustering_input(X, "meanshift")
        
        # Auto-estimate bandwidth if not provided
        if bandwidth is None:
            bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=min(500, len(X)))
            logger.info(f"Auto-estimated bandwidth: {bandwidth:.4f}")
        
        if bandwidth <= 0:
            raise ClusteringError(f"bandwidth must be positive, got {bandwidth}")
        
        logger.info(
            f"Running Mean Shift: bandwidth={bandwidth:.4f}, "
            f"bin_seeding={bin_seeding}"
        )
        
        # Modern API
        model = MeanShift(
            bandwidth=bandwidth,
            seeds=seeds,
            bin_seeding=bin_seeding,
            max_iter=max_iter,
            **kwargs
        )
        
        labels = model.fit_predict(X)
        
        # Analyze results
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        metrics = {
            "n_clusters": int(n_clusters),
            "bandwidth_used": float(bandwidth),
            "n_iterations": int(model.n_iter),
            "cluster_centers_count": int(len(model.cluster_centers_)),
        }
        
        # Calculate evaluation metrics
        if n_clusters > 1:
            try:
                metrics["silhouette"] = float(silhouette_score(X, labels))
                metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
                logger.debug("Mean Shift metrics calculated")
            except Exception as e:
                logger.warning(f"Could not calculate metrics: {e}")
        else:
            logger.warning("Only 1 cluster found")
        
        logger.info(f"Mean Shift completed: {n_clusters} clusters found")
        
        return model, labels, metrics
    
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"Mean Shift failed: {str(e)}", exc_info=True)
        raise ClusteringError(f"Mean Shift failed: {str(e)}")


def tune_meanshift_bandwidth(
    X: np.ndarray,
    quantile_range: np.ndarray = None,
    n_samples: int = 500
) -> Dict[float, Dict]:
    """
    Tune bandwidth parameter using quantile range.
    
    Bandwidth estimation: Uses quantile of k-nearest neighbor distances.
    Lower quantile = smaller bandwidth = more clusters
    
    Args:
        X: Input feature matrix
        quantile_range: Array of quantiles (0-1) to test
        n_samples: Samples for bandwidth estimation
    
    Returns:
        Dictionary mapping bandwidth -> metrics
    
    Example:
        >>> quantiles = np.linspace(0.1, 0.5, 10)
        >>> results = tune_meanshift_bandwidth(X, quantile_range=quantiles)
        >>> # Find quantile that gives desired cluster count
        >>> for bw, metrics in results.items():
        ...     print(f"Bandwidth={bw:.4f}: {metrics['n_clusters']} clusters")
    """
    if quantile_range is None:
        quantile_range = np.linspace(0.1, 0.5, 10)
    
    logger.info(f"Tuning Mean Shift bandwidth: {len(quantile_range)} values")
    
    results = {}
    for quantile in quantile_range:
        try:
            bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)
            _, labels, metrics = run_meanshift(X, bandwidth=bandwidth)
            results[float(bandwidth)] = metrics
            logger.debug(
                f"Quantile={quantile:.2f}, Bandwidth={bandwidth:.4f}: "
                f"{metrics['n_clusters']} clusters"
            )
        except Exception as e:
            logger.warning(f"Failed at quantile={quantile}: {e}")
            results[float(quantile)] = {"error": str(e)}
    
    return results
