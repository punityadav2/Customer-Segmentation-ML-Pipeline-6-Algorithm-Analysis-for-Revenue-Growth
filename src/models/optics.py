"""OPTICS (Ordering Points To Identify Clustering Structure) algorithm."""
import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional

from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.utils.validator import validate_clustering_input, ClusteringError

logger = logging.getLogger(__name__)


def run_optics(
    X: np.ndarray,
    min_samples: int = 5,
    max_eps: Optional[float] = None,
    xi: Optional[float] = None,
    metric: str = "euclidean",
    cluster_method: str = "xi",
    **kwargs
) -> Tuple[OPTICS, np.ndarray, Dict[str, Any]]:
    """
    Run OPTICS clustering - density-based with automatic cluster extraction.
    
    OPTICS Advantages:
    - Like DBSCAN but doesn't require eps parameter
    - Handles varying density clusters better
    - Can identify hierarchical structure
    - Modern alternative to DBSCAN
    
    Parameters:
    - min_samples: Minimum points to form core point (like DBSCAN)
    - max_eps: Maximum distance (bounds search region)
    - xi: Steepness threshold for cluster extraction (0-1)
         Higher = stricter, fewer clusters
    - cluster_method: 'xi' (new) or 'dbscan' (legacy)
    
    Args:
        X: Input feature matrix (n_samples, n_features)
        min_samples: Minimum samples in neighborhood (default: 5)
        max_eps: Maximum distance (None = infinity)
        xi: Steepness parameter for cluster extraction
        metric: Distance metric ('euclidean', 'manhattan', etc.)
        cluster_method: Method for cluster extraction
        **kwargs: Additional OPTICS parameters
    
    Returns:
        Tuple of (model, cluster_labels, metrics_dict)
        
        Note: Labels include -1 for noise points
        
        Metrics include:
        - n_clusters: Number of clusters found
        - n_noise_points: Number of noise points
        - reachability_range: Min-max reachability distance
    
    Raises:
        ClusteringError: If validation fails
    
    Example:
        >>> X = np.random.randn(100, 3)
        >>> model, labels, metrics = run_optics(X, min_samples=5)
        >>> print(f"Found {metrics['n_clusters']} clusters")
        
        >>> # Stricter clustering with xi
        >>> model, labels, metrics = run_optics(X, min_samples=5, xi=0.05)
    """
    try:
        validate_clustering_input(X, "optics", min_samples=2)
        
        # Validate parameters
        if min_samples < 1:
            raise ClusteringError(f"min_samples must be >= 1, got {min_samples}")
        
        if max_eps is not None and max_eps <= 0:
            raise ClusteringError(f"max_eps must be positive or None, got {max_eps}")
        
        if xi is not None and not (0 < xi < 1):
            raise ClusteringError(f"xi must be in (0, 1), got {xi}")
        
        if cluster_method not in ["xi", "dbscan"]:
            raise ClusteringError(f"cluster_method must be 'xi' or 'dbscan', got {cluster_method}")
        
        logger.info(
            f"Running OPTICS: min_samples={min_samples}, max_eps={max_eps}, "
            f"xi={xi}, cluster_method='{cluster_method}'"
        )
        
        # Modern API (scikit-learn 0.23+)
        model = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            xi=xi,
            metric=metric,
            cluster_method=cluster_method,
            **kwargs
        )
        
        labels = model.fit_predict(X)
        
        # Analyze results
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = (labels == -1).sum()
        
        # Get reachability info
        reachability = model.reachability_[model.ordering_]
        reachability_valid = reachability[reachability != np.inf]
        
        metrics = {
            "n_clusters": int(n_clusters),
            "n_noise_points": int(n_noise),
            "noise_percentage": float((n_noise / X.shape[0]) * 100),
            "n_core_points": int((reachability != np.inf).sum()),
        }
        
        if len(reachability_valid) > 0:
            metrics["reachability_min"] = float(reachability_valid.min())
            metrics["reachability_max"] = float(reachability_valid.max())
        
        # Calculate metrics only for core points
        if n_clusters > 1:
            X_clean = X[labels != -1]
            labels_clean = labels[labels != -1]
            
            if len(X_clean) > 0:
                try:
                    metrics["silhouette"] = float(silhouette_score(X_clean, labels_clean))
                    metrics["davies_bouldin"] = float(davies_bouldin_score(X_clean, labels_clean))
                    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_clean, labels_clean))
                    logger.debug("OPTICS metrics calculated (excluding noise)")
                except Exception as e:
                    logger.warning(f"Could not calculate metrics: {e}")
        
        logger.info(
            f"OPTICS completed: {n_clusters} clusters, "
            f"{n_noise} noise points ({metrics['noise_percentage']:.1f}%)"
        )
        
        return model, labels, metrics
    
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"OPTICS failed: {str(e)}", exc_info=True)
        raise ClusteringError(f"OPTICS failed: {str(e)}")


def tune_optics_xi(
    X: np.ndarray,
    min_samples: int = 5,
    xi_range: np.ndarray = None,
    max_eps: Optional[float] = None
) -> Dict[float, Dict]:
    """
    Tune xi parameter for OPTICS cluster extraction.
    
    Xi Parameter:
    - Low xi (0.01-0.05): Very strict, many clusters
    - Medium xi (0.05-0.1): Moderate clustering
    - High xi (0.1-0.5): Loose, few clusters
    
    Args:
        X: Input feature matrix
        min_samples: Minimum samples parameter
        xi_range: Array of xi values to test
        max_eps: Maximum distance
    
    Returns:
        Dictionary mapping xi -> metrics
    
    Example:
        >>> xi_vals = np.linspace(0.01, 0.5, 15)
        >>> results = tune_optics_xi(X, xi_range=xi_vals)
        >>> # Find xi for desired cluster count
    """
    if xi_range is None:
        xi_range = np.linspace(0.01, 0.5, 15)
    
    logger.info(f"Tuning OPTICS xi: {len(xi_range)} values")
    
    results = {}
    for xi in xi_range:
        try:
            _, labels, metrics = run_optics(
                X, min_samples=min_samples, max_eps=max_eps, xi=xi
            )
            results[float(xi)] = metrics
            logger.debug(f"Xi={xi:.3f}: {metrics['n_clusters']} clusters")
        except Exception as e:
            logger.warning(f"Failed at xi={xi}: {e}")
            results[float(xi)] = {"error": str(e)}
    
    return results
