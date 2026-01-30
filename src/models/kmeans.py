"""K-Means clustering algorithm with modern scikit-learn API."""
import logging
import numpy as np
from typing import Tuple, Dict, Any

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.utils.validator import validate_clustering_input, ClusteringError

logger = logging.getLogger(__name__)


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    init: str = "k-means++",
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int = 42,
    verbose: int = 0,
    **kwargs
) -> Tuple[KMeans, np.ndarray, Dict[str, float]]:
    """
    Run K-Means clustering with error handling and validation.
    
    Modern API (scikit-learn 1.0+):
    - Uses 'k-means++' initialization (faster, better results)
    - Automatic n_init handling
    - Enhanced convergence criteria
    
    Args:
        X: Input feature matrix (n_samples, n_features)
        n_clusters: Number of clusters (2-100 typical range)
        init: Initialization method ('k-means++' or 'random')
        max_iter: Maximum iterations (default: 300)
        tol: Convergence tolerance (default: 1e-4)
        random_state: Random state for reproducibility
        verbose: Verbosity level (0-2)
        **kwargs: Additional KMeans parameters
    
    Returns:
        Tuple of (model, cluster_labels, metrics_dict)
        
        Metrics include:
        - inertia: Sum of squared distances to nearest cluster center
        - silhouette: Silhouette coefficient (-1 to 1, higher is better)
        - davies_bouldin: Davies-Bouldin Index (lower is better)
        - calinski_harabasz: Calinski-Harabasz Index (higher is better)
    
    Raises:
        ClusteringError: If validation fails or clustering encounters issues
    
    Example:
        >>> X = np.random.randn(100, 3)
        >>> model, labels, metrics = run_kmeans(X, n_clusters=3)
        >>> print(f"Silhouette Score: {metrics['silhouette']:.3f}")
        >>> print(f"Inertia: {metrics['inertia']:.2f}")
    """
    try:
        validate_clustering_input(X, "kmeans")
        
        # Validate parameters
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ClusteringError(f"n_clusters must be positive integer, got {n_clusters}")
        
        if n_clusters > X.shape[0]:
            raise ClusteringError(
                f"n_clusters ({n_clusters}) cannot exceed n_samples ({X.shape[0]})"
            )
        
        if init not in ["k-means++", "random"]:
            raise ClusteringError(f"init must be 'k-means++' or 'random', got '{init}'")
        
        logger.info(
            f"Running K-Means: n_clusters={n_clusters}, init='{init}', "
            f"max_iter={max_iter}, random_state={random_state}"
        )
        
        # Modern scikit-learn API
        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init=10,  # Modern default (automatic in newer versions)
            verbose=verbose,
            **kwargs
        )
        
        labels = model.fit_predict(X)
        
        # Calculate comprehensive metrics
        metrics = {
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
        }
        
        try:
            metrics["silhouette"] = float(silhouette_score(X, labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            logger.debug(f"K-Means metrics calculated successfully")
        except Exception as e:
            logger.warning(f"Could not calculate all metrics: {e}")
        
        logger.info(
            f"K-Means completed successfully: "
            f"Inertia={metrics['inertia']:.2f}, "
            f"Iterations={metrics['n_iter']}"
        )
        
        return model, labels, metrics
    
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"K-Means clustering failed: {str(e)}", exc_info=True)
        raise ClusteringError(f"K-Means failed: {str(e)}")


def find_optimal_clusters(
    X: np.ndarray,
    cluster_range: range = range(2, 11),
    metric: str = "silhouette",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Find optimal number of clusters using elbow method or silhouette score.
    
    Args:
        X: Input feature matrix
        cluster_range: Range of cluster numbers to test
        metric: Metric to optimize ('silhouette', 'davies_bouldin', 'calinski_harabasz')
        random_state: Random state
    
    Returns:
        Dictionary with:
        - optimal_clusters: Best number of clusters
        - scores: List of metric scores
        - metrics_by_cluster: Detailed metrics for each cluster count
    
    Example:
        >>> result = find_optimal_clusters(X, range(2, 11), metric='silhouette')
        >>> print(f"Optimal clusters: {result['optimal_clusters']}")
    """
    if metric not in ["silhouette", "davies_bouldin", "calinski_harabasz"]:
        raise ClusteringError(f"Unknown metric: {metric}")
    
    logger.info(f"Finding optimal clusters using {metric} metric")
    
    scores = []
    metrics_list = []
    
    for n_clusters in cluster_range:
        try:
            _, labels, metrics = run_kmeans(
                X, n_clusters, random_state=random_state, verbose=0
            )
            score = metrics.get(metric, 0)
            scores.append(score)
            metrics_list.append({"n_clusters": n_clusters, **metrics})
        except Exception as e:
            logger.warning(f"Failed for n_clusters={n_clusters}: {e}")
            scores.append(None)
    
    # Find optimal
    if metric == "davies_bouldin":
        # Lower is better for davies_bouldin
        valid_scores = [s for s in scores if s is not None]
        optimal_idx = scores.index(min(valid_scores))
    else:
        # Higher is better for silhouette and calinski_harabasz
        valid_scores = [s for s in scores if s is not None]
        optimal_idx = scores.index(max(valid_scores))
    
    optimal_clusters = list(cluster_range)[optimal_idx]
    
    logger.info(f"Optimal clusters found: {optimal_clusters} (score: {scores[optimal_idx]:.3f})")
    
    return {
        "optimal_clusters": optimal_clusters,
        "scores": scores,
        "metrics_by_cluster": metrics_list,
        "metric_used": metric
    }
