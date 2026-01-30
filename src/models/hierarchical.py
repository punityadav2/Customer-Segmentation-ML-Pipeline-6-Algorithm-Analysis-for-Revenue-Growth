"""Hierarchical (Agglomerative) clustering algorithm."""
import logging
import numpy as np
from typing import Tuple, Dict, Any, Literal

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.utils.validator import validate_clustering_input, ClusteringError

logger = logging.getLogger(__name__)

LinkageMethod = Literal["ward", "complete", "average", "single"]


def run_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    linkage: LinkageMethod = "ward",
    distance_threshold: float = None,
    **kwargs
) -> Tuple[AgglomerativeClustering, np.ndarray, Dict[str, float]]:
    """
    Run Hierarchical (Agglomerative) clustering with modern scikit-learn API.
    
    Linkage Methods:
    - 'ward': Minimizes variance (default, recommended for K-means-like results)
    - 'complete': Maximum distance between clusters
    - 'average': Average distance between clusters
    - 'single': Minimum distance between clusters
    
    Args:
        X: Input feature matrix (n_samples, n_features)
        n_clusters: Number of clusters to form
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        distance_threshold: Threshold distance (alternative to n_clusters)
        **kwargs: Additional AgglomerativeClustering parameters
    
    Returns:
        Tuple of (model, cluster_labels, metrics_dict)
        
        Metrics include:
        - silhouette: Silhouette coefficient (-1 to 1)
        - davies_bouldin: Davies-Bouldin Index (lower is better)
        - calinski_harabasz: Calinski-Harabasz Index (higher is better)
        - n_clusters: Number of clusters found
    
    Raises:
        ClusteringError: If validation fails
    
    Example:
        >>> X = np.random.randn(100, 3)
        >>> model, labels, metrics = run_hierarchical(X, n_clusters=4, linkage='ward')
        >>> print(f"Silhouette: {metrics['silhouette']:.3f}")
    """
    try:
        validate_clustering_input(X, "hierarchical")
        
        # Validate parameters
        if linkage not in ["ward", "complete", "average", "single"]:
            raise ClusteringError(
                f"linkage must be 'ward', 'complete', 'average', or 'single', "
                f"got '{linkage}'"
            )
        
        if distance_threshold is None:
            if not isinstance(n_clusters, int) or n_clusters <= 0 or n_clusters > X.shape[0]:
                raise ClusteringError(
                    f"n_clusters must be in range [1, {X.shape[0]}], got {n_clusters}"
                )
        
        logger.info(
            f"Running Hierarchical clustering: linkage='{linkage}', "
            f"n_clusters={n_clusters if distance_threshold is None else 'auto'}"
        )
        
        # Modern API - use n_clusters or distance_threshold
        model = AgglomerativeClustering(
            n_clusters=n_clusters if distance_threshold is None else None,
            linkage=linkage,
            distance_threshold=distance_threshold,
            **kwargs
        )
        
        labels = model.fit_predict(X)
        n_clusters_found = len(np.unique(labels))
        
        metrics = {
            "n_clusters": n_clusters_found,
            "n_leaves": model.n_leaves_,
        }
        
        try:
            if n_clusters_found > 1:
                metrics["silhouette"] = float(silhouette_score(X, labels))
                metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
                logger.debug("Hierarchical metrics calculated successfully")
            else:
                logger.warning("Only 1 cluster found, skipping similarity metrics")
        except Exception as e:
            logger.warning(f"Could not calculate metrics: {e}")
        
        logger.info(f"Hierarchical completed: {n_clusters_found} clusters found")
        return model, labels, metrics
    
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"Hierarchical clustering failed: {str(e)}", exc_info=True)
        raise ClusteringError(f"Hierarchical failed: {str(e)}")


def compare_linkage_methods(
    X: np.ndarray,
    n_clusters: int,
    linkage_methods: list = None
) -> Dict[str, Dict]:
    """
    Compare different linkage methods for hierarchical clustering.
    
    Args:
        X: Input feature matrix
        n_clusters: Number of clusters
        linkage_methods: List of linkage methods to compare (default: all)
    
    Returns:
        Dictionary with results for each linkage method
    
    Example:
        >>> results = compare_linkage_methods(X, n_clusters=4)
        >>> for method, metrics in results.items():
        ...     print(f"{method}: Silhouette = {metrics['silhouette']:.3f}")
    """
    if linkage_methods is None:
        linkage_methods = ["ward", "complete", "average", "single"]
    
    logger.info(f"Comparing linkage methods: {linkage_methods}")
    
    results = {}
    for linkage in linkage_methods:
        try:
            _, labels, metrics = run_hierarchical(X, n_clusters, linkage=linkage)
            results[linkage] = metrics
            logger.debug(f"✓ {linkage}: silhouette={metrics.get('silhouette', 'N/A'):.3f}")
        except Exception as e:
            logger.warning(f"✗ {linkage} failed: {e}")
            results[linkage] = {"error": str(e)}
    
    return results
