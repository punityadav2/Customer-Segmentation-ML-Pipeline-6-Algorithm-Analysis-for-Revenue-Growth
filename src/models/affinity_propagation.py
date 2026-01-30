"""Affinity Propagation clustering algorithm."""
import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional

from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.utils.validator import validate_clustering_input, ClusteringError

logger = logging.getLogger(__name__)


def run_affinity_propagation(
    X: np.ndarray,
    preference: Optional[float] = None,
    damping: float = 0.9,
    max_iter: int = 200,
    convergence_iter: int = 15,
    random_state: int = 42,
    verbose: bool = False,
    **kwargs
) -> Tuple[AffinityPropagation, np.ndarray, Dict[str, Any]]:
    """
    Run Affinity Propagation clustering - automatic cluster number selection.
    
    Key Feature: No need to specify number of clusters beforehand!
    Automatically discovers optimal number by message passing between samples.
    
    Parameters:
    - preference: Preference for each sample being a cluster center.
                  Default (None) = median of similarities.
                  Higher values = more clusters.
                  Lower values = fewer clusters.
    - damping: Damping factor for convergence (0.5-1.0).
              Higher = slower convergence but more stable.
              Default = 0.9 (recommended).
    
    Args:
        X: Input feature matrix (n_samples, n_features)
        preference: Cluster center preference (None for auto)
        damping: Damping factor for iterative updates (0.5-1.0)
        max_iter: Maximum number of iterations
        convergence_iter: Iterations with no change to declare convergence
        random_state: Random state for reproducibility
        verbose: Print convergence messages
        **kwargs: Additional AffinityPropagation parameters
    
    Returns:
        Tuple of (model, cluster_labels, metrics_dict)
        
        Metrics include:
        - n_clusters: Number of clusters found
        - n_exemplars: Number of exemplar points (cluster centers)
        - n_iter: Iterations until convergence
        - silhouette: Silhouette coefficient
    
    Raises:
        ClusteringError: If validation fails
    
    Example:
        >>> X = np.random.randn(100, 3)
        >>> model, labels, metrics = run_affinity_propagation(X)
        >>> print(f"Found {metrics['n_clusters']} clusters")
        
        >>> # Control number of clusters via preference
        >>> model, labels, metrics = run_affinity_propagation(X, preference=-10)
        >>> print(f"More clusters: {metrics['n_clusters']}")
    """
    try:
        validate_clustering_input(X, "affinity_propagation")
        
        # Validate parameters
        if not (0.5 <= damping < 1.0):
            raise ClusteringError(
                f"damping must be in [0.5, 1.0), got {damping}"
            )
        
        if max_iter < 1:
            raise ClusteringError(f"max_iter must be >= 1, got {max_iter}")
        
        logger.info(
            f"Running Affinity Propagation: preference={preference}, "
            f"damping={damping}, max_iter={max_iter}"
        )
        
        # Modern API
        model = AffinityPropagation(
            preference=preference,
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            random_state=random_state,
            verbose=int(verbose),
            **kwargs
        )
        
        labels = model.fit_predict(X)
        
        # Analyze results
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_exemplars = len(model.cluster_centers_indices_)
        
        metrics = {
            "n_clusters": int(n_clusters),
            "n_exemplars": int(n_exemplars),
            "n_iter": int(model.n_iter_),
            "preference_used": float(model.preference) if hasattr(model, 'preference') else preference,
        }
        
        # Calculate evaluation metrics
        if n_clusters > 1:
            try:
                metrics["silhouette"] = float(silhouette_score(X, labels))
                metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
                metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
                logger.debug("Affinity Propagation metrics calculated")
            except Exception as e:
                logger.warning(f"Could not calculate metrics: {e}")
        else:
            logger.warning("Only 1 cluster found")
        
        logger.info(
            f"Affinity Propagation completed: {n_clusters} clusters found "
            f"(exemplars: {n_exemplars}, iterations: {metrics['n_iter']})"
        )
        
        return model, labels, metrics
    
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"Affinity Propagation failed: {str(e)}", exc_info=True)
        raise ClusteringError(f"Affinity Propagation failed: {str(e)}")


def tune_affinity_propagation_preference(
    X: np.ndarray,
    preference_range: np.ndarray = None,
    damping: float = 0.9,
    random_state: int = 42
) -> Dict[float, Dict]:
    """
    Tune preference parameter to find desired number of clusters.
    
    Args:
        X: Input feature matrix
        preference_range: Array of preference values to test
        damping: Damping factor
        random_state: Random state
    
    Returns:
        Dictionary mapping preference -> metrics
    
    Example:
        >>> prefs = np.linspace(-50, 0, 10)
        >>> results = tune_affinity_propagation_preference(X, preference_range=prefs)
        >>> # Find preference that gives desired cluster count
        >>> for pref, metrics in results.items():
        ...     if metrics['n_clusters'] == 4:
        ...         print(f"Use preference={pref}")
    """
    if preference_range is None:
        # Generate preference range
        similarity_median = np.median(
            np.corrcoef(X.T) if X.shape[1] > 1 else [[1]]
        )
        preference_range = np.linspace(similarity_median - 50, similarity_median, 15)
    
    logger.info(f"Tuning preference parameter: {len(preference_range)} values")
    
    results = {}
    for preference in preference_range:
        try:
            _, labels, metrics = run_affinity_propagation(
                X, preference=preference, damping=damping, random_state=random_state
            )
            results[float(preference)] = metrics
            logger.debug(
                f"Preference={preference:.2f}: {metrics['n_clusters']} clusters"
            )
        except Exception as e:
            logger.warning(f"Failed at preference={preference}: {e}")
            results[float(preference)] = {"error": str(e)}
    
    return results
