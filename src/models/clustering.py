"""Clustering algorithms module with error handling and logging."""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, 
    AffinityPropagation, MeanShift, OPTICS
)
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.evaluation.metrics import evaluate_clustering

logger = logging.getLogger(__name__)


class ClusteringError(Exception):
    """Custom exception for clustering errors."""
    pass


def validate_clustering_input(
    X: np.ndarray,
    algorithm: str,
    min_samples: int = 2
) -> None:
    """
    Validate input data for clustering.
    
    Args:
        X: Input feature matrix (n_samples, n_features)
        algorithm: Name of clustering algorithm
        min_samples: Minimum number of samples required
    
    Raises:
        ClusteringError: If validation fails
    """
    if X is None:
        raise ClusteringError("Input data X is None")
    
    if not isinstance(X, np.ndarray):
        raise ClusteringError(f"X must be numpy array, got {type(X)}")
    
    if X.shape[0] < min_samples:
        raise ClusteringError(
            f"Need at least {min_samples} samples, got {X.shape[0]}"
        )
    
    if X.shape[1] == 0:
        raise ClusteringError("Input data has 0 features")
    
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        raise ClusteringError(f"Input data contains {nan_count} NaN values")
    
    if np.isinf(X).any():
        raise ClusteringError("Input data contains infinite values")
    
    logger.debug(f"Validation passed for {algorithm}: {X.shape}")


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    **kwargs
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Run KMeans clustering with error handling.
    
    Args:
        X: Input feature matrix (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        **kwargs: Additional arguments for KMeans
    
    Returns:
        Tuple of (model, labels, metrics)
    
    Raises:
        ClusteringError: If clustering fails
    
    Example:
        >>> X = np.random.randn(100, 3)
        >>> model, labels, metrics = run_kmeans(X, n_clusters=3)
        >>> print(f"Silhouette: {metrics['silhouette']:.3f}")
    """
    try:
        validate_clustering_input(X, 'kmeans')
        
        if n_clusters <= 0:
            raise ClusteringError(f"n_clusters must be positive, got {n_clusters}")
        if n_clusters > X.shape[0]:
            raise ClusteringError(
                f"n_clusters ({n_clusters}) > n_samples ({X.shape[0]})"
            )
        
        logger.info(f"Running KMeans with n_clusters={n_clusters}")
        
        model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init='auto',
            random_state=random_state,
            **kwargs
        )
        labels = model.fit_predict(X)
        
        # Calculate metrics
        metrics = {'inertia': float(model.inertia_)}
        try:
            metrics['silhouette'] = float(silhouette_score(X, labels))
            metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
        except Exception as e:
            logger.warning(f"Could not calculate all metrics: {e}")
        
        logger.info(f"KMeans completed: {n_clusters} clusters found")
        return model, labels, metrics
        
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"KMeans failed: {str(e)}")
        raise ClusteringError(f"KMeans failed: {str(e)}")

def run_hierarchical(
    X: np.ndarray,
    n_clusters: int,
    linkage: str = 'ward'
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Run Hierarchical (Agglomerative) clustering.
    
    Args:
        X: Input feature matrix
        n_clusters: Number of clusters
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
    
    Returns:
        Tuple of (model, labels, metrics)
    
    Raises:
        ClusteringError: If clustering fails
    """
    try:
        validate_clustering_input(X, 'hierarchical')
        
        if n_clusters <= 0 or n_clusters > X.shape[0]:
            raise ClusteringError(f"Invalid n_clusters: {n_clusters}")
        
        logger.info(f"Running Hierarchical clustering with n_clusters={n_clusters}")
        
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X)
        
        metrics = {}
        try:
            metrics['silhouette'] = float(silhouette_score(X, labels))
            metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
        except Exception as e:
            logger.warning(f"Could not calculate metrics: {e}")
        
        logger.info(f"Hierarchical completed: {n_clusters} clusters found")
        return model, labels, metrics
        
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"Hierarchical clustering failed: {str(e)}")
        raise ClusteringError(f"Hierarchical failed: {str(e)}")


def run_dbscan(
    X: np.ndarray,
    eps: float,
    min_samples: int = 5
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Run DBSCAN clustering.
    
    Args:
        X: Input feature matrix
        eps: Maximum distance between samples
        min_samples: Minimum samples in neighborhood
    
    Returns:
        Tuple of (model, labels, metrics)
    
    Raises:
        ClusteringError: If clustering fails
    """
    try:
        validate_clustering_input(X, 'dbscan', min_samples=2)
        
        if eps <= 0:
            raise ClusteringError(f"eps must be positive, got {eps}")
        
        logger.info(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        metrics = {'n_clusters': n_clusters, 'n_noise_points': int(n_noise)}
        
        if n_clusters > 1:
            X_clean = X[labels != -1]
            labels_clean = labels[labels != -1]
            try:
                metrics['silhouette'] = float(silhouette_score(X_clean, labels_clean))
                metrics['davies_bouldin'] = float(davies_bouldin_score(X_clean, labels_clean))
            except Exception as e:
                logger.warning(f"Could not calculate metrics: {e}")
        
        logger.info(f"DBSCAN completed: {n_clusters} clusters, {n_noise} noise points")
        return model, labels, metrics
        
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"DBSCAN failed: {str(e)}")
        raise ClusteringError(f"DBSCAN failed: {str(e)}")


def run_affinity_propagation(
    X: np.ndarray,
    preference: Optional[float] = None,
    damping: float = 0.9,
    random_state: int = 42
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Run Affinity Propagation clustering.
    
    Args:
        X: Input feature matrix
        preference: Preference value for cluster selection
        damping: Damping factor
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (model, labels, metrics)
    
    Raises:
        ClusteringError: If clustering fails
    """
    try:
        validate_clustering_input(X, 'affinity_propagation')
        
        logger.info("Running Affinity Propagation clustering")
        
        model = AffinityPropagation(
            preference=preference,
            damping=damping,
            random_state=random_state
        )
        labels = model.fit_predict(X)
        
        n_clusters = len(set(labels))
        metrics = {'n_clusters': n_clusters}
        
        if n_clusters > 1:
            try:
                metrics['silhouette'] = float(silhouette_score(X, labels))
                metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
            except Exception as e:
                logger.warning(f"Could not calculate metrics: {e}")
        
        logger.info(f"Affinity Propagation completed: {n_clusters} clusters found")
        return model, labels, metrics
        
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"Affinity Propagation failed: {str(e)}")
        raise ClusteringError(f"Affinity Propagation failed: {str(e)}")


def run_meanshift(
    X: np.ndarray,
    bandwidth: Optional[float] = None
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Run Mean Shift clustering.
    
    Args:
        X: Input feature matrix
        bandwidth: Bandwidth for kernel
    
    Returns:
        Tuple of (model, labels, metrics)
    
    Raises:
        ClusteringError: If clustering fails
    """
    try:
        validate_clustering_input(X, 'meanshift')
        
        logger.info("Running Mean Shift clustering")
        
        model = MeanShift(bandwidth=bandwidth)
        labels = model.fit_predict(X)
        
        n_clusters = len(set(labels))
        metrics = {'n_clusters': n_clusters}
        
        if n_clusters > 1:
            try:
                metrics['silhouette'] = float(silhouette_score(X, labels))
                metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
            except Exception as e:
                logger.warning(f"Could not calculate metrics: {e}")
        
        logger.info(f"Mean Shift completed: {n_clusters} clusters found")
        return model, labels, metrics
        
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"Mean Shift failed: {str(e)}")
        raise ClusteringError(f"Mean Shift failed: {str(e)}")


def run_optics(
    X: np.ndarray,
    min_samples: int = 5,
    eps: Optional[float] = None
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Run OPTICS clustering.
    
    Args:
        X: Input feature matrix
        min_samples: Minimum samples in neighborhood
        eps: Maximum distance (optional)
    
    Returns:
        Tuple of (model, labels, metrics)
    
    Raises:
        ClusteringError: If clustering fails
    """
    try:
        validate_clustering_input(X, 'optics', min_samples=min_samples)
        
        logger.info(f"Running OPTICS with min_samples={min_samples}")
        
        model = OPTICS(min_samples=min_samples, eps=eps)
        labels = model.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        metrics = {'n_clusters': n_clusters, 'n_noise_points': int(n_noise)}
        
        if n_clusters > 1:
            X_clean = X[labels != -1]
            labels_clean = labels[labels != -1]
            try:
                metrics['silhouette'] = float(silhouette_score(X_clean, labels_clean))
                metrics['davies_bouldin'] = float(davies_bouldin_score(X_clean, labels_clean))
            except Exception as e:
                logger.warning(f"Could not calculate metrics: {e}")
        
        logger.info(f"OPTICS completed: {n_clusters} clusters, {n_noise} noise points")
        return model, labels, metrics
        
    except ClusteringError:
        raise
    except Exception as e:
        logger.error(f"OPTICS failed: {str(e)}")
        raise ClusteringError(f"OPTICS failed: {str(e)}")

def tune_dbscan(X, eps_range, min_samples_range):
    results = []
    from itertools import product
    for eps, ms in product(eps_range, min_samples_range):
        _, labels = run_dbscan(X, eps, ms)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        metrics = evaluate_clustering(X[labels != -1], labels[labels != -1]) if n_clusters > 1 else {}
        metrics.update({'eps': eps, 'min_samples': ms, 'n_clusters': n_clusters})
        results.append(metrics)
    return pd.DataFrame(results)

# Add similar tuning functions for others if needed
def tune_affinity_propagation(X, preference_range):
    results = []
    for p in preference_range:
        _, labels = run_affinity_propagation(X, p)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        metrics = {}
        if n_clusters > 1:
            metrics = evaluate_clustering(X, labels)
        
        metrics.update({'preference': p, 'n_clusters': n_clusters})
        results.append(metrics)
    return pd.DataFrame(results)

def tune_meanshift(X, quantile_range, n_samples=500):
    from sklearn.cluster import estimate_bandwidth
    results = []
    for q in quantile_range:
        bw = estimate_bandwidth(X, quantile=q, n_samples=n_samples)
        _, labels = run_meanshift(X, bw)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        metrics = {}
        if n_clusters > 1:
            metrics = evaluate_clustering(X, labels)
            
        metrics.update({'quantile': q, 'bandwidth': bw, 'n_clusters': n_clusters})
        results.append(metrics)
    return pd.DataFrame(results)

def tune_optics(X, min_samples_range, eps_range):
    results = []
    from itertools import product
    for ms, eps in product(min_samples_range, eps_range):
        # OPTICS eps is really max_eps, but acts as a bound
        try:
            _, labels = run_optics(X, ms, eps)
            unique_labels = np.unique(labels)
            # Remove noise
            clean_labels = labels[labels != -1]
            n_clusters = len(np.unique(clean_labels)) # only valid clusters
            
            metrics = {}
            if n_clusters > 1:
                metrics = evaluate_clustering(X[labels != -1], clean_labels)
            
            metrics.update({'min_samples': ms, 'eps': eps, 'n_clusters': n_clusters})
            results.append(metrics)
        except Exception as e:
            continue
            
    return pd.DataFrame(results)

