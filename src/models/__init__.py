"""Clustering models package - modular architecture."""

# Individual clustering algorithms
from src.models.kmeans import run_kmeans, find_optimal_clusters
from src.models.hierarchical import run_hierarchical, compare_linkage_methods
from src.models.dbscan import run_dbscan, find_optimal_eps
from src.models.affinity_propagation import run_affinity_propagation, tune_affinity_propagation_preference
from src.models.meanshift import run_meanshift, tune_meanshift_bandwidth
from src.models.optics import run_optics, tune_optics_xi

# Error handling
from src.utils.validator import ClusteringError

__all__ = [
    # K-Means
    "run_kmeans",
    "find_optimal_clusters",
    
    # Hierarchical
    "run_hierarchical",
    "compare_linkage_methods",
    
    # DBSCAN
    "run_dbscan",
    "find_optimal_eps",
    
    # Affinity Propagation
    "run_affinity_propagation",
    "tune_affinity_propagation_preference",
    
    # Mean Shift
    "run_meanshift",
    "tune_meanshift_bandwidth",
    
    # OPTICS
    "run_optics",
    "tune_optics_xi",
    
    # Exceptions
    "ClusteringError",
]
