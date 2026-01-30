"""
Clustering module - imports all algorithms from separate modules.

This module provides backwards compatibility with the original clustering.py
while using the new modular architecture with individual algorithm files:

- kmeans.py: K-Means clustering
- hierarchical.py: Hierarchical Agglomerative Clustering  
- dbscan.py: DBSCAN clustering
- affinity_propagation.py: Affinity Propagation
- meanshift.py: Mean Shift
- optics.py: OPTICS

All algorithms have been updated to use current scikit-learn APIs (2025+).
"""

# Import all public functions from individual modules
from src.models.kmeans import run_kmeans, find_optimal_clusters
from src.models.hierarchical import run_hierarchical, compare_linkage_methods
from src.models.dbscan import run_dbscan, find_optimal_eps
from src.models.affinity_propagation import run_affinity_propagation, tune_affinity_propagation_preference
from src.models.meanshift import run_meanshift, tune_meanshift_bandwidth
from src.models.optics import run_optics, tune_optics_xi
from src.utils.validator import ClusteringError

# For backwards compatibility - algorithm dictionary
ALGORITHMS = {
    "kmeans": {"func": run_kmeans, "name": "K-Means"},
    "hierarchical": {"func": run_hierarchical, "name": "Hierarchical"},
    "dbscan": {"func": run_dbscan, "name": "DBSCAN"},
    "affinity_propagation": {"func": run_affinity_propagation, "name": "Affinity Propagation"},
    "meanshift": {"func": run_meanshift, "name": "Mean Shift"},
    "optics": {"func": run_optics, "name": "OPTICS"},
}

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
    
    # For backwards compatibility
    "ALGORITHMS",
]
