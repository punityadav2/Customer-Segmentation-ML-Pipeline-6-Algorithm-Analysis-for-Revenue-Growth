"""
End-to-end clustering pipeline module.

Orchestrates the complete workflow: data loading, preprocessing,
clustering, and evaluation. Provides a clean interface for running
the entire customer segmentation pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

from src.data.loader import load_data
from src.data.validator import validate_and_prepare_data, DataValidationError
from src.features.preprocess import Preprocessor, PreprocessingError
from src.models.clustering import (
    run_kmeans, run_hierarchical, run_dbscan, run_affinity_propagation,
    run_meanshift, run_optics, ClusteringError
)
from src.evaluation.metrics import evaluate_clustering, MetricsError
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline failures."""
    pass


class ClusteringPipeline:
    """
    End-to-end clustering pipeline.
    
    Manages the complete workflow from data loading to evaluation with
    proper error handling, logging, and configuration management.
    
    Attributes:
        config: Configuration dictionary
        preprocessor: Preprocessor instance for data transformation
        results: Dictionary storing algorithm results
    
    Example:
        >>> pipeline = ClusteringPipeline(config_path='config/config.yaml')
        >>> pipeline.load_data()
        >>> X = pipeline.preprocess()
        >>> pipeline.run_all_algorithms(X)
        >>> comparison = pipeline.get_comparison()
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        
        Raises:
            PipelineError: If configuration cannot be loaded
        """
        try:
            logger.info(f"Initializing pipeline with config: {config_path}")
            self.config = load_config(config_path)
            self.df = None
            self.X = None
            self.preprocessor = None
            self.results = {}
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {str(e)}")
            raise PipelineError(f"Pipeline init failed: {str(e)}")
    
    def load_data(self, raw_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            raw_path: Path to CSV file. If None, uses config['data']['raw_path']
        
        Returns:
            Loaded DataFrame
        
        Raises:
            PipelineError: If data loading fails
        
        Example:
            >>> pipeline = ClusteringPipeline()
            >>> df = pipeline.load_data()
            >>> print(f"Loaded {len(df)} samples")
        """
        try:
            if raw_path is None:
                raw_path = self.config['data']['raw_path']
            
            logger.info(f"Loading data from {raw_path}")
            self.df = load_data(raw_path)
            logger.info(f"Data loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")
            return self.df
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise PipelineError(f"Data loading failed: {str(e)}")
    
    def preprocess(self) -> np.ndarray:
        """
        Preprocess loaded data.
        
        Performs: renaming, feature dropping, categorical encoding, scaling
        
        Returns:
            Scaled feature matrix
        
        Raises:
            PipelineError: If preprocessing fails
        
        Example:
            >>> pipeline = ClusteringPipeline()
            >>> pipeline.load_data()
            >>> X = pipeline.preprocess()
            >>> print(f"Preprocessed shape: {X.shape}")
        """
        try:
            if self.df is None:
                raise PipelineError("No data loaded. Call load_data() first.")
            
            logger.info("Starting preprocessing")
            
            # Initialize preprocessor
            rename_map = self.config['columns'].get('rename', {})
            features_to_drop = self.config['columns'].get('drop', [])
            
            self.preprocessor = Preprocessor(
                rename_map=rename_map,
                features_to_drop=features_to_drop
            )
            
            # Fit and transform
            df_scaled, scaler = self.preprocessor.fit_transform(self.df)
            
            # Get numeric columns for clustering
            numeric_cols = self.config['columns'].get('numeric', [])
            if not numeric_cols:
                numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            
            self.X = df_scaled[numeric_cols].values
            
            logger.info(f"Preprocessing completed: {self.X.shape[0]} samples, {self.X.shape[1]} features")
            return self.X
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise PipelineError(f"Preprocessing failed: {str(e)}")
    
    def run_algorithm(
        self,
        algorithm: str,
        **params
    ) -> Tuple[Any, np.ndarray, Dict[str, float]]:
        """
        Run a single clustering algorithm.
        
        Args:
            algorithm: Algorithm name ('kmeans', 'hierarchical', 'dbscan', etc.)
            **params: Algorithm-specific parameters
        
        Returns:
            Tuple of (model, labels, metrics)
        
        Raises:
            PipelineError: If algorithm not supported or execution fails
        
        Example:
            >>> pipeline = ClusteringPipeline()
            >>> pipeline.load_data()
            >>> X = pipeline.preprocess()
            >>> model, labels, metrics = pipeline.run_algorithm('kmeans', n_clusters=3)
        """
        if self.X is None:
            raise PipelineError("No preprocessed data. Call preprocess() first.")
        
        try:
            logger.info(f"Running {algorithm} with params: {params}")
            
            if algorithm == 'kmeans':
                model, labels, metrics = run_kmeans(self.X, **params)
            elif algorithm == 'hierarchical':
                model, labels, metrics = run_hierarchical(self.X, **params)
            elif algorithm == 'dbscan':
                model, labels, metrics = run_dbscan(self.X, **params)
            elif algorithm == 'affinity_propagation':
                model, labels, metrics = run_affinity_propagation(self.X, **params)
            elif algorithm == 'meanshift':
                model, labels, metrics = run_meanshift(self.X, **params)
            elif algorithm == 'optics':
                model, labels, metrics = run_optics(self.X, **params)
            else:
                raise PipelineError(f"Unknown algorithm: {algorithm}")
            
            # Store metrics
            self.results[algorithm] = metrics
            logger.info(f"{algorithm} completed successfully")
            
            return model, labels, metrics
            
        except ClusteringError:
            raise
        except Exception as e:
            logger.error(f"{algorithm} failed: {str(e)}")
            raise PipelineError(f"{algorithm} failed: {str(e)}")
    
    def run_all_algorithms(self) -> Dict[str, Dict[str, float]]:
        """
        Run all algorithms configured in config.yaml.
        
        Returns:
            Dictionary mapping algorithm names to their metrics
        
        Example:
            >>> pipeline = ClusteringPipeline()
            >>> pipeline.load_data()
            >>> X = pipeline.preprocess()
            >>> all_results = pipeline.run_all_algorithms()
            >>> for algo, metrics in all_results.items():
            ...     print(f"{algo}: {metrics['Silhouette Score']:.3f}")
        """
        try:
            logger.info("Running all configured algorithms")
            algorithms = self.config.get('models', {})
            
            for algo_name, algo_params in algorithms.items():
                try:
                    self.run_algorithm(algo_name, **algo_params)
                except Exception as e:
                    logger.warning(f"Skipping {algo_name}: {str(e)}")
            
            logger.info(f"Completed {len(self.results)} algorithms")
            return self.results
            
        except Exception as e:
            logger.error(f"Failed to run all algorithms: {str(e)}")
            raise PipelineError(f"Batch execution failed: {str(e)}")
    
    def get_comparison(self) -> pd.DataFrame:
        """
        Get comparison DataFrame of all algorithm results.
        
        Returns:
            DataFrame with algorithms as rows and metrics as columns
        
        Example:
            >>> comparison = pipeline.get_comparison()
            >>> print(comparison.to_string())
        """
        if not self.results:
            logger.warning("No results to compare")
            return pd.DataFrame()
        
        return pd.DataFrame(self.results).T
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete pipeline: load → preprocess → cluster → evaluate.
        
        Returns:
            Dictionary with 'data', 'preprocessor', and 'results' keys
        
        Example:
            >>> result = pipeline.run_complete_pipeline()
            >>> print(result['results'])
        """
        try:
            logger.info("Starting complete pipeline")
            
            # Step 1: Load
            self.load_data()
            
            # Step 2: Preprocess
            self.preprocess()
            
            # Step 3: Cluster (all algorithms)
            self.run_all_algorithms()
            
            logger.info("Pipeline completed successfully")
            
            return {
                'data': self.df,
                'X': self.X,
                'preprocessor': self.preprocessor,
                'results': self.results
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise PipelineError(f"Pipeline failed: {str(e)}")
