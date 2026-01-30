"""
Data loading and saving utilities.

Provides functions for reading raw data, saving processed results,
and managing data file I/O with error handling.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Custom exception for data loading failures."""
    pass


def load_data(
    path: str,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        path: Path to CSV file
        **kwargs: Additional arguments to pass to pd.read_csv()
    
    Returns:
        Loaded DataFrame
    
    Raises:
        DataLoaderError: If file not found or read fails
    
    Example:
        >>> df = load_data('data/raw/Mall_Customers.csv')
        >>> print(f"Loaded {len(df)} samples")
    """
    try:
        if not os.path.exists(path):
            raise DataLoaderError(f"File not found: {path}")
        
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
        
    except DataLoaderError:
        raise
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise DataLoaderError(f"Data loading failed: {str(e)}")


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV (wrapper for compatibility).
    
    Args:
        path: Path to CSV file
    
    Returns:
        Loaded DataFrame
    
    Raises:
        DataLoaderError: If file not found or read fails
    
    Example:
        >>> df = load_raw_data('data/raw/Mall_Customers.csv')
    """
    return load_data(path)


def save_processed_data(
    df: pd.DataFrame,
    path: str,
    index: bool = False,
    create_dirs: bool = True
) -> None:
    """
    Save processed dataset to CSV file.
    
    Args:
        df: DataFrame to save
        path: Output file path
        index: Whether to save index column
        create_dirs: Whether to create directories if they don't exist
    
    Raises:
        DataLoaderError: If save fails
    
    Example:
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> save_processed_data(df, 'data/processed/output.csv')
    """
    try:
        if create_dirs:
            output_dir = os.path.dirname(path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created directory: {output_dir}")
        
        df.to_csv(path, index=index)
        logger.info(f"Data saved to {path} ({len(df)} rows)")
        
    except Exception as e:
        logger.error(f"Failed to save data to {path}: {str(e)}")
        raise DataLoaderError(f"Data saving failed: {str(e)}")


def load_cluster_results(
    path: str,
    **kwargs
) -> pd.DataFrame:
    """
    Load clustering results from CSV.
    
    Args:
        path: Path to results CSV
        **kwargs: Additional arguments to pd.read_csv()
    
    Returns:
        Results DataFrame
    
    Raises:
        DataLoaderError: If file not found or read fails
    
    Example:
        >>> results = load_cluster_results('reports/cluster_results.csv')
    """
    try:
        logger.info(f"Loading cluster results from {path}")
        df = load_data(path, **kwargs)
        logger.info(f"Results loaded: {df.shape}")
        return df
    except DataLoaderError:
        raise
    except Exception as e:
        logger.error(f"Failed to load results: {str(e)}")
        raise DataLoaderError(f"Results loading failed: {str(e)}")


def save_cluster_results(
    df: pd.DataFrame,
    path: str,
    **kwargs
) -> None:
    """
    Save clustering results to CSV.
    
    Args:
        df: Results DataFrame
        path: Output file path
        **kwargs: Additional arguments to df.to_csv()
    
    Example:
        >>> results = pd.DataFrame({'algorithm': ['kmeans'], 'score': [0.65]})
        >>> save_cluster_results(results, 'reports/results.csv')
    """
    try:
        logger.info(f"Saving cluster results to {path}")
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        df.to_csv(path, index=kwargs.pop('index', False), **kwargs)
        logger.info(f"Results saved: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise DataLoaderError(f"Results saving failed: {str(e)}")
