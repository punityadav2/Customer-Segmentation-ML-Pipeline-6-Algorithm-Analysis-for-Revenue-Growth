"""
Unit tests for the clustering pipeline.

Tests core functionality of data loading, preprocessing, clustering,
and evaluation modules.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_data, DataLoaderError
from src.data.validator import validate_features, validate_dataframe, validate_and_prepare_data, DataValidationError
from src.features.preprocess import Preprocessor, PreprocessingError
from src.models.clustering import (
    run_kmeans, run_hierarchical, run_dbscan, ClusteringError, validate_clustering_input
)
from src.evaluation.metrics import evaluate_clustering, MetricsError


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_nonexistent_file(self):
        """Test error handling for missing files."""
        with pytest.raises(DataLoaderError):
            load_data("nonexistent_file.csv")
    
    def test_load_existing_file(self):
        """Test loading valid CSV file."""
        # Create test file
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        test_file = 'test_data.csv'
        test_data.to_csv(test_file, index=False)
        
        try:
            df = load_data(test_file)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert list(df.columns) == ['col1', 'col2']
        finally:
            os.remove(test_file)


class TestValidator:
    """Test data validation functionality."""
    
    def test_validate_features_valid(self):
        """Test validation of valid feature matrix."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        is_valid, errors = validate_features(X)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_features_nan(self):
        """Test detection of NaN values."""
        X = np.array([[1, np.nan], [3, 4], [5, 6]])
        is_valid, errors = validate_features(X)
        assert not is_valid
        assert any('NaN' in e for e in errors)
    
    def test_validate_features_inf(self):
        """Test detection of infinite values."""
        X = np.array([[1, 2], [3, np.inf], [5, 6]])
        is_valid, errors = validate_features(X)
        assert not is_valid
        assert any('Infinite' in e for e in errors)
    
    def test_validate_dataframe_valid(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        is_valid, errors = validate_dataframe(df)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_dataframe_empty(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        is_valid, errors = validate_dataframe(df)
        assert not is_valid
        assert len(errors) > 0


class TestPreprocessor:
    """Test data preprocessing functionality."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'ID': [1, 2, 3],
            'Gender': ['M', 'F', 'M'],
            'Age': [25, 30, 35],
            'Income': [50000, 60000, 70000]
        })
    
    def test_preprocessor_init(self, sample_df):
        """Test preprocessor initialization."""
        prep = Preprocessor(
            rename_map={'ID': 'customer_id'},
            features_to_drop=['customer_id']
        )
        assert prep.rename_map == {'ID': 'customer_id'}
        assert prep.features_to_drop == ['customer_id']
    
    def test_initial_cleaning(self, sample_df):
        """Test initial column renaming."""
        prep = Preprocessor(rename_map={'ID': 'customer_id'})
        df_clean = prep.initial_cleaning(sample_df)
        assert 'customer_id' in df_clean.columns
        assert 'ID' not in df_clean.columns
    
    def test_fit_transform(self, sample_df):
        """Test fit_transform pipeline."""
        prep = Preprocessor(
            rename_map={'ID': 'customer_id'},
            features_to_drop=['customer_id', 'Gender']
        )
        df_scaled, scaler = prep.fit_transform(sample_df)
        
        assert isinstance(df_scaled, pd.DataFrame)
        assert len(df_scaled) == 3
        assert 'customer_id' not in df_scaled.columns
        assert 'Gender' not in df_scaled.columns
    
    def test_preprocessor_transform(self, sample_df):
        """Test transform on new data."""
        prep = Preprocessor(
            rename_map={'ID': 'customer_id'},
            features_to_drop=['customer_id', 'Gender']
        )
        df_scaled, _ = prep.fit_transform(sample_df)
        
        # Create new data with same structure
        new_df = pd.DataFrame({
            'ID': [4, 5],
            'Gender': ['F', 'M'],
            'Age': [40, 45],
            'Income': [80000, 90000]
        })
        
        df_transformed = prep.transform(new_df)
        assert len(df_transformed) == 2


class TestClustering:
    """Test clustering algorithms."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for clustering."""
        # 3 clusters with clear separation
        X1 = np.random.normal([0, 0], 0.5, (20, 2))
        X2 = np.random.normal([5, 5], 0.5, (20, 2))
        X3 = np.random.normal([10, 0], 0.5, (20, 2))
        return np.vstack([X1, X2, X3])
    
    def test_kmeans_basic(self, sample_data):
        """Test K-Means clustering."""
        model, labels, metrics = run_kmeans(sample_data, n_clusters=3)
        assert labels.shape == (60,)
        assert len(np.unique(labels)) == 3
        assert 'silhouette' in metrics
    
    def test_hierarchical_basic(self, sample_data):
        """Test Hierarchical clustering."""
        model, labels, metrics = run_hierarchical(sample_data, n_clusters=3)
        assert labels.shape == (60,)
        assert len(np.unique(labels)) == 3
    
    def test_dbscan_basic(self, sample_data):
        """Test DBSCAN clustering."""
        model, labels, metrics = run_dbscan(sample_data, eps=1.5, min_samples=5)
        assert labels.shape == (60,)
    
    def test_clustering_invalid_input(self, sample_data):
        """Test error handling for invalid input."""
        with pytest.raises(ClusteringError):
            run_kmeans(sample_data, n_clusters=-1)


class TestMetrics:
    """Test evaluation metrics."""
    
    @pytest.fixture
    def sample_clustering(self):
        """Create sample clustering data."""
        X = np.array([[1, 2], [2, 1], [8, 9], [9, 8]])
        labels = np.array([0, 0, 1, 1])
        return X, labels
    
    def test_evaluate_clustering(self, sample_clustering):
        """Test clustering evaluation."""
        X, labels = sample_clustering
        metrics = evaluate_clustering(X, labels)
        
        assert 'Clusters' in metrics
        assert 'Silhouette Score' in metrics
        assert metrics['Clusters'] == 2
    
    def test_evaluate_single_cluster(self):
        """Test evaluation with single cluster."""
        X = np.array([[1, 2], [2, 1], [3, 2]])
        labels = np.array([0, 0, 0])
        metrics = evaluate_clustering(X, labels)
        
        assert metrics['Clusters'] == 1
        assert np.isnan(metrics['Silhouette Score'])


def test_integration():
    """Integration test for complete pipeline."""
    # Create test data
    df = pd.DataFrame({
        'ID': range(1, 31),
        'Category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
        'Value1': np.random.randn(30),
        'Value2': np.random.randn(30)
    })
    
    # Test preprocessing
    prep = Preprocessor(
        rename_map={'ID': 'id'},
        features_to_drop=['id', 'Category']
    )
    df_scaled, scaler = prep.fit_transform(df)
    X = df_scaled.values
    
    # Test clustering
    model, labels, metrics = run_kmeans(X, n_clusters=3)
    
    # Test evaluation
    metrics_eval = evaluate_clustering(X, labels)
    
    assert len(labels) == 30
    assert 'Silhouette Score' in metrics_eval
    assert metrics_eval['Clusters'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
