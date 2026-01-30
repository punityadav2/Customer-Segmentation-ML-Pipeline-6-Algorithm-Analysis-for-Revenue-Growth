"""
Feature preprocessing module for customer segmentation.

Handles data cleaning, categorical encoding, and feature scaling
using StandardScaler for normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Custom exception for preprocessing failures."""
    pass


class Preprocessor:
    """
    Comprehensive data preprocessing pipeline for clustering.
    
    Handles column renaming, dropping unnecessary features, categorical encoding,
    and numerical scaling.
    
    Attributes:
        rename_map: Dictionary of column name mappings
        features_to_drop: List of features to drop
        scaler: StandardScaler instance for normalization
        encoders: Dictionary of label encoders for categorical features
    
    Example:
        >>> df = pd.DataFrame({'ID': [1, 2], 'Gender': ['M', 'F'], 'Age': [25, 30]})
        >>> prep = Preprocessor(
        ...     rename_map={'ID': 'customer_id'},
        ...     features_to_drop=['customer_id']
        ... )
        >>> df_processed, scaler = prep.fit_transform(df)
    """
    
    def __init__(
        self,
        rename_map: Optional[Dict[str, str]] = None,
        features_to_drop: Optional[List[str]] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            rename_map: Dictionary mapping old column names to new ones
            features_to_drop: List of column names to drop
        """
        self.rename_map = rename_map or {}
        self.features_to_drop = features_to_drop or []
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        logger.info(f"Preprocessor initialized with {len(self.rename_map)} renames")

    def initial_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data cleaning and renaming.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame with renamed columns
        """
        if df is None or df.empty:
            raise PreprocessingError("Input DataFrame is None or empty")
        
        df_clean = df.copy()
        
        if self.rename_map:
            df_clean.rename(columns=self.rename_map, inplace=True)
            logger.info(f"Renamed {len(self.rename_map)} columns")
        
        return df_clean

    def fit_transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Fit preprocessor and transform data in one step.
        
        Performs: renaming → feature dropping → categorical encoding → scaling
        
        Args:
            df: Input DataFrame with raw data
        
        Returns:
            Tuple of (scaled_dataframe, fitted_scaler)
        
        Raises:
            PreprocessingError: If preprocessing fails
        
        Example:
            >>> df = pd.DataFrame({
            ...     'CustomerID': [1, 2, 3],
            ...     'Gender': ['M', 'F', 'M'],
            ...     'Age': [25, 30, 35]
            ... })
            >>> preprocessor = Preprocessor(
            ...     rename_map={'CustomerID': 'ID'},
            ...     features_to_drop=['ID']
            ... )
            >>> df_scaled, scaler = preprocessor.fit_transform(df)
        """
        try:
            logger.info(f"Starting fit_transform with {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Step 1: Initial cleaning
            df_processed = self.initial_cleaning(df)
            
            # Step 2: Drop features
            if self.features_to_drop:
                cols_to_drop = [col for col in self.features_to_drop if col in df_processed.columns]
                if cols_to_drop:
                    df_processed.drop(columns=cols_to_drop, inplace=True)
                    logger.info(f"Dropped {len(cols_to_drop)} features")
            
            # Step 3: Identify column types
            categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
            
            self.categorical_columns = categorical_cols
            self.numeric_columns = numeric_cols
            
            logger.info(f"Found {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns")
            
            # Step 4: Encode categorical variables
            for col in categorical_cols:
                try:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col])
                    self.encoders[col] = le
                    logger.info(f"Encoded column '{col}' with {len(le.classes_)} classes")
                except Exception as e:
                    raise PreprocessingError(f"Encoding failed for column '{col}': {str(e)}")
            
            # Step 5: Scale numerical features
            try:
                scaled_data = self.scaler.fit_transform(df_processed)
                df_scaled = pd.DataFrame(scaled_data, columns=df_processed.columns)
                logger.info("Scaling completed successfully")
            except Exception as e:
                raise PreprocessingError(f"Scaling failed: {str(e)}")
            
            return df_scaled, self.scaler
            
        except PreprocessingError:
            raise
        except Exception as e:
            logger.error(f"fit_transform failed: {str(e)}")
            raise PreprocessingError(f"fit_transform failed: {str(e)}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: New DataFrame to transform
        
        Returns:
            Scaled DataFrame
        
        Raises:
            PreprocessingError: If transformation fails
        
        Example:
            >>> new_data = pd.DataFrame({
            ...     'Gender': ['F', 'M'],
            ...     'Age': [28, 32]
            ... })
            >>> df_transformed = preprocessor.transform(new_data)
        """
        try:
            if not hasattr(self.scaler, 'mean_'):
                raise PreprocessingError("Preprocessor not fitted. Call fit_transform first.")
            
            logger.info(f"Transforming {df.shape[0]} samples")
            
            df_processed = df.copy()
            
            # Drop features
            if self.features_to_drop:
                cols_to_drop = [col for col in self.features_to_drop if col in df_processed.columns]
                if cols_to_drop:
                    df_processed.drop(columns=cols_to_drop, inplace=True)
            
            # Transform categorical columns
            for col, le in self.encoders.items():
                if col in df_processed.columns:
                    try:
                        df_processed[col] = le.transform(df_processed[col])
                    except Exception as e:
                        logger.warning(f"Could not encode column '{col}': {str(e)}")
            
            # Scale
            scaled_data = self.scaler.transform(df_processed)
            df_transformed = pd.DataFrame(scaled_data, columns=df_processed.columns)
            
            logger.info("Transformation completed successfully")
            return df_transformed
            
        except PreprocessingError:
            raise
        except Exception as e:
            logger.error(f"Transform failed: {str(e)}")
            raise PreprocessingError(f"Transform failed: {str(e)}")

    def get_feature_info(self) -> Dict[str, List[str]]:
        """
        Get information about processed features.
        
        Returns:
            Dictionary with categorical and numeric column lists
        """
        return {
            'categorical': self.categorical_columns,
            'numeric': self.numeric_columns,
            'total_features': len(self.categorical_columns) + len(self.numeric_columns)
        }

