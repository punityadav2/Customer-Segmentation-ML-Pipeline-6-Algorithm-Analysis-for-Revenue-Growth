import unittest
import pandas as pd
import numpy as np
from src.features.preprocess import Preprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        data = {
            'CustomerID': [1, 2, 3],
            'Gender': ['Male', 'Female', 'Male'],
            'Age': [20, 30, 40],
            'Annual Income (k$)': [15, 20, 25],
            'Spending Score (1-100)': [10, 50, 90]
        }
        self.df = pd.DataFrame(data)
        self.rename_map = {'Annual Income (k$)': 'annual_income', 'Spending Score (1-100)': 'spending_score'}
        self.features_to_drop = ['CustomerID']

    def test_initial_cleaning(self):
        p = Preprocessor(rename_map=self.rename_map)
        df_clean = p.initial_cleaning(self.df)
        self.assertIn('annual_income', df_clean.columns)
        self.assertNotIn('Annual Income (k$)', df_clean.columns)

    def test_fit_transform(self):
        p = Preprocessor(rename_map=self.rename_map, features_to_drop=self.features_to_drop)
        df_clean = p.initial_cleaning(self.df)
        df_scaled, scaler = p.fit_transform(df_clean)
        
        # Checks
        self.assertEqual(df_scaled.shape[0], 3)
        self.assertNotIn('CustomerID', df_scaled.columns)
        # Check if scaled value is approximately standardized
        self.assertTrue(np.isclose(df_scaled['annual_income'].mean(), 0, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
