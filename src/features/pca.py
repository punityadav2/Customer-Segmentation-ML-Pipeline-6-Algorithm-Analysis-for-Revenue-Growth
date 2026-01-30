from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class PCAFeatureGenerator:
    def __init__(self, n_components=None):
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components

    def fit_transform(self, df, threshold=0.95):
        """
        Fit PCA and transform data.
        If n_components is not set, tries to find components explaining 'threshold' variance.
        """
        # First fit to find components if not specified
        if self.n_components is None:
            temp_pca = PCA()
            temp_pca.fit(df)
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumulative_variance >= threshold) + 1
            self.pca = PCA(n_components=self.n_components)
        
        pca_data = self.pca.fit_transform(df)
        columns = [f'PCA_{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(pca_data, columns=columns), self.pca

    def get_explained_variance(self):
        return self.pca.explained_variance_ratio_
