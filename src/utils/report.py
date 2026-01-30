import pandas as pd
import numpy as np
from src.evaluation.metrics import evaluate_clustering

def generate_cluster_stats(df, cluster_col):
    """Generate descriptive statistics for each cluster."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude categorical encodings if they shouldn't be averaged, 
    # but initially assuming all numeric columns are relevant or already filtered.
    # Exclude the clustering columns themselves if possible, or just group by the specific one.
    
    stats = df.groupby(cluster_col)[numeric_cols].agg(['mean', 'median', 'std', 'count']).round(2)
    return stats

def compare_algorithms(df, models_dict, X):
    """
    Compare multiple algorithms.
    models_dict: {'Name': (labels)} 
    X: data used for evaluation
    """
    results = []
    for name, labels in models_dict.items():
        if -1 in labels:
            # Handle noise for metrics if needed
            clean_mask = labels != -1
            if np.sum(clean_mask) > 1:
                metrics = evaluate_clustering(X[clean_mask], labels[clean_mask])
                metrics['Algo'] = name
                metrics['Noise_Points'] = np.sum(labels == -1)
                results.append(metrics)
        else:
            metrics = evaluate_clustering(X, labels)
            metrics['Algo'] = name
            metrics['Noise_Points'] = 0
            results.append(metrics)
            
    return pd.DataFrame(results).set_index('Algo')
