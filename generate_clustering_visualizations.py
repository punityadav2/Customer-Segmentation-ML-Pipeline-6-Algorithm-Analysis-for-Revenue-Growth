"""
Generate Clustering Result Visualizations

Creates visual representations of clustering results including:
- 2D cluster scatter plots (using first 2 features)
- Cluster size distribution
- Algorithm comparison metrics
- Cluster characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Ensure visualizations directory exists
Path("visualizations").mkdir(exist_ok=True)


def load_data_and_run_clustering():
    """Load data and run all clustering algorithms."""
    from src.data.loader import load_data
    from src.features.preprocess import Preprocessor
    from src.utils.config_loader import load_config
    from src.models.kmeans import run_kmeans
    from src.models.hierarchical import run_hierarchical
    from src.models.dbscan import run_dbscan
    from src.models.affinity_propagation import run_affinity_propagation
    from src.models.meanshift import run_meanshift
    from src.models.optics import run_optics
    
    # Load configuration
    config = load_config('config/config.yaml')
    logger.info("Configuration loaded")
    
    # Load data
    df = load_data('data/raw/Mall_Customers.csv')
    logger.info(f"Data loaded: {df.shape}")
    
    # Preprocess
    preprocessor = Preprocessor(config)
    result = preprocessor.fit_transform(df)
    
    # Handle both tuple and dataframe returns
    if isinstance(result, tuple):
        df_processed = result[0]
    else:
        df_processed = result
    
    logger.info(f"Data preprocessed: {df_processed.shape}")
    
    # Get numeric columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Prepare data for clustering
    X = df_processed[numeric_cols].values
    
    # Run all algorithms
    results = {}
    
    # K-Means
    try:
        model_km, labels_km, metrics_km = run_kmeans(X, n_clusters=4)
        results['K-Means'] = {
            'model': model_km,
            'labels': labels_km,
            'metrics': metrics_km,
            'X': X,
            'features': numeric_cols
        }
        logger.info("✓ K-Means completed")
    except Exception as e:
        logger.error(f"K-Means failed: {e}")
    
    # Hierarchical
    try:
        model_hc, labels_hc, metrics_hc = run_hierarchical(X, n_clusters=4)
        results['Hierarchical'] = {
            'model': model_hc,
            'labels': labels_hc,
            'metrics': metrics_hc,
            'X': X,
            'features': numeric_cols
        }
        logger.info("✓ Hierarchical completed")
    except Exception as e:
        logger.error(f"Hierarchical failed: {e}")
    
    # DBSCAN
    try:
        model_db, labels_db, metrics_db = run_dbscan(X, eps=1.5, min_samples=5)
        results['DBSCAN'] = {
            'model': model_db,
            'labels': labels_db,
            'metrics': metrics_db,
            'X': X,
            'features': numeric_cols
        }
        logger.info("✓ DBSCAN completed")
    except Exception as e:
        logger.error(f"DBSCAN failed: {e}")
    
    # Affinity Propagation
    try:
        model_ap, labels_ap, metrics_ap = run_affinity_propagation(X)
        results['Affinity Propagation'] = {
            'model': model_ap,
            'labels': labels_ap,
            'metrics': metrics_ap,
            'X': X,
            'features': numeric_cols
        }
        logger.info("✓ Affinity Propagation completed")
    except Exception as e:
        logger.error(f"Affinity Propagation failed: {e}")
    
    # Mean Shift
    try:
        model_ms, labels_ms, metrics_ms = run_meanshift(X)
        results['Mean Shift'] = {
            'model': model_ms,
            'labels': labels_ms,
            'metrics': metrics_ms,
            'X': X,
            'features': numeric_cols
        }
        logger.info("✓ Mean Shift completed")
    except Exception as e:
        logger.error(f"Mean Shift failed: {e}")
    
    # OPTICS
    try:
        model_opt, labels_opt, metrics_opt = run_optics(X)
        results['OPTICS'] = {
            'model': model_opt,
            'labels': labels_opt,
            'metrics': metrics_opt,
            'X': X,
            'features': numeric_cols
        }
        logger.info("✓ OPTICS completed")
    except Exception as e:
        logger.error(f"OPTICS failed: {e}")
    
    return results, df_processed


def plot_2d_clusters(results, algorithm, X, labels, features):
    """Create 2D scatter plot of clusters using first 2 features with colors and shapes."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique clusters
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Different marker shapes for variety
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd']
    
    # Plot each cluster with different color AND marker shape
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Noise points in black
            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], c='black', marker='x', 
                      s=120, alpha=0.6, label='Noise', edgecolors='darkred', linewidth=1.5, zorder=3)
        else:
            mask = labels == label
            marker = markers[i % len(markers)]
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], marker=marker,
                      label=f'Cluster {label}', s=80, alpha=0.8, 
                      edgecolors='black', linewidth=0.8, zorder=2)
    
    # Add cluster centers if available
    if hasattr(results[algorithm]['model'], 'cluster_centers_'):
        centers = results[algorithm]['model'].cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='gold', marker='*', 
                  s=800, edgecolors='darkred', linewidth=2.5, label='Centroids', zorder=5)
    
    ax.set_xlabel(f'{features[0]} (Normalized)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{features[1]} (Normalized)', fontsize=11, fontweight='bold')
    ax.set_title(f'{algorithm} - 2D Cluster Visualization', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'visualizations/{algorithm.lower().replace(" ", "_")}_2d_clusters.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {filename}")
    plt.close()


def plot_cluster_sizes(results, algorithm, labels):
    """Create bar chart of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors_list = []
    labels_list = []
    for i, label in enumerate(unique):
        if label == -1:
            colors_list.append('black')
            labels_list.append('Noise')
        else:
            colors_list.append(f'C{i % 10}')
            labels_list.append(f'Cluster {label}')
    
    bars = ax.bar(labels_list, counts, color=colors_list, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Number of Points', fontsize=11, fontweight='bold')
    ax.set_title(f'{algorithm} - Cluster Size Distribution', fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    filename = f'visualizations/{algorithm.lower().replace(" ", "_")}_cluster_sizes.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {filename}")
    plt.close()


def plot_all_algorithms_comparison(results):
    """Create comparison visualization of all algorithms with colors and shapes."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Different marker shapes
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    features_to_use = None
    
    for idx, (algorithm, data) in enumerate(results.items()):
        if idx >= 6:
            break
            
        X = data['X']
        labels = data['labels']
        features = data['features']
        
        if features_to_use is None:
            features_to_use = features[:2]
        
        ax = axes[idx]
        
        # Plot clusters with colors and shapes
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 3)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                mask = labels == label
                ax.scatter(X[mask, 0], X[mask, 1], c='black', marker='x', 
                          s=60, alpha=0.6, edgecolors='darkred', linewidth=1)
            else:
                mask = labels == label
                marker = markers[i % len(markers)]
                ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i % len(colors)]], 
                          marker=marker, s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add centroids if available
        if hasattr(data['model'], 'cluster_centers_'):
            centers = data['model'].cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], c='gold', marker='*', 
                      s=400, edgecolors='darkred', linewidth=2, zorder=5)
        
        # Metrics string
        metrics = data['metrics']
        silhouette = metrics.get('silhouette', np.nan)
        davies_bouldin = metrics.get('davies_bouldin', np.nan)
        n_clusters = metrics.get('n_clusters', 0)
        
        metrics_text = f'Silhouette: {silhouette:.3f}\nDB Index: {davies_bouldin:.3f}\nClusters: {int(n_clusters)}'
        
        ax.set_title(f'{algorithm}\n{metrics_text}', fontsize=10, fontweight='bold')
        ax.set_xlabel(features[0], fontsize=9)
        ax.set_ylabel(features[1], fontsize=9)
        ax.grid(True, alpha=0.2)
    
    # Hide extra subplots
    for idx in range(len(results), 6):
        axes[idx].set_visible(False)
    
    fig.suptitle('All Clustering Algorithms - Comparison (Colors + Shapes)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('visualizations/all_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: visualizations/all_algorithms_comparison.png")
    plt.close()


def plot_metrics_comparison(results):
    """Create comparison of metrics across all algorithms."""
    algorithms = []
    silhouettes = []
    davies_bouldin = []
    n_clusters = []
    
    for algorithm, data in results.items():
        algorithms.append(algorithm)
        metrics = data['metrics']
        silhouettes.append(metrics.get('silhouette', np.nan))
        davies_bouldin.append(metrics.get('davies_bouldin', np.nan))
        n_clusters.append(metrics.get('n_clusters', 0))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Silhouette Score
    colors_sil = ['green' if x > 0.5 else 'orange' if x > 0 else 'red' for x in silhouettes]
    bars1 = axes[0].bar(range(len(algorithms)), silhouettes, color=colors_sil, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Score (Higher is Better)', fontweight='bold')
    axes[0].set_title('Silhouette Score Comparison', fontweight='bold', fontsize=12)
    axes[0].set_ylim([np.nanmin(silhouettes)-0.1, np.nanmax(silhouettes)+0.1])
    axes[0].axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good (>0.5)')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].legend()
    
    for bar, val in zip(bars1, silhouettes):
        if not np.isnan(val):
            axes[0].text(bar.get_x() + bar.get_width()/2., val, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Davies-Bouldin Index
    colors_db = ['green' if x < 1 else 'orange' if x < 1.5 else 'red' for x in davies_bouldin]
    bars2 = axes[1].bar(range(len(algorithms)), davies_bouldin, color=colors_db, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Index (Lower is Better)', fontweight='bold')
    axes[1].set_title('Davies-Bouldin Index Comparison', fontweight='bold', fontsize=12)
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Good (<1.0)')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend()
    
    for bar, val in zip(bars2, davies_bouldin):
        if not np.isnan(val):
            axes[1].text(bar.get_x() + bar.get_width()/2., val, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Number of Clusters
    bars3 = axes[2].bar(range(len(algorithms)), n_clusters, color='steelblue', alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Count', fontweight='bold')
    axes[2].set_title('Number of Clusters Found', fontweight='bold', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, n_clusters):
        axes[2].text(bar.get_x() + bar.get_width()/2., val, f'{int(val)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for ax in axes:
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
    
    fig.suptitle('Clustering Algorithms - Metrics Comparison', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('visualizations/metrics_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: visualizations/metrics_comparison.png")
    plt.close()


def main():
    """Generate all clustering visualizations."""
    logger.info("=" * 60)
    logger.info("Starting Clustering Visualization Generation...")
    logger.info("=" * 60)
    
    # Load data and run clustering
    logger.info("\nStep 1: Loading data and running clustering algorithms...")
    results, df_processed = load_data_and_run_clustering()
    
    logger.info(f"\nStep 2: Found {len(results)} algorithms with results")
    
    # Generate individual algorithm visualizations
    logger.info("\nStep 3: Generating individual algorithm visualizations...")
    for algorithm, data in results.items():
        try:
            plot_2d_clusters(results, algorithm, data['X'], data['labels'], data['features'])
            plot_cluster_sizes(results, algorithm, data['labels'])
        except Exception as e:
            logger.error(f"Failed to visualize {algorithm}: {e}")
    
    # Generate comparison visualizations
    logger.info("\nStep 4: Generating comparison visualizations...")
    try:
        plot_all_algorithms_comparison(results)
    except Exception as e:
        logger.error(f"Failed to create algorithm comparison: {e}")
    
    try:
        plot_metrics_comparison(results)
    except Exception as e:
        logger.error(f"Failed to create metrics comparison: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ All clustering visualizations generated successfully!")
    logger.info("=" * 60)
    logger.info("\nGenerated Files:")
    logger.info("  - visualizations/k_means_2d_clusters.png")
    logger.info("  - visualizations/k_means_cluster_sizes.png")
    logger.info("  - visualizations/hierarchical_2d_clusters.png")
    logger.info("  - visualizations/hierarchical_cluster_sizes.png")
    logger.info("  - visualizations/dbscan_2d_clusters.png")
    logger.info("  - visualizations/dbscan_cluster_sizes.png")
    logger.info("  - visualizations/affinity_propagation_2d_clusters.png")
    logger.info("  - visualizations/affinity_propagation_cluster_sizes.png")
    logger.info("  - visualizations/mean_shift_2d_clusters.png")
    logger.info("  - visualizations/mean_shift_cluster_sizes.png")
    logger.info("  - visualizations/optics_2d_clusters.png")
    logger.info("  - visualizations/optics_cluster_sizes.png")
    logger.info("  - visualizations/all_algorithms_comparison.png")
    logger.info("  - visualizations/metrics_comparison.png")
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
