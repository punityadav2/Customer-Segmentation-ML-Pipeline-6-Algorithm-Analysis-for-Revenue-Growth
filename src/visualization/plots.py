import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans

def plot_elbow_method(model, X, k_range=(2, 11)):
    fig, ax = plt.subplots(figsize=(10, 6))
    visualizer = KElbowVisualizer(model, k=k_range, ax=ax)
    visualizer.fit(X)
    visualizer.finalize()
    plt.show()

def plot_silhouette_analysis(model_class, X, k_range=(2, 11)):
    # Line plot
    from sklearn.metrics import silhouette_score
    silhouettes = []
    for k in range(k_range[0], k_range[1]):
        model = model_class(n_clusters=k)
        labels = model.fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))

    plt.figure(figsize=(10, 6))
    plt.plot(range(k_range[0], k_range[1]), silhouettes, marker='o', linestyle='--', color='r')
    plt.title('Silhouette Score vs Clusters')
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
    
    # Subplots
    n_clusters = len(range(k_range[0], k_range[1]))
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, k in enumerate(range(k_range[0], k_range[1])):
        model = model_class(n_clusters=k)
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick', ax=axes[idx])
        visualizer.fit(X)
        axes[idx].set_title(f'k={k}')
        
    plt.tight_layout()
    plt.show()

def plot_clusters_2d(df, x_col, y_col, cluster_col, title, centroids=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=cluster_col, palette='hls')
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='red', label='Centroids')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_dendrograms(X, method_list=['ward', 'complete', 'average', 'single']):
    from scipy.cluster.hierarchy import linkage, dendrogram
    
    n = len(method_list)
    rows = (n + 1) // 2
    plt.figure(figsize=(16, 8))

    for i, method in enumerate(method_list, start=1):
        plt.subplot(rows, 2, i)
        plt.title(method.capitalize())
        plt.xlabel("Observations")
        plt.ylabel("Distance")
        
        linkage_matrix = linkage(X, method=method)
        dendrogram(linkage_matrix, leaf_font_size=12, truncate_mode='lastp')
    
    plt.tight_layout()
    plt.show()

def plot_pca_variance(pca_obj):
    explained_variance = pca_obj.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    features = range(len(explained_variance))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot
    axes[0].bar(features, explained_variance)
    axes[0].set_xlabel('PCA Feature')
    axes[0].set_ylabel('Variance Ratio')
    axes[0].set_title('Explained Variance Ratio')
    
    # Cumulative plot
    axes[1].plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_tuning_results(df_results, x_col, y1_col, y2_col, title):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y1_col, color=color)
    ax1.plot(df_results[x_col], df_results[y1_col], color=color, marker='o', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel(y2_col, color=color)  
    ax2.plot(df_results[x_col], df_results[y2_col], color=color, marker='x', linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    fig.tight_layout()  
    plt.show()
