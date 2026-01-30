"""
Exploratory Data Analysis (EDA) visualization module.

Provides functions for comprehensive univariate, bivariate, and 
correlation analysis with professional visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from itertools import combinations
from scipy.stats import pearsonr
from typing import List, Optional, Tuple
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default visualization directory
VIS_DIR = Path('visualizations')
VIS_DIR.mkdir(exist_ok=True)


class VisualizationError(Exception):
    """Custom exception for visualization failures."""
    pass


def save_plot(
    filename: str,
    format: str = 'png',
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> str:
    """
    Save current matplotlib figure to file.
    
    Args:
        filename: Name of file (without extension)
        format: File format ('png', 'pdf', 'jpg')
        dpi: Resolution (dots per inch)
        bbox_inches: Bounding box ('tight' removes extra whitespace)
    
    Returns:
        Path to saved file
    """
    filepath = VIS_DIR / f"{filename}.{format}"
    plt.savefig(filepath, format=format, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Plot saved to {filepath}")
    return str(filepath)


def set_style(style: str = "whitegrid") -> None:
    """
    Set seaborn plotting style.
    
    Args:
        style: Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
    """
    sns.set_style(style)
    logger.info(f"Set plotting style to {style}")


def plot_categorical_analysis(
    df: pd.DataFrame,
    column: str,
    figsize: Tuple[int, int] = (16, 7),
    palette: str = 'pastel',
    save_fig: bool = False
) -> None:
    """
    Plot count and pie chart for categorical column.
    
    Args:
        df: Input DataFrame
        column: Categorical column name
        figsize: Figure size (width, height)
        palette: Color palette name
        save_fig: Whether to save plot to file
    
    Raises:
        VisualizationError: If column not found
    
    Example:
        >>> df = pd.DataFrame({'Gender': ['M', 'F', 'M', 'F']})
        >>> plot_categorical_analysis(df, 'Gender')
    """
    try:
        if column not in df.columns:
            raise VisualizationError(f"Column '{column}' not found in DataFrame")
        
        palette_colors = sns.color_palette(palette, n_colors=len(df[column].unique()))
        fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1]})

        sns.countplot(
            data=df,
            x=column,
            hue=column,
            ax=axs[0],
            order=df[column].value_counts().index,
            palette=palette_colors
        )
        for container in axs[0].containers:
            axs[0].bar_label(container, fontsize=12, color='black')
        axs[0].set_title(f"Count of {column.capitalize()}", fontsize=16, fontweight='bold')
        axs[0].legend_.remove()

        counts = df[column].value_counts()
        axs[1].pie(
            counts,
            labels=counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=palette_colors,
            wedgeprops=dict(width=0.3, edgecolor='w', linewidth=1.5),
            textprops=dict(color='black', fontsize=12)
        )
        axs[1].add_artist(plt.Circle((0, 0), 0.25, fc='white', edgecolor='w', linewidth=1.5))
        axs[1].set_title(f"Ratio of {column.capitalize()}", fontsize=16, fontweight='bold')

        plt.tight_layout()
        
        if save_fig:
            save_plot(f"categorical_analysis_{column}")
        
        plt.show()
        
        logger.info(f"Categorical analysis plot created for {column}")
        
    except VisualizationError:
        raise
    except Exception as e:
        logger.error(f"Failed to plot categorical analysis: {str(e)}")
        raise VisualizationError(f"Categorical plot failed: {str(e)}")


def plot_numerical_analysis(
    df: pd.DataFrame,
    num_cols: List[str],
    dist_color: str = 'teal',
    box_color: str = 'orange',
    figsize_per_row: Tuple[int, int] = (18, 5),
    save_fig: bool = False
) -> None:
    """
    Plot distribution and boxplot for numerical columns.
    
    Args:
        df: Input DataFrame
        num_cols: List of numerical column names
        dist_color: Color for distribution plots
        box_color: Color for boxplots
        figsize_per_row: Figure size per row (width, height)
        save_fig: Whether to save plot to file
    
    Raises:
        VisualizationError: If columns not found
    
    Example:
        >>> df = pd.DataFrame({'Age': [25, 30, 35], 'Income': [50000, 60000, 70000]})
        >>> plot_numerical_analysis(df, ['Age', 'Income'])
    """
    try:
        missing_cols = [col for col in num_cols if col not in df.columns]
        if missing_cols:
            raise VisualizationError(f"Columns not found: {missing_cols}")
        
        n = len(num_cols)
        height = figsize_per_row[1] * n
        fig, axes = plt.subplots(
            nrows=n,
            ncols=2,
            figsize=(figsize_per_row[0], height)
        )
        fig.tight_layout(pad=5.0)

        for i, col in enumerate(num_cols):
            # Distribution Plot
            sns.histplot(df[col], kde=True, ax=axes[i, 0], color=dist_color)
            axes[i, 0].set_title(f'Distribution of {col}', fontsize=16, fontweight='bold')
            
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            
            axes[i, 0].axvline(mean_val, color='blue', linestyle='--', linewidth=2)
            axes[i, 0].axvline(median_val, color='yellow', linestyle='-', linewidth=2)
            
            legend_elements = [
                Line2D([0], [0], color='blue', linestyle='--', lw=2, label=f'Mean: {mean_val:.2f}'),
                Line2D([0], [0], color='yellow', linestyle='-', lw=2, label=f'Median: {median_val:.2f}'),
                Line2D([0], [0], color='red', linestyle='-', lw=2, label=f'Std Dev: {std_val:.2f}')
            ]
            axes[i, 0].legend(handles=legend_elements)

            # Boxplot
            sns.boxplot(data=df, x=col, ax=axes[i, 1], color=box_color, width=0.5)
            axes[i, 1].set_title(f'Boxplot of {col}', fontsize=16, fontweight='bold')

        plt.suptitle('Numerical Columns Analysis', fontsize=20, fontweight='bold', y=1.002)
        
        if save_fig:
            save_plot('numerical_analysis')
        
        plt.show()
        
        logger.info(f"Numerical analysis plots created for {len(num_cols)} columns")
        
    except VisualizationError:
        raise
    except Exception as e:
        logger.error(f"Failed to plot numerical analysis: {str(e)}")
        raise VisualizationError(f"Numerical plots failed: {str(e)}")


def plot_bivariate_analysis(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_col: str,
    figsize_per_row: Tuple[int, int] = (18, 5),
    save_fig: bool = False
) -> None:
    """
    Plot bivariate relationships between numerical and categorical columns.
    
    Args:
        df: Input DataFrame
        num_cols: List of numerical column names
        cat_col: Categorical column for grouping
        figsize_per_row: Figure size per row
        save_fig: Whether to save plot to file
    
    Raises:
        VisualizationError: If columns not found
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Gender': ['M', 'F', 'M'],
        ...     'Age': [25, 30, 35],
        ...     'Income': [50000, 60000, 70000]
        ... })
        >>> plot_bivariate_analysis(df, ['Age', 'Income'], 'Gender')
    """
    try:
        missing_cols = [col for col in num_cols if col not in df.columns]
        if cat_col not in df.columns:
            missing_cols.append(cat_col)
        if missing_cols:
            raise VisualizationError(f"Columns not found: {missing_cols}")
        
        n = len(num_cols)
        height = figsize_per_row[1] * n
        fig, axes = plt.subplots(
            nrows=n,
            ncols=3,
            figsize=(figsize_per_row[0], height)
        )
        fig.tight_layout(pad=5.0)

        for i, col in enumerate(num_cols):
            # Boxplot by category
            sns.boxplot(data=df, x=cat_col, y=col, hue=cat_col, ax=axes[i, 0], palette='Set1')
            axes[i, 0].set_title(f'Boxplot of {col} by {cat_col}', fontsize=12, fontweight='bold')
            axes[i, 0].legend_.remove()
            
            # Category count
            sns.countplot(data=df, x=cat_col, hue=cat_col, ax=axes[i, 1], palette='Set1')
            axes[i, 1].set_title(f'Count of {cat_col}', fontsize=12, fontweight='bold')
            axes[i, 1].legend_.remove()
            
            # Stripplot
            sns.stripplot(
                data=df,
                x=cat_col,
                hue=cat_col,
                y=col,
                ax=axes[i, 2],
                palette='Set1',
                jitter=True,
                size=8
            )
            axes[i, 2].set_title(f'Stripplot of {col} by {cat_col}', fontsize=12, fontweight='bold')
            axes[i, 2].legend_.remove()

        plt.suptitle(f'Bivariate Analysis by {cat_col}', fontsize=18, fontweight='bold', y=1.002)
        
        if save_fig:
            save_plot(f'bivariate_analysis_{cat_col}')
        
        plt.show()
        
        logger.info(f"Bivariate analysis plots created for {len(num_cols)} numerical vs {cat_col}")
        
    except VisualizationError:
        raise
    except Exception as e:
        logger.error(f"Failed to plot bivariate analysis: {str(e)}")
        raise VisualizationError(f"Bivariate plots failed: {str(e)}")


def plot_correlation(
    df: pd.DataFrame,
    num_cols: List[str],
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    save_fig: bool = False
) -> pd.DataFrame:
    """
    Plot correlation heatmap for numerical columns.
    
    Args:
        df: Input DataFrame
        num_cols: List of numerical column names
        figsize: Figure size
        cmap: Color map name
        save_fig: Whether to save plot to file
    
    Returns:
        Correlation matrix DataFrame
    
    Raises:
        VisualizationError: If columns not found
    
    Example:
        >>> df = pd.DataFrame({'Age': [25, 30, 35], 'Income': [50000, 60000, 70000]})
        >>> corr_matrix = plot_correlation(df, ['Age', 'Income'])
    """
    try:
        missing_cols = [col for col in num_cols if col not in df.columns]
        if missing_cols:
            raise VisualizationError(f"Columns not found: {missing_cols}")
        
        plt.figure(figsize=figsize)
        corr_matrix = df[num_cols].corr()
        
        sns.heatmap(
            corr_matrix,
            linewidths=0.5,
            square=True,
            cmap=cmap,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Correlation'}
        )
        plt.title("Correlation Matrix", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_fig:
            save_plot('correlation_matrix')
        
        plt.show()
        
        logger.info(f"Correlation heatmap created for {len(num_cols)} columns")
        return corr_matrix
        
    except VisualizationError:
        raise
    except Exception as e:
        logger.error(f"Failed to plot correlation: {str(e)}")
        raise VisualizationError(f"Correlation plot failed: {str(e)}")
