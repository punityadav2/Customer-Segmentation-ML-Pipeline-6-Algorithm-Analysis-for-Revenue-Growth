"""
Customer Segmentation Pipeline - Main Entry Point

Command-line interface for running the customer segmentation clustering pipeline.
Supports running individual algorithms or the complete analysis.

Usage:
    python main.py --config config/config.yaml --run-eda --algorithm kmeans
    python main.py --algorithm all
"""

import argparse
import sys
import pandas as pd
import warnings
import logging
import os
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.pipeline import ClusteringPipeline, PipelineError
from src.visualization.eda import set_style, plot_numerical_analysis, plot_correlation
from src.visualization.plots import plot_elbow_method, plot_silhouette_analysis, plot_clusters_2d

# Suppress warnings
warnings.filterwarnings("ignore")


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['logs', 'reports', 'data/processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def run_eda(pipeline: ClusteringPipeline) -> None:
    """
    Run exploratory data analysis.
    
    Args:
        pipeline: Initialized ClusteringPipeline instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Running Exploratory Data Analysis...")
    
    try:
        set_style()
        
        # Get numeric columns from config
        numeric_cols = pipeline.config['columns']['numeric']
        
        # Plot numerical analysis
        logger.info("Generating numerical analysis plots...")
        plot_numerical_analysis(pipeline.df, numeric_cols)
        
        # Plot correlation
        logger.info("Generating correlation matrix...")
        plot_correlation(pipeline.df, numeric_cols)
        
        logger.info("EDA completed successfully")
        
    except Exception as e:
        logger.error(f"EDA failed: {str(e)}")
        raise


def run_single_algorithm(pipeline: ClusteringPipeline, algorithm: str) -> None:
    """
    Run a single clustering algorithm.
    
    Args:
        pipeline: Initialized ClusteringPipeline instance
        algorithm: Algorithm name
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running {algorithm}...")
    
    try:
        # Get algorithm parameters from config
        algo_params = pipeline.config['models'].get(algorithm, {})
        model, labels, metrics = pipeline.run_algorithm(algorithm, **algo_params)
        
        # Display metrics
        logger.info(f"{algorithm} Results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value}")
        
    except PipelineError as e:
        logger.error(f"Algorithm execution failed: {str(e)}")
        raise


def main():
    """Main entry point for the clustering pipeline."""
    
    # Setup
    setup_directories()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Customer Segmentation Clustering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                  # Run all algorithms
  python main.py --algorithm kmeans              # Run K-Means only
  python main.py --run-eda                       # Run EDA only
  python main.py --config custom_config.yaml --algorithm all
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="Run exploratory data analysis"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="all",
        choices=["kmeans", "hierarchical", "dbscan", "affinity_propagation", "meanshift", "optics", "all"],
        help="Algorithm to run"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save clustering results to CSV"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(__name__)
    logger.info("="*60)
    logger.info("Customer Segmentation Pipeline Started")
    logger.info("="*60)
    
    try:
        # Initialize pipeline
        logger.info(f"Loading configuration from {args.config}")
        pipeline = ClusteringPipeline(config_path=args.config)
        
        # Load data
        logger.info("Loading data...")
        pipeline.load_data()
        
        # Run EDA if requested
        if args.run_eda:
            run_eda(pipeline)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        X = pipeline.preprocess()
        logger.info(f"Preprocessed features: {X.shape}")
        
        # Run clustering
        if args.algorithm == "all":
            logger.info("Running all configured algorithms...")
            results = pipeline.run_all_algorithms()
            
            # Display comparison
            logger.info("\nAlgorithm Comparison:")
            comparison_df = pipeline.get_comparison()
            logger.info("\n" + comparison_df.to_string())
            
            # Save comparison
            if args.save_results:
                comparison_df.to_csv("reports/algorithm_comparison.csv")
                logger.info("Results saved to reports/algorithm_comparison.csv")
        else:
            logger.info(f"Running {args.algorithm}...")
            run_single_algorithm(pipeline, args.algorithm)
        
        logger.info("="*60)
        logger.info("Pipeline Completed Successfully")
        logger.info("="*60)
        
    except PipelineError as e:
        logger.error(f"Pipeline Error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
