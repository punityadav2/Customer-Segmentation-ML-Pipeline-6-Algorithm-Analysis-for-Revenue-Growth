"""
Logging configuration and setup module.

Provides centralized logging configuration for the clustering pipeline
with both file and console handlers.
"""

import logging
import os
import yaml
import logging.config
from pathlib import Path


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: str = "logs",
    log_file: str = "clustering.log"
) -> logging.Logger:
    """
    Setup and return a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_file: Log file name
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Pipeline started")
        >>> logger.error("Error occurred")
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_logging(
    config_path: str = "config/config.yaml",
    default_level: int = logging.INFO
) -> None:
    """
    Setup logging configuration from YAML config file.
    
    Args:
        config_path: Path to configuration YAML file
        default_level: Default logging level if config not found
    
    Example:
        >>> setup_logging("config/config.yaml")
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    path = config_path
    if os.path.exists(path):
        try:
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
                
                if config and 'logging' in config:
                    logging_config = config['logging']
                    level = logging.getLevelName(logging_config.get('level', 'INFO'))
                    log_file = logging_config.get('file', 'clustering.log')
                    
                    # Setup logger
                    setup_logger('', level=level, log_file=log_file)
                    return
        except Exception as e:
            print(f'Error reading logging config: {e}')
            print('Using default logging configuration')
    
    # Default basic config
    logging.basicConfig(
        level=default_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "app.log")),
            logging.StreamHandler()
        ]
    )
