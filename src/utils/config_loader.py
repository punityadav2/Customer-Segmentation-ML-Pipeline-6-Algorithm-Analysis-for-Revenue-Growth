import yaml
import os

def load_config(config_path="config/config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
