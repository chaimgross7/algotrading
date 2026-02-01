"""Utility functions."""

import copy
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_device(device: str = "auto") -> torch.device:
    """Get torch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    
    Values from `override` take precedence. Nested dicts are merged recursively.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
    
    Returns:
        Merged configuration dictionary
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    path: str,
    data_config: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load YAML config file with optional data config merging.
    
    Args:
        path: Path to the main experiment config file
        data_config: Optional path to a separate data config file.
                     If provided, its 'data' section is merged into the result.
    
    Returns:
        Merged configuration dictionary
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    
    # Merge data config if provided
    if data_config:
        with open(data_config) as f:
            data_cfg = yaml.safe_load(f)
        # Data config's 'data' section takes precedence, then experiment config can override
        if "data" in data_cfg:
            base_data = data_cfg.get("data", {})
            exp_data = config.get("data", {})
            config["data"] = deep_merge(base_data, exp_data)
    
    return config


def save_config(config: Dict[str, Any], path: str):
    """Save config to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
