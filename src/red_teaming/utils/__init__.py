"""Utility functions for red teaming experiments."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for PyTorch operations.
    
    Args:
        device: Preferred device ('cpu', 'cuda', 'mps').
        
    Returns:
        PyTorch device.
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level.
        log_file: Path to log file (optional).
        format_string: Custom format string (optional).
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
    )
    
    logger.info(f"Logging setup completed with level {level}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save configuration.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")


def create_experiment_directory(
    base_dir: Union[str, Path] = "experiments",
    experiment_name: str = "experiment",
) -> Path:
    """Create a new experiment directory with standard structure.
    
    Args:
        base_dir: Base directory for experiments.
        experiment_name: Name of the experiment.
        
    Returns:
        Path to the created experiment directory.
    """
    base_dir = Path(base_dir)
    exp_dir = base_dir / experiment_name
    
    # Create subdirectories
    subdirs = ["data", "models", "results", "plots", "logs"]
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Experiment directory created: {exp_dir}")
    return exp_dir


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "Progress",
    suffix: str = "Complete",
    length: int = 50,
) -> None:
    """Print a progress bar.
    
    Args:
        current: Current progress value.
        total: Total value.
        prefix: Prefix text.
        suffix: Suffix text.
        length: Length of the progress bar.
    """
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="\r")
    
    if current == total:
        print()


class ExperimentTracker:
    """Simple experiment tracker for logging results."""
    
    def __init__(self, log_file: Union[str, Path]) -> None:
        """Initialize the experiment tracker.
        
        Args:
            log_file: Path to log file.
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
    
    def log_result(self, result: Dict[str, Any]) -> None:
        """Log a result.
        
        Args:
            result: Result dictionary to log.
        """
        self.results.append(result)
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(f"{result}\n")
        
        logger.info(f"Result logged: {result}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get all logged results.
        
        Returns:
            List of all results.
        """
        return self.results
    
    def clear_results(self) -> None:
        """Clear all results."""
        self.results = []
        if self.log_file.exists():
            self.log_file.unlink()
        logger.info("Results cleared")


def validate_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> bool:
    """Validate input data for common issues.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        feature_names: Names of features (optional).
        
    Returns:
        True if data is valid, False otherwise.
    """
    issues = []
    
    # Check shapes
    if len(X) != len(y):
        issues.append(f"X and y have different lengths: {len(X)} vs {len(y)}")
    
    # Check for NaN values
    if np.isnan(X).any():
        issues.append("X contains NaN values")
    
    if np.isnan(y).any():
        issues.append("y contains NaN values")
    
    # Check for infinite values
    if np.isinf(X).any():
        issues.append("X contains infinite values")
    
    if np.isinf(y).any():
        issues.append("y contains infinite values")
    
    # Check feature names
    if feature_names and len(feature_names) != X.shape[1]:
        issues.append(f"Feature names length ({len(feature_names)}) doesn't match X shape ({X.shape[1]})")
    
    if issues:
        logger.warning(f"Data validation issues found: {issues}")
        return False
    
    logger.info("Data validation passed")
    return True


def normalize_features(
    X: np.ndarray,
    method: str = "standard",
    fit_transform: bool = True,
) -> np.ndarray:
    """Normalize features using various methods.
    
    Args:
        X: Feature matrix.
        method: Normalization method ('standard', 'minmax', 'robust').
        fit_transform: Whether to fit and transform or just transform.
        
    Returns:
        Normalized feature matrix.
    """
    if method == "standard":
        if fit_transform:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            return (X - mean) / (std + 1e-8)
        else:
            return X  # Would need stored parameters in practice
    
    elif method == "minmax":
        if fit_transform:
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            return (X - min_vals) / (max_vals - min_vals + 1e-8)
        else:
            return X  # Would need stored parameters in practice
    
    elif method == "robust":
        if fit_transform:
            median = np.median(X, axis=0)
            mad = np.median(np.abs(X - median), axis=0)
            return (X - median) / (mad + 1e-8)
        else:
            return X  # Would need stored parameters in practice
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics for data.
    
    Args:
        data: Input data array.
        
    Returns:
        Dictionary of statistics.
    """
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "median": np.median(data),
        "q25": np.percentile(data, 25),
        "q75": np.percentile(data, 75),
    }
