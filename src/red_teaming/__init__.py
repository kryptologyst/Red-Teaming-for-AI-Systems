"""Red Teaming for AI Systems - Adversarial Robustness Testing Framework."""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

from .data import DataLoader, SyntheticDataGenerator
from .models import ModelFactory, RobustModel
from .attacks import AttackFactory, AdversarialAttacker
from .evaluation import RobustnessEvaluator, MetricsCalculator
from .visualization import RobustnessVisualizer

__all__ = [
    "DataLoader",
    "SyntheticDataGenerator", 
    "ModelFactory",
    "RobustModel",
    "AttackFactory",
    "AdversarialAttacker",
    "RobustnessEvaluator",
    "MetricsCalculator",
    "RobustnessVisualizer",
]
