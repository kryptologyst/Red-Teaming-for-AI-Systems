"""Data loading and preprocessing utilities for red teaming experiments."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and preprocessing for red teaming experiments."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize the data loader.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        
    def load_iris_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Load and preprocess the Iris dataset.
        
        Returns:
            Tuple of (X, y, feature_names, target_names).
        """
        logger.info("Loading Iris dataset")
        data = load_iris()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_names = data.target_names.tolist()
        
        logger.info(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names, target_names
    
    def load_synthetic_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        n_informative: int = 10,
        n_redundant: int = 5,
        n_classes: int = 3,
        n_clusters_per_class: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Generate a synthetic classification dataset.
        
        Args:
            n_samples: Number of samples to generate.
            n_features: Total number of features.
            n_informative: Number of informative features.
            n_redundant: Number of redundant features.
            n_classes: Number of classes.
            n_clusters_per_class: Number of clusters per class.
            
        Returns:
            Tuple of (X, y, feature_names, target_names).
        """
        logger.info(f"Generating synthetic dataset: {n_samples} samples, {n_features} features")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            random_state=self.random_state,
        )
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        target_names = [f"class_{i}" for i in range(n_classes)]
        
        logger.info(f"Generated synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names, target_names
    
    def preprocess_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3,
        scale_features: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess the data for training and testing.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of data to use for testing.
            scale_features: Whether to scale features.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        logger.info("Preprocessing data")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features if requested
        if scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            logger.info("Features scaled using StandardScaler")
        
        logger.info(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
    
    def save_dataset_metadata(
        self,
        feature_names: List[str],
        target_names: List[str],
        output_path: Union[str, Path],
    ) -> None:
        """Save dataset metadata to a JSON file.
        
        Args:
            feature_names: List of feature names.
            target_names: List of target class names.
            output_path: Path to save the metadata file.
        """
        metadata = {
            "feature_names": feature_names,
            "target_names": target_names,
            "n_features": len(feature_names),
            "n_classes": len(target_names),
            "feature_types": ["continuous"] * len(feature_names),
            "sensitive_attributes": [],
            "monotonic_features": [],
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset metadata saved to {output_path}")


class SyntheticDataGenerator:
    """Generates synthetic datasets for testing adversarial robustness."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize the synthetic data generator.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
    
    def generate_adversarial_test_set(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epsilon: float = 0.1,
        attack_type: str = "gaussian",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate adversarial test examples.
        
        Args:
            X_test: Original test features.
            y_test: Original test labels.
            epsilon: Perturbation strength.
            attack_type: Type of attack ('gaussian', 'uniform', 'fgsm').
            
        Returns:
            Tuple of (X_adversarial, y_test) where y_test remains unchanged.
        """
        logger.info(f"Generating adversarial test set with epsilon={epsilon}, type={attack_type}")
        
        if attack_type == "gaussian":
            perturbation = np.random.normal(0, epsilon, X_test.shape)
        elif attack_type == "uniform":
            perturbation = np.random.uniform(-epsilon, epsilon, X_test.shape)
        elif attack_type == "fgsm":
            # Simple FGSM-like perturbation (requires gradients in practice)
            perturbation = epsilon * np.sign(np.random.randn(*X_test.shape))
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        X_adversarial = X_test + perturbation
        
        logger.info(f"Generated {X_adversarial.shape[0]} adversarial examples")
        return X_adversarial, y_test
    
    def generate_out_of_distribution_samples(
        self,
        X_train: np.ndarray,
        n_samples: int = 100,
        shift_factor: float = 2.0,
    ) -> np.ndarray:
        """Generate out-of-distribution samples by shifting the distribution.
        
        Args:
            X_train: Training data to base the shift on.
            n_samples: Number of OOD samples to generate.
            shift_factor: Factor by which to shift the distribution.
            
        Returns:
            Array of OOD samples.
        """
        logger.info(f"Generating {n_samples} OOD samples with shift_factor={shift_factor}")
        
        # Calculate mean and std of training data
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        
        # Generate samples with shifted distribution
        X_ood = np.random.normal(
            mean + shift_factor * std,
            std,
            size=(n_samples, X_train.shape[1])
        )
        
        logger.info(f"Generated {X_ood.shape[0]} OOD samples")
        return X_ood
