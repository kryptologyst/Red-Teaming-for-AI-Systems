"""Model definitions and training utilities for red teaming experiments."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass


class SklearnModel(BaseModel):
    """Wrapper for scikit-learn models."""
    
    def __init__(self, model: Any) -> None:
        """Initialize with a scikit-learn model.
        
        Args:
            model: Scikit-learn model instance.
        """
        self.model = model
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        logger.info(f"Training {type(self.model).__name__}")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Model training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)


class RobustNeuralNetwork(nn.Module):
    """A simple neural network with robustness features."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        num_classes: int = 3,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize the neural network.
        
        Args:
            input_dim: Input dimension.
            hidden_dims: List of hidden layer dimensions.
            num_classes: Number of output classes.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class PyTorchModel(BaseModel):
    """Wrapper for PyTorch models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        learning_rate: float = 0.001,
        epochs: int = 100,
    ) -> None:
        """Initialize the PyTorch model wrapper.
        
        Args:
            model: PyTorch model.
            device: Device to use ('cpu', 'cuda', 'mps').
            learning_rate: Learning rate for training.
            epochs: Number of training epochs.
        """
        self.model = model
        self.device = self._get_device(device)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.is_fitted = False
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Get the appropriate device."""
        if device is not None:
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        logger.info(f"Training PyTorch model on {self.device}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self.is_fitted = True
        logger.info("PyTorch model training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()


class ModelFactory:
    """Factory for creating different types of models."""
    
    @staticmethod
    def create_model(
        model_type: str,
        **kwargs: Any,
    ) -> BaseModel:
        """Create a model instance.
        
        Args:
            model_type: Type of model to create.
            **kwargs: Additional arguments for model creation.
            
        Returns:
            Model instance.
        """
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                random_state=kwargs.get("random_state", 42),
                max_depth=kwargs.get("max_depth", None),
            )
            return SklearnModel(model)
        
        elif model_type == "logistic_regression":
            model = LogisticRegression(
                random_state=kwargs.get("random_state", 42),
                max_iter=kwargs.get("max_iter", 1000),
            )
            return SklearnModel(model)
        
        elif model_type == "svm":
            model = SVC(
                probability=True,
                random_state=kwargs.get("random_state", 42),
                kernel=kwargs.get("kernel", "rbf"),
            )
            return SklearnModel(model)
        
        elif model_type == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (100, 50)),
                random_state=kwargs.get("random_state", 42),
                max_iter=kwargs.get("max_iter", 1000),
            )
            return SklearnModel(model)
        
        elif model_type == "neural_network":
            input_dim = kwargs.get("input_dim", 4)
            hidden_dims = kwargs.get("hidden_dims", [64, 32])
            num_classes = kwargs.get("num_classes", 3)
            
            model = RobustNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout_rate=kwargs.get("dropout_rate", 0.2),
            )
            
            return PyTorchModel(
                model=model,
                device=kwargs.get("device", None),
                learning_rate=kwargs.get("learning_rate", 0.001),
                epochs=kwargs.get("epochs", 100),
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class RobustModel:
    """Wrapper for models with robustness evaluation capabilities."""
    
    def __init__(self, model: BaseModel, model_name: str = "model") -> None:
        """Initialize the robust model wrapper.
        
        Args:
            model: Base model instance.
            model_name: Name of the model for logging.
        """
        self.model = model
        self.model_name = model_name
        self.training_history: List[Dict[str, float]] = []
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        logger.info(f"Training {self.model_name}")
        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_name} training completed")
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            verbose: Whether to print evaluation results.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            "accuracy": accuracy,
            "n_test_samples": len(y_test),
        }
        
        if verbose:
            logger.info(f"{self.model_name} Test Accuracy: {accuracy:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
