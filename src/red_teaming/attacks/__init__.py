"""Adversarial attack implementations for red teaming experiments."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchattacks import FGSM, PGD, CW, DeepFool, AutoAttack

logger = logging.getLogger(__name__)


class BaseAttack(ABC):
    """Abstract base class for adversarial attacks."""
    
    @abstractmethod
    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate adversarial examples."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the attack."""
        pass


class SimpleAttack(BaseAttack):
    """Simple adversarial attacks for non-PyTorch models."""
    
    def __init__(
        self,
        attack_type: str = "gaussian",
        epsilon: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the simple attack.
        
        Args:
            attack_type: Type of attack ('gaussian', 'uniform', 'fgsm_like').
            epsilon: Perturbation strength.
            random_state: Random seed for reproducibility.
        """
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate adversarial examples."""
        if self.attack_type == "gaussian":
            perturbation = np.random.normal(0, self.epsilon, X.shape)
        elif self.attack_type == "uniform":
            perturbation = np.random.uniform(-self.epsilon, self.epsilon, X.shape)
        elif self.attack_type == "fgsm_like":
            # Simple FGSM-like perturbation (random direction)
            perturbation = self.epsilon * np.sign(np.random.randn(*X.shape))
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
        
        X_adv = X + perturbation
        return X_adv
    
    def get_name(self) -> str:
        """Get the name of the attack."""
        return f"Simple_{self.attack_type}_{self.epsilon}"


class PyTorchAttack(BaseAttack):
    """PyTorch-based adversarial attacks using torchattacks."""
    
    def __init__(
        self,
        attack_type: str = "fgsm",
        epsilon: float = 0.1,
        model: Optional[nn.Module] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PyTorch attack.
        
        Args:
            attack_type: Type of attack ('fgsm', 'pgd', 'cw', 'deepfool', 'autoattack').
            epsilon: Perturbation strength.
            model: PyTorch model to attack.
            device: Device to use.
            **kwargs: Additional arguments for the attack.
        """
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.model = model
        self.device = self._get_device(device)
        self.kwargs = kwargs
        
        if model is not None:
            self.model.to(self.device)
            self._setup_attack()
    
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
    
    def _setup_attack(self) -> None:
        """Setup the attack based on the attack type."""
        if self.attack_type == "fgsm":
            self.attack = FGSM(self.model, eps=self.epsilon)
        elif self.attack_type == "pgd":
            self.attack = PGD(
                self.model,
                eps=self.epsilon,
                alpha=self.kwargs.get("alpha", self.epsilon / 4),
                steps=self.kwargs.get("steps", 10),
            )
        elif self.attack_type == "cw":
            self.attack = CW(
                self.model,
                c=self.kwargs.get("c", 1.0),
                kappa=self.kwargs.get("kappa", 0),
                steps=self.kwargs.get("steps", 1000),
            )
        elif self.attack_type == "deepfool":
            self.attack = DeepFool(
                self.model,
                steps=self.kwargs.get("steps", 50),
            )
        elif self.attack_type == "autoattack":
            self.attack = AutoAttack(
                self.model,
                eps=self.epsilon,
                version=self.kwargs.get("version", "standard"),
            )
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
    
    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate adversarial examples."""
        if self.model is None:
            raise ValueError("Model must be provided for PyTorch attacks")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Generate adversarial examples
        X_adv_tensor = self.attack(X_tensor, y_tensor)
        
        # Convert back to numpy
        X_adv = X_adv_tensor.cpu().numpy()
        
        return X_adv
    
    def get_name(self) -> str:
        """Get the name of the attack."""
        return f"PyTorch_{self.attack_type}_{self.epsilon}"


class GradientBasedAttack(BaseAttack):
    """Gradient-based attacks for models with gradient access."""
    
    def __init__(
        self,
        model: Any,
        epsilon: float = 0.1,
        attack_type: str = "fgsm",
        **kwargs: Any,
    ) -> None:
        """Initialize the gradient-based attack.
        
        Args:
            model: Model to attack (must support gradient computation).
            epsilon: Perturbation strength.
            attack_type: Type of attack ('fgsm', 'pgd').
            **kwargs: Additional arguments.
        """
        self.model = model
        self.epsilon = epsilon
        self.attack_type = attack_type
        self.kwargs = kwargs
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradients of the loss with respect to inputs."""
        # This is a simplified implementation
        # In practice, you'd use autograd or similar
        X_tensor = torch.FloatTensor(X, requires_grad=True)
        y_tensor = torch.LongTensor(y)
        
        # Forward pass
        outputs = self.model(X_tensor)
        loss = nn.CrossEntropyLoss()(outputs, y_tensor)
        
        # Backward pass
        loss.backward()
        
        return X_tensor.grad.numpy()
    
    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate adversarial examples using gradients."""
        gradients = self._compute_gradients(X, y)
        
        if self.attack_type == "fgsm":
            perturbation = self.epsilon * np.sign(gradients)
        elif self.attack_type == "pgd":
            # Simplified PGD implementation
            alpha = self.kwargs.get("alpha", self.epsilon / 4)
            steps = self.kwargs.get("steps", 10)
            
            X_adv = X.copy()
            for _ in range(steps):
                gradients = self._compute_gradients(X_adv, y)
                X_adv = X_adv + alpha * np.sign(gradients)
                X_adv = np.clip(X_adv, X - self.epsilon, X + self.epsilon)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
        
        X_adv = X + perturbation
        return X_adv
    
    def get_name(self) -> str:
        """Get the name of the attack."""
        return f"Gradient_{self.attack_type}_{self.epsilon}"


class AttackFactory:
    """Factory for creating different types of attacks."""
    
    @staticmethod
    def create_attack(
        attack_type: str,
        **kwargs: Any,
    ) -> BaseAttack:
        """Create an attack instance.
        
        Args:
            attack_type: Type of attack to create.
            **kwargs: Additional arguments for attack creation.
            
        Returns:
            Attack instance.
        """
        if attack_type in ["gaussian", "uniform", "fgsm_like"]:
            return SimpleAttack(
                attack_type=attack_type,
                epsilon=kwargs.get("epsilon", 0.1),
                random_state=kwargs.get("random_state", None),
            )
        
        elif attack_type in ["fgsm", "pgd", "cw", "deepfool", "autoattack"]:
            return PyTorchAttack(
                attack_type=attack_type,
                epsilon=kwargs.get("epsilon", 0.1),
                model=kwargs.get("model", None),
                device=kwargs.get("device", None),
                **kwargs,
            )
        
        elif attack_type in ["gradient_fgsm", "gradient_pgd"]:
            return GradientBasedAttack(
                model=kwargs.get("model"),
                epsilon=kwargs.get("epsilon", 0.1),
                attack_type=attack_type.replace("gradient_", ""),
                **kwargs,
            )
        
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")


class AdversarialAttacker:
    """Main class for performing adversarial attacks."""
    
    def __init__(self, model: Any, model_name: str = "model") -> None:
        """Initialize the adversarial attacker.
        
        Args:
            model: Model to attack.
            model_name: Name of the model for logging.
        """
        self.model = model
        self.model_name = model_name
        self.attack_results: Dict[str, Dict[str, Any]] = {}
    
    def perform_attack(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        attack: BaseAttack,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Perform an adversarial attack.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            attack: Attack instance to use.
            verbose: Whether to print results.
            
        Returns:
            Dictionary containing attack results.
        """
        attack_name = attack.get_name()
        
        if verbose:
            logger.info(f"Performing {attack_name} attack on {self.model_name}")
        
        # Generate adversarial examples
        X_adv = attack.generate(X_test, y_test)
        
        # Evaluate on original and adversarial data
        y_pred_orig = self.model.predict(X_test)
        y_pred_adv = self.model.predict(X_adv)
        
        # Calculate metrics
        orig_accuracy = np.mean(y_pred_orig == y_test)
        adv_accuracy = np.mean(y_pred_adv == y_test)
        
        # Calculate perturbation statistics
        perturbation = X_adv - X_test
        l2_norm = np.linalg.norm(perturbation, axis=1)
        linf_norm = np.max(np.abs(perturbation), axis=1)
        
        results = {
            "attack_name": attack_name,
            "original_accuracy": orig_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "accuracy_drop": orig_accuracy - adv_accuracy,
            "perturbation_l2_mean": np.mean(l2_norm),
            "perturbation_l2_std": np.std(l2_norm),
            "perturbation_linf_mean": np.mean(linf_norm),
            "perturbation_linf_std": np.std(linf_norm),
            "n_samples": len(X_test),
            "X_adv": X_adv,
            "perturbation": perturbation,
        }
        
        # Store results
        self.attack_results[attack_name] = results
        
        if verbose:
            logger.info(f"Original accuracy: {orig_accuracy:.4f}")
            logger.info(f"Adversarial accuracy: {adv_accuracy:.4f}")
            logger.info(f"Accuracy drop: {results['accuracy_drop']:.4f}")
            logger.info(f"Mean L2 perturbation: {results['perturbation_l2_mean']:.4f}")
            logger.info(f"Mean L∞ perturbation: {results['perturbation_linf_mean']:.4f}")
        
        return results
    
    def perform_multiple_attacks(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        attacks: List[BaseAttack],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Perform multiple adversarial attacks.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            attacks: List of attack instances.
            verbose: Whether to print results.
            
        Returns:
            Dictionary containing results for all attacks.
        """
        all_results = {}
        
        for attack in attacks:
            results = self.perform_attack(X_test, y_test, attack, verbose)
            all_results[results["attack_name"]] = results
        
        return all_results
    
    def get_attack_summary(self) -> Dict[str, float]:
        """Get a summary of all performed attacks.
        
        Returns:
            Dictionary with summary statistics.
        """
        if not self.attack_results:
            return {}
        
        summary = {}
        for attack_name, results in self.attack_results.items():
            summary[f"{attack_name}_accuracy_drop"] = results["accuracy_drop"]
            summary[f"{attack_name}_perturbation_l2"] = results["perturbation_l2_mean"]
            summary[f"{attack_name}_perturbation_linf"] = results["perturbation_linf_mean"]
        
        return summary
