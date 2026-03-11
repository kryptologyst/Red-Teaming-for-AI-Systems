"""Tests for the red teaming package."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from red_teaming.data import DataLoader, SyntheticDataGenerator
from red_teaming.models import ModelFactory, RobustModel
from red_teaming.attacks import AttackFactory, AdversarialAttacker
from red_teaming.evaluation import RobustnessEvaluator, MetricsCalculator
from red_teaming.utils import set_seed, validate_data, normalize_features


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_load_iris_dataset(self):
        """Test loading Iris dataset."""
        loader = DataLoader(random_state=42)
        X, y, feature_names, target_names = loader.load_iris_dataset()
        
        assert X.shape == (150, 4)
        assert len(y) == 150
        assert len(feature_names) == 4
        assert len(target_names) == 3
        assert np.all(np.unique(y) == [0, 1, 2])
    
    def test_load_synthetic_dataset(self):
        """Test loading synthetic dataset."""
        loader = DataLoader(random_state=42)
        X, y, feature_names, target_names = loader.load_synthetic_dataset(
            n_samples=100, n_features=5, n_classes=2
        )
        
        assert X.shape == (100, 5)
        assert len(y) == 100
        assert len(feature_names) == 5
        assert len(target_names) == 2
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        loader = DataLoader(random_state=42)
        X, y, _, _ = loader.load_iris_dataset()
        
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y, test_size=0.3)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert X_train.shape[1] == X_test.shape[1]
    
    def test_save_dataset_metadata(self, tmp_path):
        """Test saving dataset metadata."""
        loader = DataLoader(random_state=42)
        feature_names = ["feat1", "feat2"]
        target_names = ["class1", "class2"]
        
        output_path = tmp_path / "metadata.json"
        loader.save_dataset_metadata(feature_names, target_names, output_path)
        
        assert output_path.exists()


class TestSyntheticDataGenerator:
    """Test cases for SyntheticDataGenerator class."""
    
    def test_generate_adversarial_test_set(self):
        """Test generating adversarial test set."""
        generator = SyntheticDataGenerator(random_state=42)
        X_test = np.random.randn(50, 4)
        y_test = np.random.randint(0, 3, 50)
        
        X_adv, y_adv = generator.generate_adversarial_test_set(
            X_test, y_test, epsilon=0.1, attack_type="gaussian"
        )
        
        assert X_adv.shape == X_test.shape
        assert np.array_equal(y_adv, y_test)
        assert not np.array_equal(X_adv, X_test)
    
    def test_generate_out_of_distribution_samples(self):
        """Test generating OOD samples."""
        generator = SyntheticDataGenerator(random_state=42)
        X_train = np.random.randn(100, 4)
        
        X_ood = generator.generate_out_of_distribution_samples(X_train, n_samples=20)
        
        assert X_ood.shape == (20, 4)


class TestModelFactory:
    """Test cases for ModelFactory class."""
    
    def test_create_random_forest(self):
        """Test creating Random Forest model."""
        model = ModelFactory.create_model("random_forest", random_state=42)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_create_logistic_regression(self):
        """Test creating Logistic Regression model."""
        model = ModelFactory.create_model("logistic_regression", random_state=42)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_create_neural_network(self):
        """Test creating Neural Network model."""
        model = ModelFactory.create_model(
            "neural_network", 
            input_dim=4, 
            num_classes=3,
            epochs=1  # Short training for test
        )
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_invalid_model_type(self):
        """Test creating invalid model type."""
        with pytest.raises(ValueError):
            ModelFactory.create_model("invalid_model")


class TestRobustModel:
    """Test cases for RobustModel class."""
    
    def test_train_and_evaluate(self):
        """Test training and evaluating a model."""
        # Create a simple dataset
        X_train = np.random.randn(100, 4)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(30, 4)
        y_test = np.random.randint(0, 3, 30)
        
        # Create and train model
        base_model = ModelFactory.create_model("random_forest", random_state=42)
        robust_model = RobustModel(base_model, "test_model")
        
        robust_model.train(X_train, y_train)
        metrics = robust_model.evaluate(X_test, y_test, verbose=False)
        
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0
        assert metrics["accuracy"] <= 1


class TestAttackFactory:
    """Test cases for AttackFactory class."""
    
    def test_create_gaussian_attack(self):
        """Test creating Gaussian attack."""
        attack = AttackFactory.create_attack("gaussian", epsilon=0.1)
        assert hasattr(attack, 'generate')
        assert hasattr(attack, 'get_name')
    
    def test_create_uniform_attack(self):
        """Test creating Uniform attack."""
        attack = AttackFactory.create_attack("uniform", epsilon=0.1)
        assert hasattr(attack, 'generate')
        assert hasattr(attack, 'get_name')
    
    def test_create_fgsm_like_attack(self):
        """Test creating FGSM-like attack."""
        attack = AttackFactory.create_attack("fgsm_like", epsilon=0.1)
        assert hasattr(attack, 'generate')
        assert hasattr(attack, 'get_name')
    
    def test_invalid_attack_type(self):
        """Test creating invalid attack type."""
        with pytest.raises(ValueError):
            AttackFactory.create_attack("invalid_attack")


class TestAdversarialAttacker:
    """Test cases for AdversarialAttacker class."""
    
    def test_perform_attack(self):
        """Test performing an adversarial attack."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.random.randint(0, 3, 50)
        
        attacker = AdversarialAttacker(mock_model, "test_model")
        attack = AttackFactory.create_attack("gaussian", epsilon=0.1)
        
        X_test = np.random.randn(50, 4)
        y_test = np.random.randint(0, 3, 50)
        
        result = attacker.perform_attack(X_test, y_test, attack, verbose=False)
        
        assert "original_accuracy" in result
        assert "adversarial_accuracy" in result
        assert "accuracy_drop" in result
        assert "X_adv" in result


class TestRobustnessEvaluator:
    """Test cases for RobustnessEvaluator class."""
    
    def test_evaluate_baseline_performance(self):
        """Test evaluating baseline performance."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.random.randint(0, 3, 50)
        mock_model.predict_proba.return_value = np.random.rand(50, 3)
        
        evaluator = RobustnessEvaluator(mock_model, "test_model")
        
        X_test = np.random.randn(50, 4)
        y_test = np.random.randint(0, 3, 50)
        
        metrics = evaluator.evaluate_baseline_performance(X_test, y_test, verbose=False)
        
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0
        assert metrics["accuracy"] <= 1


class TestMetricsCalculator:
    """Test cases for MetricsCalculator class."""
    
    def test_calculate_basic_metrics(self):
        """Test calculating basic metrics."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        y_proba = np.random.rand(5, 3)
        
        metrics = MetricsCalculator.calculate_basic_metrics(y_true, y_pred, y_proba)
        
        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "mean_confidence" in metrics
    
    def test_calculate_robustness_metrics(self):
        """Test calculating robustness metrics."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred_orig = np.array([0, 1, 2, 0, 1])
        y_pred_adv = np.array([0, 1, 1, 0, 2])
        X_orig = np.random.randn(5, 4)
        X_adv = X_orig + 0.1 * np.random.randn(5, 4)
        
        metrics = MetricsCalculator.calculate_robustness_metrics(
            y_true, y_pred_orig, y_pred_adv, X_orig, X_adv
        )
        
        assert "original_accuracy" in metrics
        assert "adversarial_accuracy" in metrics
        assert "accuracy_drop" in metrics
        assert "attack_success_rate" in metrics
        assert "perturbation_l2_mean" in metrics


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test setting random seed."""
        set_seed(42)
        # This is hard to test directly, but we can check it doesn't raise an error
        assert True
    
    def test_validate_data(self):
        """Test data validation."""
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)
        feature_names = ["feat1", "feat2", "feat3", "feat4"]
        
        assert validate_data(X, y, feature_names) == True
        
        # Test with invalid data
        X_invalid = np.random.randn(50, 4)  # Different length
        assert validate_data(X_invalid, y, feature_names) == False
    
    def test_normalize_features(self):
        """Test feature normalization."""
        X = np.random.randn(100, 4)
        
        X_norm = normalize_features(X, method="standard", fit_transform=True)
        
        assert X_norm.shape == X.shape
        assert np.allclose(np.mean(X_norm, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_norm, axis=0), 1, atol=1e-10)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_experiment(self):
        """Test a complete end-to-end experiment."""
        # Set seed for reproducibility
        set_seed(42)
        
        # Load data
        loader = DataLoader(random_state=42)
        X, y, feature_names, target_names = loader.load_iris_dataset()
        X_train, X_test, y_train, y_test = loader.preprocess_data(X, y, test_size=0.3)
        
        # Train model
        base_model = ModelFactory.create_model("random_forest", random_state=42)
        robust_model = RobustModel(base_model, "test_model")
        robust_model.train(X_train, y_train)
        
        # Perform attack
        attack = AttackFactory.create_attack("gaussian", epsilon=0.1)
        attacker = AdversarialAttacker(robust_model.model, "test_model")
        result = attacker.perform_attack(X_test, y_test, attack, verbose=False)
        
        # Evaluate robustness
        evaluator = RobustnessEvaluator(robust_model.model, "test_model")
        adversarial_results = {"gaussian": result["X_adv"]}
        robustness_report = evaluator.create_robustness_report(
            X_test, y_test, adversarial_results, verbose=False
        )
        
        # Check that we got reasonable results
        assert robustness_report["summary"]["baseline_accuracy"] > 0.8
        assert robustness_report["summary"]["mean_accuracy_drop"] >= 0
        assert robustness_report["summary"]["n_attacks_tested"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
