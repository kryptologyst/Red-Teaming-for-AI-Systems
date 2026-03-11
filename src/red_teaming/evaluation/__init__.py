"""Evaluation metrics and robustness assessment for red teaming experiments."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for various robustness and performance metrics."""
    
    @staticmethod
    def calculate_basic_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate basic classification metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).
            
        Returns:
            Dictionary of metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        
        if y_proba is not None:
            # Calculate confidence metrics
            max_proba = np.max(y_proba, axis=1)
            metrics.update({
                "mean_confidence": np.mean(max_proba),
                "std_confidence": np.std(max_proba),
                "min_confidence": np.min(max_proba),
                "max_confidence": np.max(max_proba),
            })
        
        return metrics
    
    @staticmethod
    def calculate_robustness_metrics(
        y_true: np.ndarray,
        y_pred_orig: np.ndarray,
        y_pred_adv: np.ndarray,
        X_orig: np.ndarray,
        X_adv: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate robustness-specific metrics.
        
        Args:
            y_true: True labels.
            y_pred_orig: Predictions on original data.
            y_pred_adv: Predictions on adversarial data.
            X_orig: Original features.
            X_adv: Adversarial features.
            
        Returns:
            Dictionary of robustness metrics.
        """
        # Accuracy metrics
        orig_accuracy = accuracy_score(y_true, y_pred_orig)
        adv_accuracy = accuracy_score(y_true, y_pred_adv)
        
        # Perturbation metrics
        perturbation = X_adv - X_orig
        l2_norms = np.linalg.norm(perturbation, axis=1)
        linf_norms = np.max(np.abs(perturbation), axis=1)
        
        # Robustness metrics
        accuracy_drop = orig_accuracy - adv_accuracy
        relative_accuracy_drop = accuracy_drop / orig_accuracy if orig_accuracy > 0 else 0
        
        # Attack success rate (how many predictions changed)
        prediction_changes = np.sum(y_pred_orig != y_pred_adv)
        attack_success_rate = prediction_changes / len(y_true)
        
        # Robust accuracy (accuracy on adversarial examples)
        robust_accuracy = adv_accuracy
        
        metrics = {
            "original_accuracy": orig_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "accuracy_drop": accuracy_drop,
            "relative_accuracy_drop": relative_accuracy_drop,
            "attack_success_rate": attack_success_rate,
            "robust_accuracy": robust_accuracy,
            "perturbation_l2_mean": np.mean(l2_norms),
            "perturbation_l2_std": np.std(l2_norms),
            "perturbation_l2_max": np.max(l2_norms),
            "perturbation_linf_mean": np.mean(linf_norms),
            "perturbation_linf_std": np.std(linf_norms),
            "perturbation_linf_max": np.max(linf_norms),
        }
        
        return metrics
    
    @staticmethod
    def calculate_calibration_metrics(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for calibration curve.
            
        Returns:
            Dictionary of calibration metrics.
        """
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {
            "ece": ece,
            "mce": mce,
        }


class RobustnessEvaluator:
    """Main evaluator for robustness assessment."""
    
    def __init__(self, model: Any, model_name: str = "model") -> None:
        """Initialize the robustness evaluator.
        
        Args:
            model: Model to evaluate.
            model_name: Name of the model for logging.
        """
        self.model = model
        self.model_name = model_name
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
    
    def evaluate_baseline_performance(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate baseline model performance.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            verbose: Whether to print results.
            
        Returns:
            Dictionary of baseline metrics.
        """
        logger.info(f"Evaluating baseline performance for {self.model_name}")
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        metrics = MetricsCalculator.calculate_basic_metrics(y_test, y_pred, y_proba)
        
        if verbose:
            logger.info(f"Baseline Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Baseline F1 (macro): {metrics['f1_macro']:.4f}")
            logger.info(f"Mean Confidence: {metrics.get('mean_confidence', 0):.4f}")
        
        self.evaluation_results["baseline"] = metrics
        return metrics
    
    def evaluate_adversarial_robustness(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_adv: np.ndarray,
        attack_name: str,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate adversarial robustness.
        
        Args:
            X_test: Original test features.
            y_test: Test labels.
            X_adv: Adversarial features.
            attack_name: Name of the attack.
            verbose: Whether to print results.
            
        Returns:
            Dictionary of robustness metrics.
        """
        logger.info(f"Evaluating adversarial robustness for {self.model_name} against {attack_name}")
        
        y_pred_orig = self.model.predict(X_test)
        y_pred_adv = self.model.predict(X_adv)
        
        metrics = MetricsCalculator.calculate_robustness_metrics(
            y_test, y_pred_orig, y_pred_adv, X_test, X_adv
        )
        
        if verbose:
            logger.info(f"Original Accuracy: {metrics['original_accuracy']:.4f}")
            logger.info(f"Adversarial Accuracy: {metrics['adversarial_accuracy']:.4f}")
            logger.info(f"Accuracy Drop: {metrics['accuracy_drop']:.4f}")
            logger.info(f"Attack Success Rate: {metrics['attack_success_rate']:.4f}")
            logger.info(f"Mean L2 Perturbation: {metrics['perturbation_l2_mean']:.4f}")
        
        self.evaluation_results[f"adversarial_{attack_name}"] = metrics
        return metrics
    
    def evaluate_calibration(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate model calibration.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            verbose: Whether to print results.
            
        Returns:
            Dictionary of calibration metrics.
        """
        logger.info(f"Evaluating calibration for {self.model_name}")
        
        y_proba = self.model.predict_proba(X_test)
        max_proba = np.max(y_proba, axis=1)
        
        # Convert to binary for calibration curve
        y_pred = self.model.predict(X_test)
        y_binary = (y_pred == y_test).astype(int)
        
        metrics = MetricsCalculator.calculate_calibration_metrics(y_binary, max_proba)
        
        if verbose:
            logger.info(f"Expected Calibration Error: {metrics['ece']:.4f}")
            logger.info(f"Maximum Calibration Error: {metrics['mce']:.4f}")
        
        self.evaluation_results["calibration"] = metrics
        return metrics
    
    def create_robustness_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        adversarial_results: Dict[str, np.ndarray],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Create a comprehensive robustness report.
        
        Args:
            X_test: Original test features.
            y_test: Test labels.
            adversarial_results: Dictionary mapping attack names to adversarial features.
            verbose: Whether to print results.
            
        Returns:
            Comprehensive robustness report.
        """
        logger.info(f"Creating comprehensive robustness report for {self.model_name}")
        
        report = {
            "model_name": self.model_name,
            "n_test_samples": len(y_test),
            "baseline_performance": {},
            "adversarial_robustness": {},
            "calibration": {},
            "summary": {},
        }
        
        # Baseline performance
        baseline_metrics = self.evaluate_baseline_performance(X_test, y_test, verbose)
        report["baseline_performance"] = baseline_metrics
        
        # Adversarial robustness
        for attack_name, X_adv in adversarial_results.items():
            adv_metrics = self.evaluate_adversarial_robustness(
                X_test, y_test, X_adv, attack_name, verbose
            )
            report["adversarial_robustness"][attack_name] = adv_metrics
        
        # Calibration
        cal_metrics = self.evaluate_calibration(X_test, y_test, verbose)
        report["calibration"] = cal_metrics
        
        # Summary statistics
        accuracy_drops = [
            metrics["accuracy_drop"]
            for metrics in report["adversarial_robustness"].values()
        ]
        
        report["summary"] = {
            "mean_accuracy_drop": np.mean(accuracy_drops) if accuracy_drops else 0,
            "max_accuracy_drop": np.max(accuracy_drops) if accuracy_drops else 0,
            "n_attacks_tested": len(adversarial_results),
            "baseline_accuracy": baseline_metrics["accuracy"],
            "worst_adversarial_accuracy": min(
                metrics["adversarial_accuracy"]
                for metrics in report["adversarial_robustness"].values()
            ) if report["adversarial_robustness"] else baseline_metrics["accuracy"],
        }
        
        if verbose:
            logger.info("=== ROBUSTNESS REPORT SUMMARY ===")
            logger.info(f"Baseline Accuracy: {report['summary']['baseline_accuracy']:.4f}")
            logger.info(f"Worst Adversarial Accuracy: {report['summary']['worst_adversarial_accuracy']:.4f}")
            logger.info(f"Mean Accuracy Drop: {report['summary']['mean_accuracy_drop']:.4f}")
            logger.info(f"Max Accuracy Drop: {report['summary']['max_accuracy_drop']:.4f}")
            logger.info(f"Attacks Tested: {report['summary']['n_attacks_tested']}")
        
        return report
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """Get a summary of all evaluations as a DataFrame.
        
        Returns:
            DataFrame with evaluation results.
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        summary_data = []
        for eval_name, metrics in self.evaluation_results.items():
            row = {"evaluation": eval_name}
            row.update(metrics)
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save_evaluation_results(self, filepath: str) -> None:
        """Save evaluation results to a file.
        
        Args:
            filepath: Path to save the results.
        """
        summary_df = self.get_evaluation_summary()
        summary_df.to_csv(filepath, index=False)
        logger.info(f"Evaluation results saved to {filepath}")


class RobustnessLeaderboard:
    """Leaderboard for comparing model robustness across different attacks."""
    
    def __init__(self) -> None:
        """Initialize the robustness leaderboard."""
        self.results: List[Dict[str, Any]] = []
    
    def add_model_results(
        self,
        model_name: str,
        robustness_report: Dict[str, Any],
    ) -> None:
        """Add model results to the leaderboard.
        
        Args:
            model_name: Name of the model.
            robustness_report: Robustness report from RobustnessEvaluator.
        """
        entry = {
            "model_name": model_name,
            "baseline_accuracy": robustness_report["baseline_performance"]["accuracy"],
            "worst_adversarial_accuracy": robustness_report["summary"]["worst_adversarial_accuracy"],
            "mean_accuracy_drop": robustness_report["summary"]["mean_accuracy_drop"],
            "max_accuracy_drop": robustness_report["summary"]["max_accuracy_drop"],
            "n_attacks_tested": robustness_report["summary"]["n_attacks_tested"],
            "ece": robustness_report["calibration"]["ece"],
            "mce": robustness_report["calibration"]["mce"],
        }
        
        # Add individual attack results
        for attack_name, metrics in robustness_report["adversarial_robustness"].items():
            entry[f"{attack_name}_accuracy_drop"] = metrics["accuracy_drop"]
            entry[f"{attack_name}_perturbation_l2"] = metrics["perturbation_l2_mean"]
        
        self.results.append(entry)
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get the current leaderboard as a DataFrame.
        
        Returns:
            DataFrame with leaderboard results.
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        # Sort by robustness (higher adversarial accuracy is better)
        df = df.sort_values("worst_adversarial_accuracy", ascending=False)
        
        return df
    
    def print_leaderboard(self) -> None:
        """Print the current leaderboard."""
        df = self.get_leaderboard()
        
        if df.empty:
            print("No results in leaderboard yet.")
            return
        
        print("\n=== ROBUSTNESS LEADERBOARD ===")
        print(df[["model_name", "baseline_accuracy", "worst_adversarial_accuracy", 
                "mean_accuracy_drop", "max_accuracy_drop"]].to_string(index=False))
    
    def save_leaderboard(self, filepath: str) -> None:
        """Save the leaderboard to a file.
        
        Args:
            filepath: Path to save the leaderboard.
        """
        df = self.get_leaderboard()
        df.to_csv(filepath, index=False)
        logger.info(f"Leaderboard saved to {filepath}")
