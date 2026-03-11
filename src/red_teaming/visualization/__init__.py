"""Visualization utilities for red teaming and robustness analysis."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class RobustnessVisualizer:
    """Main class for visualizing robustness analysis results."""
    
    def __init__(self, output_dir: Union[str, Path] = "assets") -> None:
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_accuracy_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Accuracy Comparison",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot accuracy comparison between original and adversarial data.
        
        Args:
            results: Dictionary mapping attack names to metrics.
            title: Plot title.
            save_path: Path to save the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        attacks = list(results.keys())
        orig_acc = [results[attack]["original_accuracy"] for attack in attacks]
        adv_acc = [results[attack]["adversarial_accuracy"] for attack in attacks]
        
        x = np.arange(len(attacks))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, orig_acc, width, label="Original", alpha=0.8)
        bars2 = ax.bar(x + width/2, adv_acc, width, label="Adversarial", alpha=0.8)
        
        ax.set_xlabel("Attack Type")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Accuracy comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_perturbation_analysis(
        self,
        X_orig: np.ndarray,
        X_adv: np.ndarray,
        attack_name: str,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot perturbation analysis.
        
        Args:
            X_orig: Original features.
            X_adv: Adversarial features.
            attack_name: Name of the attack.
            save_path: Path to save the plot.
        """
        perturbation = X_adv - X_orig
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Perturbation Analysis - {attack_name}", fontsize=16)
        
        # L2 norms
        l2_norms = np.linalg.norm(perturbation, axis=1)
        axes[0, 0].hist(l2_norms, bins=30, alpha=0.7, edgecolor="black")
        axes[0, 0].set_xlabel("L2 Norm")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Distribution of L2 Perturbation Norms")
        axes[0, 0].grid(True, alpha=0.3)
        
        # L∞ norms
        linf_norms = np.max(np.abs(perturbation), axis=1)
        axes[0, 1].hist(linf_norms, bins=30, alpha=0.7, edgecolor="black")
        axes[0, 1].set_xlabel("L∞ Norm")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of L∞ Perturbation Norms")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Perturbation magnitude per feature
        pert_magnitude = np.mean(np.abs(perturbation), axis=0)
        feature_indices = range(len(pert_magnitude))
        axes[1, 0].bar(feature_indices, pert_magnitude, alpha=0.7)
        axes[1, 0].set_xlabel("Feature Index")
        axes[1, 0].set_ylabel("Mean Absolute Perturbation")
        axes[1, 0].set_title("Perturbation Magnitude per Feature")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Perturbation vs original feature values
        if X_orig.shape[1] >= 2:
            axes[1, 1].scatter(X_orig[:, 0], X_orig[:, 1], alpha=0.6, label="Original", s=20)
            axes[1, 1].scatter(X_adv[:, 0], X_adv[:, 1], alpha=0.6, label="Adversarial", s=20)
            axes[1, 1].set_xlabel("Feature 0")
            axes[1, 1].set_ylabel("Feature 1")
            axes[1, 1].set_title("Original vs Adversarial (First 2 Features)")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Perturbation analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_robustness_curves(
        self,
        epsilon_values: List[float],
        accuracy_values: List[float],
        attack_name: str,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot robustness curves showing accuracy vs perturbation strength.
        
        Args:
            epsilon_values: List of epsilon values.
            accuracy_values: List of corresponding accuracy values.
            attack_name: Name of the attack.
            save_path: Path to save the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epsilon_values, accuracy_values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel("Perturbation Strength (ε)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Robustness Curve - {attack_name}")
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key points
        max_acc_idx = np.argmax(accuracy_values)
        min_acc_idx = np.argmin(accuracy_values)
        
        ax.annotate(f'Max: {accuracy_values[max_acc_idx]:.3f}',
                   xy=(epsilon_values[max_acc_idx], accuracy_values[max_acc_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate(f'Min: {accuracy_values[min_acc_idx]:.3f}',
                   xy=(epsilon_values[min_acc_idx], accuracy_values[min_acc_idx]),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Robustness curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(
        self,
        y_true: np.ndarray,
        y_pred_orig: np.ndarray,
        y_pred_adv: np.ndarray,
        class_names: List[str],
        attack_name: str,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot confusion matrices for original and adversarial predictions.
        
        Args:
            y_true: True labels.
            y_pred_orig: Predictions on original data.
            y_pred_adv: Predictions on adversarial data.
            class_names: Names of the classes.
            attack_name: Name of the attack.
            save_path: Path to save the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Confusion Matrices - {attack_name}", fontsize=16)
        
        # Original confusion matrix
        cm_orig = confusion_matrix(y_true, y_pred_orig)
        sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title("Original Data")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        
        # Adversarial confusion matrix
        cm_adv = confusion_matrix(y_true, y_pred_adv)
        sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title("Adversarial Data")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrices plot saved to {save_path}")
        
        plt.show()
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot calibration curve.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for calibration curve.
            save_path: Path to save the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Convert to binary for calibration curve
        y_pred = np.argmax(y_proba, axis=1)
        y_binary = (y_pred == y_true).astype(int)
        max_proba = np.max(y_proba, axis=1)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, max_proba, n_bins=n_bins
        )
        
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
               label="Model", linewidth=2, markersize=6)
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Calibration curve plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance_robustness(
        self,
        feature_names: List[str],
        importance_scores: List[float],
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature importance scores.
        
        Args:
            feature_names: Names of the features.
            importance_scores: Importance scores for each feature.
            title: Plot title.
            save_path: Path to save the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_scores = [importance_scores[i] for i in sorted_indices]
        
        bars = ax.barh(range(len(sorted_names)), sorted_scores, alpha=0.7)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel("Importance Score")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_leaderboard(
        self,
        leaderboard_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot the robustness leaderboard.
        
        Args:
            leaderboard_df: DataFrame with leaderboard results.
            save_path: Path to save the plot.
        """
        if leaderboard_df.empty:
            print("No data to plot in leaderboard.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Robustness Leaderboard Analysis", fontsize=16)
        
        # Baseline vs Adversarial Accuracy
        axes[0, 0].scatter(leaderboard_df["baseline_accuracy"], 
                          leaderboard_df["worst_adversarial_accuracy"],
                          s=100, alpha=0.7)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel("Baseline Accuracy")
        axes[0, 0].set_ylabel("Worst Adversarial Accuracy")
        axes[0, 0].set_title("Baseline vs Adversarial Accuracy")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add model names as annotations
        for i, row in leaderboard_df.iterrows():
            axes[0, 0].annotate(row["model_name"], 
                               (row["baseline_accuracy"], row["worst_adversarial_accuracy"]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Accuracy Drop Distribution
        axes[0, 1].hist(leaderboard_df["mean_accuracy_drop"], bins=10, alpha=0.7, edgecolor="black")
        axes[0, 1].set_xlabel("Mean Accuracy Drop")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Mean Accuracy Drops")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calibration Error
        axes[1, 0].scatter(leaderboard_df["ece"], leaderboard_df["mce"], s=100, alpha=0.7)
        axes[1, 0].set_xlabel("Expected Calibration Error")
        axes[1, 0].set_ylabel("Maximum Calibration Error")
        axes[1, 0].set_title("Calibration Analysis")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Model Ranking
        model_names = leaderboard_df["model_name"]
        adversarial_acc = leaderboard_df["worst_adversarial_accuracy"]
        
        y_pos = np.arange(len(model_names))
        axes[1, 1].barh(y_pos, adversarial_acc, alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(model_names)
        axes[1, 1].set_xlabel("Worst Adversarial Accuracy")
        axes[1, 1].set_title("Model Robustness Ranking")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Leaderboard plot saved to {save_path}")
        
        plt.show()
    
    def create_summary_report(
        self,
        robustness_report: Dict[str, Any],
        save_dir: Optional[str] = None,
    ) -> None:
        """Create a comprehensive summary report with multiple visualizations.
        
        Args:
            robustness_report: Comprehensive robustness report.
            save_dir: Directory to save all plots.
        """
        if save_dir:
            self.output_dir = Path(save_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = robustness_report["model_name"]
        
        # Plot accuracy comparison
        if robustness_report["adversarial_robustness"]:
            self.plot_accuracy_comparison(
                robustness_report["adversarial_robustness"],
                title=f"Accuracy Comparison - {model_name}",
                save_path=f"{model_name}_accuracy_comparison.png"
            )
        
        # Plot calibration curve
        self.plot_calibration_curve(
            np.array([]),  # Would need actual data
            np.array([]),  # Would need actual data
            save_path=f"{model_name}_calibration_curve.png"
        )
        
        logger.info(f"Summary report created for {model_name}")
    
    def save_all_plots(self, prefix: str = "plot") -> None:
        """Save all current plots with a common prefix.
        
        Args:
            prefix: Prefix for saved plot files.
        """
        logger.info(f"Saving all plots with prefix '{prefix}' to {self.output_dir}")
        # This would save all currently open plots
        # Implementation depends on specific needs
