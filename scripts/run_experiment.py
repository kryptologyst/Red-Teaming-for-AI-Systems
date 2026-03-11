#!/usr/bin/env python3
"""Main experiment script for red teaming and adversarial robustness testing."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from red_teaming.data import DataLoader, SyntheticDataGenerator
from red_teaming.models import ModelFactory, RobustModel
from red_teaming.attacks import AttackFactory, AdversarialAttacker
from red_teaming.evaluation import RobustnessEvaluator, RobustnessLeaderboard
from red_teaming.visualization import RobustnessVisualizer
from red_teaming.utils import (
    set_seed, setup_logging, load_config, create_experiment_directory,
    ExperimentTracker, validate_data
)

logger = logging.getLogger(__name__)


class RedTeamingExperiment:
    """Main class for conducting red teaming experiments."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the experiment.
        
        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.setup_experiment()
        
        # Initialize components
        self.data_loader = DataLoader(random_state=config["experiment"]["seed"])
        self.synthetic_generator = SyntheticDataGenerator(random_state=config["experiment"]["seed"])
        self.visualizer = RobustnessVisualizer(output_dir=self.exp_dir / "plots")
        self.leaderboard = RobustnessLeaderboard()
        self.tracker = ExperimentTracker(self.exp_dir / "logs" / "results.log")
        
        # Data storage
        self.datasets: Dict[str, Any] = {}
        self.models: Dict[str, RobustModel] = {}
        self.attack_results: Dict[str, Dict[str, Any]] = {}
    
    def setup_experiment(self) -> None:
        """Setup the experiment environment."""
        # Set random seed
        set_seed(self.config["experiment"]["seed"])
        
        # Setup logging
        setup_logging(
            level=self.config["experiment"]["log_level"],
            log_file=self.exp_dir / "logs" / "experiment.log"
        )
        
        logger.info("Red Teaming Experiment initialized")
        logger.info(f"Configuration: {self.config}")
    
    @property
    def exp_dir(self) -> Path:
        """Get experiment directory."""
        return create_experiment_directory(
            base_dir=self.config["experiment"]["output_dir"],
            experiment_name=self.config["experiment"]["name"]
        )
    
    def load_data(self) -> None:
        """Load and preprocess the dataset."""
        logger.info("Loading dataset")
        
        data_config = self.config["data"]
        
        if data_config["dataset"] == "iris":
            X, y, feature_names, target_names = self.data_loader.load_iris_dataset()
        elif data_config["dataset"] == "synthetic":
            synthetic_params = data_config["synthetic"]
            X, y, feature_names, target_names = self.data_loader.load_synthetic_dataset(
                n_samples=synthetic_params["n_samples"],
                n_features=synthetic_params["n_features"],
                n_informative=synthetic_params["n_informative"],
                n_redundant=synthetic_params["n_redundant"],
                n_classes=synthetic_params["n_classes"],
                n_clusters_per_class=synthetic_params["n_clusters_per_class"],
            )
        else:
            raise ValueError(f"Unknown dataset: {data_config['dataset']}")
        
        # Preprocess data
        X_train, X_test, y_train, y_test = self.data_loader.preprocess_data(
            X, y,
            test_size=data_config["test_size"],
            scale_features=data_config["scale_features"]
        )
        
        # Validate data
        validate_data(X_train, y_train, feature_names)
        validate_data(X_test, y_test, feature_names)
        
        # Store dataset information
        self.datasets = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
            "target_names": target_names,
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
        }
        
        # Save dataset metadata
        self.data_loader.save_dataset_metadata(
            feature_names, target_names,
            self.exp_dir / "data" / "metadata.json"
        )
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    def train_models(self) -> None:
        """Train all configured models."""
        logger.info("Training models")
        
        for model_config in self.config["models"]:
            model_name = model_config["name"]
            model_type = model_config["type"]
            model_params = model_config["params"].copy()
            
            # Update model parameters based on dataset
            if model_type == "neural_network":
                model_params["input_dim"] = self.datasets["n_features"]
                model_params["num_classes"] = self.datasets["n_classes"]
            
            # Create and train model
            base_model = ModelFactory.create_model(model_type, **model_params)
            robust_model = RobustModel(base_model, model_name)
            
            robust_model.train(
                self.datasets["X_train"],
                self.datasets["y_train"]
            )
            
            # Evaluate baseline performance
            baseline_metrics = robust_model.evaluate(
                self.datasets["X_test"],
                self.datasets["y_test"],
                verbose=self.config["experiment"]["verbose"]
            )
            
            self.models[model_name] = robust_model
            
            # Log baseline results
            self.tracker.log_result({
                "model": model_name,
                "type": "baseline",
                "metrics": baseline_metrics
            })
        
        logger.info(f"Trained {len(self.models)} models")
    
    def perform_attacks(self) -> None:
        """Perform adversarial attacks on all models."""
        logger.info("Performing adversarial attacks")
        
        for model_name, model in self.models.items():
            logger.info(f"Attacking model: {model_name}")
            
            attacker = AdversarialAttacker(model.model, model_name)
            attack_results = {}
            
            for attack_config in self.config["attacks"]:
                attack_name = attack_config["name"]
                attack_type = attack_config["type"]
                attack_params = attack_config.copy()
                attack_params.pop("name", None)
                
                # Create attack
                try:
                    attack = AttackFactory.create_attack(attack_type, **attack_params)
                    
                    # Perform attack
                    result = attacker.perform_attack(
                        self.datasets["X_test"],
                        self.datasets["y_test"],
                        attack,
                        verbose=self.config["experiment"]["verbose"]
                    )
                    
                    attack_results[attack_name] = result
                    
                    # Log attack results
                    self.tracker.log_result({
                        "model": model_name,
                        "type": "attack",
                        "attack": attack_name,
                        "metrics": {
                            "original_accuracy": result["original_accuracy"],
                            "adversarial_accuracy": result["adversarial_accuracy"],
                            "accuracy_drop": result["accuracy_drop"],
                            "attack_success_rate": result["attack_success_rate"],
                        }
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to perform {attack_name} on {model_name}: {e}")
                    continue
            
            self.attack_results[model_name] = attack_results
        
        logger.info("Adversarial attacks completed")
    
    def evaluate_robustness(self) -> None:
        """Evaluate robustness of all models."""
        logger.info("Evaluating model robustness")
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating robustness: {model_name}")
            
            evaluator = RobustnessEvaluator(model.model, model_name)
            
            # Create adversarial results dictionary
            adversarial_results = {}
            if model_name in self.attack_results:
                for attack_name, result in self.attack_results[model_name].items():
                    adversarial_results[attack_name] = result["X_adv"]
            
            # Create comprehensive robustness report
            robustness_report = evaluator.create_robustness_report(
                self.datasets["X_test"],
                self.datasets["y_test"],
                adversarial_results,
                verbose=self.config["experiment"]["verbose"]
            )
            
            # Add to leaderboard
            self.leaderboard.add_model_results(model_name, robustness_report)
            
            # Save detailed results
            if self.config["experiment"]["save_results"]:
                results_file = self.exp_dir / "results" / f"{model_name}_robustness_report.json"
                import json
                with open(results_file, 'w') as f:
                    json.dump(robustness_report, f, indent=2, default=str)
        
        # Print leaderboard
        if self.config["experiment"]["verbose"]:
            self.leaderboard.print_leaderboard()
        
        # Save leaderboard
        if self.config["experiment"]["save_results"]:
            leaderboard_file = self.exp_dir / "results" / "leaderboard.csv"
            self.leaderboard.save_leaderboard(leaderboard_file)
        
        logger.info("Robustness evaluation completed")
    
    def create_visualizations(self) -> None:
        """Create visualizations for the experiment."""
        if not self.config["visualization"]["save_plots"]:
            return
        
        logger.info("Creating visualizations")
        
        # Create visualizations for each model
        for model_name, model in self.models.items():
            if model_name not in self.attack_results:
                continue
            
            # Plot accuracy comparison
            self.visualizer.plot_accuracy_comparison(
                self.attack_results[model_name],
                title=f"Accuracy Comparison - {model_name}",
                save_path=f"{model_name}_accuracy_comparison.png"
            )
            
            # Plot perturbation analysis for each attack
            for attack_name, result in self.attack_results[model_name].items():
                self.visualizer.plot_perturbation_analysis(
                    self.datasets["X_test"],
                    result["X_adv"],
                    attack_name,
                    save_path=f"{model_name}_{attack_name}_perturbation.png"
                )
                
                # Plot confusion matrices
                y_pred_orig = model.predict(self.datasets["X_test"])
                y_pred_adv = model.predict(result["X_adv"])
                
                self.visualizer.plot_confusion_matrices(
                    self.datasets["y_test"],
                    y_pred_orig,
                    y_pred_adv,
                    self.datasets["target_names"],
                    attack_name,
                    save_path=f"{model_name}_{attack_name}_confusion.png"
                )
        
        # Plot leaderboard
        leaderboard_df = self.leaderboard.get_leaderboard()
        if not leaderboard_df.empty:
            self.visualizer.plot_leaderboard(
                leaderboard_df,
                save_path="robustness_leaderboard.png"
            )
        
        logger.info("Visualizations created")
    
    def run_experiment(self) -> None:
        """Run the complete red teaming experiment."""
        logger.info("Starting Red Teaming Experiment")
        
        try:
            # Load data
            self.load_data()
            
            # Train models
            self.train_models()
            
            # Perform attacks
            self.perform_attacks()
            
            # Evaluate robustness
            self.evaluate_robustness()
            
            # Create visualizations
            self.create_visualizations()
            
            logger.info("Red Teaming Experiment completed successfully")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Red Teaming for AI Systems")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for experiments"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    if args.verbose:
        config["experiment"]["verbose"] = True
        config["experiment"]["log_level"] = "DEBUG"
    
    # Run experiment
    experiment = RedTeamingExperiment(config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
