"""Streamlit demo for Red Teaming and Adversarial Robustness Testing."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from red_teaming.data import DataLoader, SyntheticDataGenerator
from red_teaming.models import ModelFactory, RobustModel
from red_teaming.attacks import AttackFactory, AdversarialAttacker
from red_teaming.evaluation import RobustnessEvaluator
from red_teaming.visualization import RobustnessVisualizer
from red_teaming.utils import set_seed, get_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Red Teaming for AI Systems",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Red Teaming for AI Systems</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This tool is for research and educational purposes only.</strong></p>
        <p>Red teaming outputs may be unstable or misleading and should not be used for regulated decisions without human review. 
        This tool is not a substitute for professional security assessment or human judgment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Dataset selection
        st.subheader("Dataset")
        dataset = st.selectbox(
            "Select Dataset",
            ["Iris", "Synthetic"],
            help="Choose the dataset for the experiment"
        )
        
        # Model selection
        st.subheader("Model")
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "Logistic Regression", "Neural Network"],
            help="Choose the model to test"
        )
        
        # Attack parameters
        st.subheader("Attack Parameters")
        epsilon = st.slider(
            "Perturbation Strength (ε)",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Strength of adversarial perturbations"
        )
        
        attack_type = st.selectbox(
            "Attack Type",
            ["Gaussian Noise", "Uniform Noise", "FGSM-like"],
            help="Type of adversarial attack"
        )
        
        # Experiment parameters
        st.subheader("Experiment")
        test_size = st.slider(
            "Test Size",
            min_value=0.1,
            max_value=0.5,
            value=0.3,
            step=0.05,
            help="Proportion of data for testing"
        )
        
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=10000,
            value=42,
            help="Random seed for reproducibility"
        )
        
        # Run experiment button
        run_experiment = st.button("🚀 Run Red Teaming Experiment", type="primary")
    
    # Main content
    if run_experiment:
        with st.spinner("Running red teaming experiment..."):
            try:
                # Set random seed
                set_seed(random_seed)
                
                # Initialize components
                data_loader = DataLoader(random_state=random_seed)
                visualizer = RobustnessVisualizer()
                
                # Load data
                st.subheader("📊 Dataset Information")
                
                if dataset == "Iris":
                    X, y, feature_names, target_names = data_loader.load_iris_dataset()
                else:  # Synthetic
                    X, y, feature_names, target_names = data_loader.load_synthetic_dataset(
                        n_samples=500, n_features=10, n_classes=3
                    )
                
                # Preprocess data
                X_train, X_test, y_train, y_test = data_loader.preprocess_data(
                    X, y, test_size=test_size, scale_features=True
                )
                
                # Display dataset info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(X))
                with col2:
                    st.metric("Features", len(feature_names))
                with col3:
                    st.metric("Classes", len(target_names))
                
                # Train model
                st.subheader("🤖 Model Training")
                
                model_mapping = {
                    "Random Forest": "random_forest",
                    "Logistic Regression": "logistic_regression",
                    "Neural Network": "neural_network"
                }
                
                model_params = {
                    "random_state": random_seed,
                    "input_dim": X.shape[1],
                    "num_classes": len(target_names),
                }
                
                base_model = ModelFactory.create_model(
                    model_mapping[model_type], **model_params
                )
                robust_model = RobustModel(base_model, model_type)
                
                # Train model
                robust_model.train(X_train, y_train)
                
                # Evaluate baseline performance
                baseline_metrics = robust_model.evaluate(X_test, y_test, verbose=False)
                
                st.success(f"✅ {model_type} trained successfully!")
                
                # Display baseline metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Baseline Accuracy", f"{baseline_metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{baseline_metrics['precision_macro']:.4f}")
                with col3:
                    st.metric("Recall", f"{baseline_metrics['recall_macro']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{baseline_metrics['f1_macro']:.4f}")
                
                # Perform adversarial attack
                st.subheader("⚔️ Adversarial Attack")
                
                attack_mapping = {
                    "Gaussian Noise": "gaussian",
                    "Uniform Noise": "uniform",
                    "FGSM-like": "fgsm_like"
                }
                
                attack = AttackFactory.create_attack(
                    attack_mapping[attack_type],
                    epsilon=epsilon,
                    random_state=random_seed
                )
                
                attacker = AdversarialAttacker(robust_model.model, model_type)
                attack_result = attacker.perform_attack(
                    X_test, y_test, attack, verbose=False
                )
                
                # Display attack results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Accuracy", f"{attack_result['original_accuracy']:.4f}")
                with col2:
                    st.metric("Adversarial Accuracy", f"{attack_result['adversarial_accuracy']:.4f}")
                with col3:
                    st.metric("Accuracy Drop", f"{attack_result['accuracy_drop']:.4f}")
                with col4:
                    st.metric("Attack Success Rate", f"{attack_result['attack_success_rate']:.4f}")
                
                # Robustness evaluation
                st.subheader("🛡️ Robustness Analysis")
                
                evaluator = RobustnessEvaluator(robust_model.model, model_type)
                adversarial_results = {attack_type: attack_result["X_adv"]}
                
                robustness_report = evaluator.create_robustness_report(
                    X_test, y_test, adversarial_results, verbose=False
                )
                
                # Display robustness summary
                summary = robustness_report["summary"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Accuracy Drop", f"{summary['mean_accuracy_drop']:.4f}")
                with col2:
                    st.metric("Max Accuracy Drop", f"{summary['max_accuracy_drop']:.4f}")
                with col3:
                    st.metric("Worst Adversarial Accuracy", f"{summary['worst_adversarial_accuracy']:.4f}")
                
                # Visualizations
                st.subheader("📈 Visualizations")
                
                # Accuracy comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                
                categories = ["Original", "Adversarial"]
                accuracies = [attack_result["original_accuracy"], attack_result["adversarial_accuracy"]]
                colors = ["#2E8B57", "#DC143C"]
                
                bars = ax.bar(categories, accuracies, color=colors, alpha=0.7)
                ax.set_ylabel("Accuracy")
                ax.set_title(f"Accuracy Comparison - {model_type}")
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Perturbation analysis
                perturbation = attack_result["perturbation"]
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"Perturbation Analysis - {attack_type}", fontsize=16)
                
                # L2 norms
                l2_norms = np.linalg.norm(perturbation, axis=1)
                axes[0, 0].hist(l2_norms, bins=20, alpha=0.7, color="#FF6B6B")
                axes[0, 0].set_xlabel("L2 Norm")
                axes[0, 0].set_ylabel("Frequency")
                axes[0, 0].set_title("Distribution of L2 Perturbation Norms")
                
                # L∞ norms
                linf_norms = np.max(np.abs(perturbation), axis=1)
                axes[0, 1].hist(linf_norms, bins=20, alpha=0.7, color="#4ECDC4")
                axes[0, 1].set_xlabel("L∞ Norm")
                axes[0, 1].set_ylabel("Frequency")
                axes[0, 1].set_title("Distribution of L∞ Perturbation Norms")
                
                # Perturbation magnitude per feature
                pert_magnitude = np.mean(np.abs(perturbation), axis=0)
                feature_indices = range(len(pert_magnitude))
                axes[1, 0].bar(feature_indices, pert_magnitude, alpha=0.7, color="#45B7D1")
                axes[1, 0].set_xlabel("Feature Index")
                axes[1, 0].set_ylabel("Mean Absolute Perturbation")
                axes[1, 0].set_title("Perturbation Magnitude per Feature")
                
                # Original vs Adversarial (first 2 features)
                if X_test.shape[1] >= 2:
                    axes[1, 1].scatter(X_test[:, 0], X_test[:, 1], alpha=0.6, 
                                     label="Original", s=20, color="#2E8B57")
                    axes[1, 1].scatter(attack_result["X_adv"][:, 0], attack_result["X_adv"][:, 1], 
                                     alpha=0.6, label="Adversarial", s=20, color="#DC143C")
                    axes[1, 1].set_xlabel("Feature 0")
                    axes[1, 1].set_ylabel("Feature 1")
                    axes[1, 1].set_title("Original vs Adversarial (First 2 Features)")
                    axes[1, 1].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Confusion matrices
                y_pred_orig = robust_model.predict(X_test)
                y_pred_adv = robust_model.predict(attack_result["X_adv"])
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f"Confusion Matrices - {attack_type}", fontsize=16)
                
                from sklearn.metrics import confusion_matrix
                
                # Original confusion matrix
                cm_orig = confusion_matrix(y_test, y_pred_orig)
                sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=target_names, yticklabels=target_names, ax=axes[0])
                axes[0].set_title("Original Data")
                axes[0].set_xlabel("Predicted")
                axes[0].set_ylabel("True")
                
                # Adversarial confusion matrix
                cm_adv = confusion_matrix(y_test, y_pred_adv)
                sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Reds',
                           xticklabels=target_names, yticklabels=target_names, ax=axes[1])
                axes[1].set_title("Adversarial Data")
                axes[1].set_xlabel("Predicted")
                axes[1].set_ylabel("True")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Detailed results
                st.subheader("📋 Detailed Results")
                
                # Create results DataFrame
                results_data = {
                    "Metric": [
                        "Original Accuracy",
                        "Adversarial Accuracy", 
                        "Accuracy Drop",
                        "Attack Success Rate",
                        "Mean L2 Perturbation",
                        "Mean L∞ Perturbation",
                        "Max L2 Perturbation",
                        "Max L∞ Perturbation"
                    ],
                    "Value": [
                        f"{attack_result['original_accuracy']:.4f}",
                        f"{attack_result['adversarial_accuracy']:.4f}",
                        f"{attack_result['accuracy_drop']:.4f}",
                        f"{attack_result['attack_success_rate']:.4f}",
                        f"{attack_result['perturbation_l2_mean']:.4f}",
                        f"{attack_result['perturbation_linf_mean']:.4f}",
                        f"{attack_result['perturbation_l2_max']:.4f}",
                        f"{attack_result['perturbation_linf_max']:.4f}"
                    ]
                }
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Interpretation
                st.subheader("🔍 Interpretation")
                
                accuracy_drop = attack_result['accuracy_drop']
                if accuracy_drop < 0.05:
                    robustness_level = "🟢 High"
                    interpretation = "The model shows good robustness to this type of attack."
                elif accuracy_drop < 0.15:
                    robustness_level = "🟡 Medium"
                    interpretation = "The model shows moderate robustness to this type of attack."
                else:
                    robustness_level = "🔴 Low"
                    interpretation = "The model shows poor robustness to this type of attack."
                
                st.markdown(f"**Robustness Level:** {robustness_level}")
                st.markdown(f"**Interpretation:** {interpretation}")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                if accuracy_drop > 0.1:
                    st.markdown("""
                    **Recommendations for improving robustness:**
                    - Consider adversarial training
                    - Use ensemble methods
                    - Implement input preprocessing/denoising
                    - Regularize the model more strongly
                    - Use robust optimization techniques
                    """)
                else:
                    st.markdown("""
                    **Model appears robust to this attack type.**
                    Consider testing with different attack types and parameters.
                    """)
                
            except Exception as e:
                st.error(f"❌ Experiment failed: {str(e)}")
                logger.error(f"Experiment failed: {e}")
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to Red Teaming for AI Systems! 🛡️
        
        This interactive demo allows you to:
        
        - **Test model robustness** against adversarial attacks
        - **Compare different models** and their resilience
        - **Analyze perturbation patterns** and their effects
        - **Visualize attack results** with comprehensive plots
        
        ### How to use:
        1. Configure your experiment in the sidebar
        2. Select dataset, model, and attack parameters
        3. Click "Run Red Teaming Experiment"
        4. Analyze the results and visualizations
        
        ### Supported Features:
        - **Datasets:** Iris, Synthetic classification data
        - **Models:** Random Forest, Logistic Regression, Neural Networks
        - **Attacks:** Gaussian/Uniform noise, FGSM-like perturbations
        - **Analysis:** Accuracy comparison, perturbation analysis, confusion matrices
        
        Start by configuring your experiment in the sidebar! 🚀
        """)


if __name__ == "__main__":
    main()
