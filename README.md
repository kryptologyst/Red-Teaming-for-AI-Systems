# Red Teaming for AI Systems

A comprehensive framework for adversarial robustness testing and red teaming of machine learning models. This project implements various adversarial attacks, robustness evaluation metrics, and visualization tools to assess the security and reliability of AI systems.

## ⚠️ Important Disclaimer

**This tool is for research and educational purposes only.**

Red teaming outputs may be unstable or misleading and should not be used for regulated decisions without human review. This tool is not a substitute for professional security assessment or human judgment. Always validate results and consider the limitations of adversarial testing methods.

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Red-Teaming-for-AI-Systems.git
cd Red-Teaming-for-AI-Systems
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# or for development
pip install -e ".[dev]"
```

3. Run a quick experiment:
```bash
python scripts/run_experiment.py --config configs/default.yaml
```

4. Launch the interactive demo:
```bash
streamlit run demo/streamlit_app.py
```

## 📁 Project Structure

```
red-teaming-ai/
├── src/red_teaming/          # Main package
│   ├── data/                 # Data loading and preprocessing
│   ├── models/               # Model definitions and training
│   ├── attacks/              # Adversarial attack implementations
│   ├── evaluation/           # Robustness evaluation metrics
│   ├── visualization/        # Plotting and visualization tools
│   └── utils/                # Utility functions
├── configs/                  # Configuration files
├── scripts/                  # Experiment scripts
├── demo/                     # Interactive demo
├── tests/                    # Unit tests
├── data/                     # Data storage
├── assets/                   # Generated plots and outputs
└── docs/                     # Documentation
```

## 🔧 Features

### Supported Datasets
- **Iris Dataset**: Classic classification dataset
- **Synthetic Data**: Configurable synthetic datasets for testing

### Supported Models
- **Random Forest**: Ensemble method with good interpretability
- **Logistic Regression**: Linear baseline model
- **Neural Networks**: PyTorch-based neural networks with robustness features
- **Support Vector Machines**: Kernel-based models
- **Multi-layer Perceptrons**: Scikit-learn neural networks

### Adversarial Attacks
- **Simple Attacks**: Gaussian noise, uniform noise, FGSM-like perturbations
- **PyTorch Attacks**: FGSM, PGD, CW, DeepFool, AutoAttack (via torchattacks)
- **Gradient-based Attacks**: Custom implementations for models with gradient access

### Evaluation Metrics
- **Accuracy Metrics**: Original vs adversarial accuracy, accuracy drop
- **Perturbation Analysis**: L2/L∞ norms, perturbation statistics
- **Robustness Metrics**: Attack success rate, robust accuracy
- **Calibration Metrics**: Expected Calibration Error (ECE), Maximum Calibration Error (MCE)

### Visualization Tools
- Accuracy comparison plots
- Perturbation analysis (distributions, per-feature analysis)
- Confusion matrices (original vs adversarial)
- Robustness curves
- Calibration curves
- Leaderboard visualizations

## Usage Examples

### Basic Experiment

```python
from red_teaming import DataLoader, ModelFactory, AttackFactory, RobustnessEvaluator

# Load data
data_loader = DataLoader(random_state=42)
X, y, feature_names, target_names = data_loader.load_iris_dataset()
X_train, X_test, y_train, y_test = data_loader.preprocess_data(X, y)

# Train model
model = ModelFactory.create_model("random_forest", random_state=42)
robust_model = RobustModel(model, "Random Forest")
robust_model.train(X_train, y_train)

# Perform attack
attack = AttackFactory.create_attack("gaussian", epsilon=0.1)
attacker = AdversarialAttacker(robust_model.model, "Random Forest")
result = attacker.perform_attack(X_test, y_test, attack)

# Evaluate robustness
evaluator = RobustnessEvaluator(robust_model.model, "Random Forest")
robustness_report = evaluator.create_robustness_report(
    X_test, y_test, {"gaussian": result["X_adv"]}
)
```

### Configuration-based Experiment

```yaml
# configs/custom.yaml
data:
  dataset: "iris"
  test_size: 0.3
  scale_features: true

models:
  - name: "rf_model"
    type: "random_forest"
    params:
      n_estimators: 100
      random_state: 42

attacks:
  - name: "gaussian_attack"
    type: "gaussian"
    epsilon: 0.1
```

```bash
python scripts/run_experiment.py --config configs/custom.yaml
```

### Interactive Demo

Launch the Streamlit demo for interactive experimentation:

```bash
streamlit run demo/streamlit_app.py
```

The demo provides:
- Interactive parameter tuning
- Real-time visualization
- Model comparison
- Detailed analysis and interpretation

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src/red_teaming tests/
```

## Evaluation Framework

### Robustness Metrics

The framework provides comprehensive robustness evaluation:

1. **Baseline Performance**: Standard classification metrics
2. **Adversarial Robustness**: Performance under attack
3. **Perturbation Analysis**: Statistical analysis of adversarial perturbations
4. **Calibration Assessment**: Model confidence calibration
5. **Comparative Analysis**: Leaderboard for model comparison

### Leaderboard

The robustness leaderboard ranks models by:
- Worst adversarial accuracy (primary metric)
- Mean accuracy drop across attacks
- Calibration error
- Perturbation magnitude

## Limitations and Considerations

### Known Limitations

1. **Attack Effectiveness**: Simple attacks may not reflect real-world adversarial scenarios
2. **Dataset Dependency**: Results may vary significantly across different datasets
3. **Model Architecture**: Some attacks are specific to certain model types
4. **Computational Cost**: Advanced attacks can be computationally expensive
5. **Interpretability**: Adversarial examples may not always be interpretable

### Best Practices

1. **Multiple Attacks**: Test with various attack types and parameters
2. **Cross-validation**: Use multiple random seeds and data splits
3. **Baseline Comparison**: Always compare against simple baselines
4. **Human Review**: Validate results with domain experts
5. **Documentation**: Document all experimental parameters and results

## 🛠️ Development

### Code Style

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pre-commit** hooks for quality assurance

Setup development environment:

```bash
pip install -e ".[dev]"
pre-commit install
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `pre-commit run --all-files`
5. Submit a pull request

## References

### Key Papers

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083.
- Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. 2017 IEEE symposium on security and privacy (SP).

### Tools and Libraries

- [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch): PyTorch adversarial attacks
- [Captum](https://captum.ai/): Model interpretability for PyTorch
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox): IBM's robustness library

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review the example notebooks in `notebooks/`

## Changelog

### Version 1.0.0
- Initial release
- Basic adversarial attacks (Gaussian, Uniform, FGSM-like)
- PyTorch attack integration
- Comprehensive evaluation framework
- Interactive Streamlit demo
- Robustness leaderboard
- Visualization tools

---

**Remember**: This tool is for research and educational purposes. Always validate results and consider the limitations of adversarial testing methods in your specific use case.
# Red-Teaming-for-AI-Systems
