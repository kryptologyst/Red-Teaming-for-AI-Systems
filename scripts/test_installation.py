#!/usr/bin/env python3
"""Quick test script to verify the red teaming framework installation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from red_teaming.data import DataLoader, SyntheticDataGenerator
        print("✅ Data modules imported successfully")
        
        from red_teaming.models import ModelFactory, RobustModel
        print("✅ Model modules imported successfully")
        
        from red_teaming.attacks import AttackFactory, AdversarialAttacker
        print("✅ Attack modules imported successfully")
        
        from red_teaming.evaluation import RobustnessEvaluator, MetricsCalculator
        print("✅ Evaluation modules imported successfully")
        
        from red_teaming.visualization import RobustnessVisualizer
        print("✅ Visualization modules imported successfully")
        
        from red_teaming.utils import set_seed, get_device, validate_data
        print("✅ Utility modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with a simple experiment."""
    print("\nTesting basic functionality...")
    
    try:
        from red_teaming.data import DataLoader
        from red_teaming.models import ModelFactory, RobustModel
        from red_teaming.attacks import AttackFactory, AdversarialAttacker
        from red_teaming.utils import set_seed
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Load data
        data_loader = DataLoader(random_state=42)
        X, y, feature_names, target_names = data_loader.load_iris_dataset()
        X_train, X_test, y_train, y_test = data_loader.preprocess_data(X, y, test_size=0.3)
        
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train model
        model = ModelFactory.create_model("random_forest", random_state=42)
        robust_model = RobustModel(model, "test_model")
        robust_model.train(X_train, y_train)
        
        # Evaluate baseline
        baseline_metrics = robust_model.evaluate(X_test, y_test, verbose=False)
        print(f"✅ Model trained and evaluated: Accuracy = {baseline_metrics['accuracy']:.4f}")
        
        # Perform attack
        attack = AttackFactory.create_attack("gaussian", epsilon=0.1)
        attacker = AdversarialAttacker(robust_model.model, "test_model")
        result = attacker.perform_attack(X_test, y_test, attack, verbose=False)
        
        print(f"✅ Attack performed: Accuracy drop = {result['accuracy_drop']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_device_detection():
    """Test device detection."""
    print("\nTesting device detection...")
    
    try:
        from red_teaming.utils import get_device
        import torch
        
        device = get_device()
        print(f"✅ Device detected: {device}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
        else:
            print("✅ Using CPU")
            
        return True
        
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🛡️ Red Teaming Framework - Installation Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import test failed. Please check your installation.")
        return False
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Test device detection
    device_ok = test_device_detection()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Functionality: {'✅ PASS' if functionality_ok else '❌ FAIL'}")
    print(f"Device Detection: {'✅ PASS' if device_ok else '❌ FAIL'}")
    
    if imports_ok and functionality_ok and device_ok:
        print("\n🎉 All tests passed! The framework is ready to use.")
        print("\nNext steps:")
        print("1. Run a full experiment: python scripts/run_experiment.py")
        print("2. Launch the demo: streamlit run demo/streamlit_app.py")
        print("3. Check out the example notebook: notebooks/red_teaming_example.ipynb")
        return True
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
