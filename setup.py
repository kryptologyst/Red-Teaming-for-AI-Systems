#!/usr/bin/env python3
"""Setup script for the Red Teaming framework."""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🛡️ Red Teaming Framework Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        return False
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        return False
    
    # Run installation test
    print("\n🧪 Running installation test...")
    test_script = Path(__file__).parent / "scripts" / "test_installation.py"
    if test_script.exists():
        if not run_command(f"python {test_script}", "Installation test"):
            print("⚠️ Installation test failed, but setup may still work")
    else:
        print("⚠️ Installation test script not found")
    
    # Create necessary directories
    directories = ["data", "experiments", "assets", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Test the installation: python scripts/test_installation.py")
    print("2. Run an experiment: python scripts/run_experiment.py")
    print("3. Launch the demo: streamlit run demo/streamlit_app.py")
    print("4. Check out the documentation: README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
