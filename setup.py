#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    print("Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def verify_installation():
    print("\nVerifying installation...")
    
    try:
        import torch
        import librosa
        import numpy as np
        print("✅ Core libraries imported successfully!")
        
        sys.path.append('src')
        from preprocess import AudioPreprocessor
        from detector import SimpleEnergyDetector
        from classifier import AcousticClassifier
        print("✅ System components loaded successfully!")
        
        preprocessor = AudioPreprocessor()
        detector = SimpleEnergyDetector()
        classifier = AcousticClassifier()
        print("✅ System components initialized successfully!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ System error: {e}")
        return False

def create_directories():
    print("\nCreating directories...")
    
    dirs = ['models', 'results', 'data/test']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def main():
    print("=" * 60)
    print("UNDERWATER ACOUSTIC CLASSIFICATION SYSTEM SETUP")
    print("=" * 60)
    
    os.chdir(Path(__file__).parent)
    
    if not install_requirements():
        print("\n❌ Setup failed at dependency installation!")
        return 1
    
    create_directories()
    
    if not verify_installation():
        print("\n❌ Setup failed at verification!")
        return 1
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nQuick Start Commands:")
    print("1. Test system:        python test_system.py")
    print("2. Process audio:      python main.py --input data/audio.wav --output results.json")
    print("3. Train model:        python train.py --data-dir data/training")
    print("4. Run evaluation:     python main.py --evaluate --ground-truth gt.json --predictions pred.json")
    
    print("\nDocker Commands:")
    print("1. Build image:        docker build -t uda-classifier .")
    print("2. Run container:      docker run -v /data:/app/data uda-classifier")
    
    print("\nSystem is ready for underwater acoustic classification! 🌊")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
