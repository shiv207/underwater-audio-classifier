#!/usr/bin/env python3


import os
import sys
import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def retrain_model():
    
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'training')
    
    if not os.path.exists(data_dir):
        print(f"Error: Training data directory not found at {data_dir}")
        return False
    
    audio_extensions = ['*.wav', '*.mp3', '*.flac']
    total_files = 0
    categories = {}
    
    for cat_dir in Path(data_dir).iterdir():
        if cat_dir.is_dir():
            count = 0
            for ext in audio_extensions:
                count += len(list(cat_dir.glob(ext)))
            if count > 0:
                categories[cat_dir.name] = count
                total_files += count
    
    print(f"\n{'='*60}")
    print("UNDERWATER ACOUSTIC CLASSIFIER - TRAINING")
    print(f"{'='*60}")
    print(f"\nTraining Data Summary:")
    print(f"  Total files: {total_files}")
    print(f"  Categories: {len(categories)}")
    for cat, count in categories.items():
        print(f"    - {cat}: {count} files")
    
    if total_files == 0:
        print("\nError: No audio files found in training data!")
        return False
    
    from train import main as train_main
    
    sys.argv = [
        'retrain_model.py',
        '--data-dir', data_dir,
        '--model-type', 'classifier',
        '--epochs', '50',
        '--batch-size', '8',
        '--learning-rate', '0.001',
        '--save-dir', 'models',
        '--device', 'auto'
    ]
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    try:
        train_main()
        print(f"\n{'='*60}")
        print("Training completed successfully!")
        print("Model saved to: models/best_classifier.pth")
        print(f"{'='*60}\n")
        return True
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = retrain_model()
    sys.exit(0 if success else 1)
