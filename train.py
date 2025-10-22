#!/usr/bin/env python3
"""
Simplified Training Script for Underwater Acoustic Classification

Usage:
    python train.py --data-dir data/training --epochs 50
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core import UnderwaterAcousticClassifier, EnhancedDataset, AdvancedTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Underwater Acoustic Classifier')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to training data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--no-focal-loss', action='store_true', help='Disable focal loss')
    parser.add_argument('--no-mixup', action='store_true', help='Disable mixup augmentation')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("UNDERWATER ACOUSTIC CLASSIFICATION TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Focal loss: {not args.no_focal_loss}")
    print(f"Mixup: {not args.no_mixup}")
    
    # Load data
    print("\nLoading datasets...")
    train_dataset = EnhancedDataset(args.data_dir, split='train', augment=True)
    val_dataset = EnhancedDataset(args.data_dir, split='val', augment=False)
    
    if len(train_dataset) == 0:
        print("Error: No training data!")
        return
    
    # Get class counts
    class_counts = {}
    for _, label in train_dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    num_classes = len(class_counts)
    samples_per_class = [class_counts.get(i, 1) for i in range(num_classes)]
    
    # Create balanced sampler
    print("\nCreating balanced sampler...")
    sample_weights = []
    for _, label in train_dataset.samples:
        weight = 1.0 / class_counts[label]
        sample_weights.append(weight)
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print("✓ Balanced sampler created")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             sampler=train_sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    
    # Model
    print(f"\nCreating model with {num_classes} classes...")
    model = UnderwaterAcousticClassifier(num_classes=num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        learning_rate=args.learning_rate,
        use_focal_loss=not args.no_focal_loss,
        use_mixup=not args.no_mixup,
        samples_per_class=samples_per_class
    )
    
    # Train
    save_path = os.path.join(args.save_dir, 'best_model.pth')
    acc, balanced_acc = trainer.train(args.epochs, save_path)
    
    # Summary
    summary = {
        'best_accuracy': float(acc),
        'best_balanced_accuracy': float(balanced_acc),
        'num_classes': num_classes,
        'samples_per_class': samples_per_class,
        'focal_loss': not args.no_focal_loss,
        'mixup': not args.no_mixup
    }
    
    summary_path = os.path.join(args.save_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best Accuracy: {acc:.2f}%")
    print(f"Best Balanced Accuracy: {balanced_acc:.2f}%")
    print(f"Model: {save_path}")


if __name__ == '__main__':
    main()
