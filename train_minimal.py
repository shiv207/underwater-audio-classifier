#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import json
from pathlib import Path
import time
import numpy as np

from simple_config import *
from simple_data import create_dataloaders
from simple_model import create_model, count_parameters


class FocalLoss(nn.Module):
    """Focal Loss - focuses training on hard examples"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    confidence_scores = []
    
    pbar = tqdm(loader, desc=f'Training Epoch {epoch+1}')
    for batch_idx, batch_data in enumerate(pbar):
        if len(batch_data) == 3:
            data, target, weights = batch_data
            weights = weights.to(device)
        else:
            data, target = batch_data
            weights = None
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        if hasattr(model, 'forward') and 'return_confidence' in model.forward.__code__.co_varnames:
            output, confidence = model(data, return_confidence=True)
            confidence_scores.extend(confidence.cpu().detach().numpy())
        else:
            output = model(data)
        
        loss = criterion(output, target)
        if weights is not None:
            loss = (loss * weights).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        acc = 100.0 * correct / total
        avg_conf = np.mean(confidence_scores) if confidence_scores else 0
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'acc': f'{acc:.1f}%',
            'conf': f'{avg_conf:.3f}'
        })
    
    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    confidence_scores = []
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False)
        for batch_data in pbar:
            if len(batch_data) == 3:
                data, target, _ = batch_data
            else:
                data, target = batch_data
            
            data, target = data.to(device), target.to(device)
            
            if hasattr(model, 'forward') and 'return_confidence' in model.forward.__code__.co_varnames:
                output, confidence = model(data, return_confidence=True)
                confidence_scores.extend(confidence.cpu().numpy())
            else:
                output = model(data)
            
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            for t, p in zip(target, pred):
                class_total[t.item()] += 1
                if t == p:
                    class_correct[t.item()] += 1
            
            acc = 100.0 * correct / total
            avg_conf = np.mean(confidence_scores) if confidence_scores else 0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{acc:.1f}%',
                'conf': f'{avg_conf:.3f}'
            })
    
    avg_loss = total_loss / len(loader)
    avg_acc = 100.0 * correct / total
    
    class_acc = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            class_acc.append(acc)
        else:
            class_acc.append(0.0)
    
    balanced_acc = sum(class_acc) / len(class_acc)
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    return avg_loss, avg_acc, balanced_acc, class_acc, avg_confidence


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    print("🚀 Loading data...")
    start_time = time.time()
    train_loader, val_loader, class_to_id = create_dataloaders(DATA_DIR)
    print(f"✓ Data loaded in {time.time() - start_time:.1f}s")
    
    id_to_class = {v: k for k, v in class_to_id.items()}
    num_classes = len(class_to_id)
    
    model = create_model(num_classes=num_classes).to(device)
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    criterion = FocalLoss(alpha=1, gamma=2)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,
        T_mult=2,
        eta_min=LEARNING_RATE * 0.01
    )
    
    best_acc = 0
    best_balanced_acc = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_balanced_acc': []
    }
    
    print(f"\n🏃‍♂️ Starting training for {NUM_EPOCHS} epochs...")
    training_start = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch
        )
        
        val_loss, val_acc, balanced_acc, class_accs, avg_confidence = validate(
            model, val_loader, criterion, device, num_classes
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_balanced_acc'].append(balanced_acc)
        
        epoch_time = time.time() - epoch_start
        print(f"Train: {train_loss:.4f} | {train_acc:.2f}% | Time: {epoch_time:.1f}s")
        print(f"Val: {val_loss:.4f} | {val_acc:.2f}% | Balanced: {balanced_acc:.2f}% | Conf: {avg_confidence:.3f}")
        
        for i in range(num_classes):
            print(f"  {id_to_class[i]}: {class_accs[i]:.2f}%")
        
        # Prioritize balanced accuracy for model selection
        current_score = balanced_acc * 0.7 + val_acc * 0.3
        best_score = best_balanced_acc * 0.7 + best_acc * 0.3
        
        if current_score > best_score:
            best_acc = val_acc
            best_balanced_acc = balanced_acc
            patience_counter = 0
            
            save_path = MODELS_DIR / 'best_model_finetuned.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_id': class_to_id,
                'val_acc': val_acc,
                'balanced_acc': balanced_acc,
                'confidence': avg_confidence
            }, save_path)
            print(f"✓ Best model saved! Val: {val_acc:.2f}% | Balanced: {balanced_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    history_path = MODELS_DIR / 'history_simple.json'
    with open(history_path, 'w') as f:
        json.dump({
            'best_val_acc': best_acc,
            'best_balanced_acc': best_balanced_acc,
            'history': history
        }, f, indent=2)
    
    total_time = time.time() - training_start
    print(f"\n🎉 Training complete! Best: {best_acc:.2f}%")
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    print(f"⚡ Average time per epoch: {total_time/(epoch+1):.1f}s")


if __name__ == '__main__':
    main()
