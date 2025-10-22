"""
Training utilities including loss functions and trainer classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from typing import Optional, List, Dict
from tqdm import tqdm
import os


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss based on effective number of samples."""
    
    def __init__(self, samples_per_class: list, num_classes: int, loss_type: str = 'focal', 
                 beta: float = 0.9999, gamma: float = 2.0):
        super(ClassBalancedLoss, self).__init__()
        
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * num_classes
        
        self.weights = torch.FloatTensor(weights)
        
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=self.weights, gamma=gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(inputs, targets)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing regularization."""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class MixupAugmentation:
    """Mixup data augmentation."""
    
    def __init__(self, alpha=0.4):
        self.alpha = alpha
    
    def __call__(self, x, y):
        """Apply mixup to batch."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class EnhancedDataset(Dataset):
    """Enhanced dataset with augmentation strategies."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 augment: bool = True,
                 sample_rate: int = 16000):
        self.data_dir = data_dir
        self.split = split
        self.augment = augment and split == 'train'
        self.sample_rate = sample_rate
        
        from .data import AudioPreprocessor, apply_spec_augment, add_noise
        self.preprocessor = AudioPreprocessor()
        
        # Class mappings
        self.class_to_id = {}
        self.id_to_class = {}
        self.dir_name_to_class = {}
        
        class_mappings = {
            'vessels': ['vessels', 'vessel'],
            'marine_animals': ['marine_animals', 'acquatic_mammels', 'aquatic_mammals', 'mammals'],
            'natural_sounds': ['natural_sounds', 'natural', 'earthquake'],
            'other_anthropogenic': ['other_anthropogenic', 'anthropogenic', 'sonar', 'man_made']
        }
        
        available_dirs = [item for item in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, item))]
        
        class_id = 0
        for standard_class, variations in class_mappings.items():
            for dir_name in available_dirs:
                if dir_name.lower() in [v.lower() for v in variations]:
                    if standard_class not in self.class_to_id:
                        self.class_to_id[standard_class] = class_id
                        self.id_to_class[class_id] = standard_class
                        self.dir_name_to_class[dir_name] = standard_class
                        class_id += 1
        
        self.samples = self._load_samples()
        
        # Split train/val per class
        if len(self.samples) > 0:
            class_samples = {}
            for sample in self.samples:
                label = sample[1]
                if label not in class_samples:
                    class_samples[label] = []
                class_samples[label].append(sample)
            
            train_samples = []
            val_samples = []
            
            for label, samples_list in class_samples.items():
                n_samples = len(samples_list)
                
                if n_samples == 1:
                    train_samples.append(samples_list[0])
                    val_samples.append(samples_list[0])
                else:
                    n_train = max(1, int(n_samples * 0.85))
                    np.random.seed(42)
                    indices = np.random.permutation(n_samples)
                    train_samples.extend([samples_list[i] for i in indices[:n_train]])
                    val_samples.extend([samples_list[i] for i in indices[n_train:]])
            
            self.samples = train_samples if split == 'train' else val_samples
            
            # Print class distribution
            if split == 'train':
                class_counts = {}
                for _, label in self.samples:
                    class_counts[label] = class_counts.get(label, 0) + 1
                print(f"\n{split.upper()} SET:")
                for label, count in sorted(class_counts.items()):
                    print(f"  {self.id_to_class[label]:20s}: {count:4d} samples")
    
    def _load_samples(self):
        samples = []
        for dir_name, standard_class in self.dir_name_to_class.items():
            class_dir = os.path.join(self.data_dir, dir_name)
            class_id = self.class_to_id[standard_class]
            
            if not os.path.exists(class_dir):
                continue
            
            # Search recursively for audio files
            for ext in ['*.wav', '*.mp3', '*.flac']:
                import glob
                for audio_file in glob.glob(os.path.join(class_dir, '**', ext), recursive=True):
                    samples.append((audio_file, class_id))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        # Load and preprocess
        audio, log_mel_spec, metadata = self.preprocessor.process_audio_file(audio_path)
        
        if len(audio) == 0 or log_mel_spec.size == 0:
            log_mel_spec = np.zeros((128, 200))
        
        # Apply augmentation
        if self.augment:
            # Random noise injection (30% chance)
            if np.random.random() < 0.3:
                audio = add_noise(audio, noise_factor=0.02)
            
            # SpecAugment (60% chance)
            if np.random.random() < 0.6:
                from .data import apply_spec_augment
                log_mel_spec = apply_spec_augment(log_mel_spec, num_freq_masks=2, num_time_masks=2)
        
        # Ensure consistent size
        target_time_frames = 200
        if log_mel_spec.shape[1] > target_time_frames:
            log_mel_spec = log_mel_spec[:, :target_time_frames]
        elif log_mel_spec.shape[1] < target_time_frames:
            pad_width = target_time_frames - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        
        spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0)
        label_tensor = torch.LongTensor([label])
        
        return spec_tensor, label_tensor[0]


class AdvancedTrainer:
    """Advanced trainer with all improvements."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 num_classes: int,
                 device: str = 'cpu',
                 learning_rate: float = 1e-3,
                 use_focal_loss: bool = True,
                 use_label_smoothing: bool = True,
                 use_mixup: bool = True,
                 samples_per_class: list = None):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # Optimizer with discriminative learning rates
        self.optimizer = optim.AdamW([
            {'params': model.cnn_backbone.parameters(), 'lr': learning_rate * 0.1},
            {'params': model.transformer_classifier.parameters(), 'lr': learning_rate}
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Loss function
        if use_focal_loss and samples_per_class:
            print("Using Class-Balanced Focal Loss + Label Smoothing")
            self.criterion = ClassBalancedLoss(
                samples_per_class=samples_per_class,
                num_classes=num_classes,
                loss_type='focal',
                gamma=2.0
            )
            self.label_smoothing = 0.0
        elif use_label_smoothing:
            print("Using Label Smoothing Cross Entropy")
            self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        else:
            print("Using standard Cross Entropy")
            self.criterion = nn.CrossEntropyLoss()
        
        # Mixup
        self.use_mixup = use_mixup
        self.mixup = MixupAugmentation(alpha=0.4) if use_mixup else None
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_balanced_acc = 0
    
    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Mixup loss."""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
    
    def train_epoch(self, epoch: int):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply mixup
            if self.use_mixup and np.random.random() < 0.5:
                data, target_a, target_b, lam = self.mixup(data, target)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.mixup_criterion(output, target_a, target_b, lam)
            else:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def validate(self):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                for t, p in zip(target, pred):
                    t_item = t.item()
                    if t_item not in class_correct:
                        class_correct[t_item] = 0
                        class_total[t_item] = 0
                    class_total[t_item] += 1
                    if t_item == p.item():
                        class_correct[t_item] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        class_acc = {}
        for cls in class_correct:
            class_acc[cls] = 100.0 * class_correct[cls] / class_total[cls]
        
        return avg_loss, accuracy, class_acc, all_predictions, all_targets
    
    def train(self, num_epochs: int, save_path: str):
        """Full training loop."""
        patience = 25
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*60)
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, class_acc, preds, targets = self.validate()
            
            self.scheduler.step()
            
            # Calculate balanced accuracy
            balanced_acc = np.mean([class_acc[i] for i in range(self.num_classes) if i in class_acc])
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Balanced Acc: {balanced_acc:.2f}%")
            
            # Per-class accuracy
            print(f"\nPer-Class Accuracy:")
            for i in range(self.num_classes):
                if i in class_acc:
                    class_name = list(self.train_loader.dataset.class_to_id.keys())[i] if hasattr(self, 'train_dataset') else f"Class {i}"
                    print(f"  {class_name:20s}: {class_acc[i]:.2f}%")
            
            # Save best model
            if balanced_acc > self.best_balanced_acc:
                self.best_balanced_acc = balanced_acc
                patience_counter = 0
                self.save_model(save_path, epoch, val_acc, balanced_acc)
                print(f"\n✓ New best model! Balanced Acc: {balanced_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered. No improvement for {patience} epochs.")
                    break
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
        
        return val_acc, self.best_balanced_acc
    
    def save_model(self, save_path: str, epoch: int, accuracy: float, balanced_accuracy: float):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'num_classes': self.num_classes
        }
        
        torch.save(checkpoint, save_path)
