
import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import AudioPreprocessor, apply_spec_augment, add_noise
from detector import CNNBiLSTMDetector
from classifier import UnderwaterAcousticClassifier
from evaluate import UnderwaterAcousticEvaluator

class UnderwaterAcousticDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 augment: bool = True,
                 max_duration: float = 10.0,
                 sample_rate: int = 22050):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and split == 'train'
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.preprocessor = AudioPreprocessor()
        self.class_to_id = {}
        self.id_to_class = {}
        self.dir_name_to_class = {}
        class_mappings = {
            'vessels': ['vessels', 'vessel'],
            'marine_animals': ['marine_animals', 'acquatic_mammels', 'aquatic_mammals', 'mammals'],
            'natural_sounds': ['natural_sounds', 'natural', 'earthquake'],
            'other_anthropogenic': ['other_anthropogenic', 'anthropogenic', 'sonar', 'man_made']
        }
        available_dirs = []
        if os.path.exists(self.data_dir):
            for item in os.listdir(self.data_dir):
                item_path = self.data_dir / item
                if item_path.is_dir():
                    available_dirs.append(item)
        class_id = 0
        for standard_class, variations in class_mappings.items():
            for dir_name in available_dirs:
                if dir_name.lower() in [v.lower() for v in variations]:
                    if standard_class not in self.class_to_id:
                        self.class_to_id[standard_class] = class_id
                        self.id_to_class[class_id] = standard_class
                        self.dir_name_to_class[dir_name] = standard_class
                        class_id += 1
        if not self.class_to_id:
            for idx, dir_name in enumerate(sorted(available_dirs)):
                self.class_to_id[dir_name] = idx
                self.id_to_class[idx] = dir_name
                self.dir_name_to_class[dir_name] = dir_name
        
        self.samples = self._load_samples()
        
        if len(self.samples) > 0:
            class_counts = {}
            for _, label in self.samples:
                class_counts[label] = class_counts.get(label, 0) + 1
            
            min_class_count = min(class_counts.values()) if class_counts else 0
            
            if min_class_count >= 2 and len(self.samples) >= 4:
                train_samples, val_samples = train_test_split(
                    self.samples, test_size=0.2, random_state=42, 
                    stratify=[s[1] for s in self.samples]
                )
            else:
                split_idx = max(1, int(len(self.samples) * 0.8))
                train_samples = self.samples[:split_idx]
                val_samples = self.samples[split_idx:] if split_idx < len(self.samples) else self.samples[-1:]
            
            if split == 'train':
                self.samples = train_samples
            else:
                self.samples = val_samples
    
    def _load_samples(self):
        samples = []
        for dir_name, standard_class in self.dir_name_to_class.items():
            class_dir = self.data_dir / dir_name
            class_id = self.class_to_id[standard_class]
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            audio_extensions = ['*.wav', '*.mp3', '*.flac']
            for ext in audio_extensions:
                for audio_file in class_dir.glob(ext):
                    samples.append((str(audio_file), class_id))
        
        print(f"Found {len(samples)} samples for {self.split} split")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        audio, log_mel_spec, metadata = self.preprocessor.process_audio_file(audio_path)
        
        if len(audio) == 0 or log_mel_spec.size == 0:
            log_mel_spec = np.zeros((128, 100))
        
        if self.augment:
            if np.random.random() < 0.3:
                audio = add_noise(audio, noise_factor=0.01)
            
            if np.random.random() < 0.5:
                log_mel_spec = apply_spec_augment(log_mel_spec)
        
        target_time_frames = 200  # Fixed time dimension
        
        if log_mel_spec.shape[1] > target_time_frames:
            log_mel_spec = log_mel_spec[:, :target_time_frames]
        elif log_mel_spec.shape[1] < target_time_frames:
            pad_width = target_time_frames - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        
        spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0)  # Add channel dim
        label_tensor = torch.LongTensor([label])
        
        return spec_tensor, label_tensor[0]

class Trainer:
    
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        
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
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self, num_epochs: int, save_path: str):
        
        best_accuracy = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch()
            
            val_loss, val_accuracy, predictions, targets = self.validate()
            
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model(save_path, epoch, val_accuracy)
                print(f"New best model saved with accuracy: {val_accuracy:.4f}")
            
            if (epoch + 1) % 10 == 0 and len(set(targets)) > 1:
                unique_classes = sorted(set(targets + predictions))
                class_names = [self.model.class_names.get(i, f'class_{i}') for i in unique_classes]
                try:
                    report = classification_report(
                        targets, predictions, 
                        labels=unique_classes,
                        target_names=class_names,
                        zero_division=0
                    )
                    print(f"\nClassification Report:\n{report}")
                except Exception as e:
                    print(f"Could not generate classification report: {e}")
        
        print(f"\nTraining completed. Best accuracy: {best_accuracy:.4f}")
        return best_accuracy
    
    def save_model(self, save_path: str, epoch: int, accuracy: float):
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, save_path)
    
    def plot_training_history(self, save_path: str = None):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training plots saved to {save_path}")
        
        plt.show()

def main():
    
    parser = argparse.ArgumentParser(description='Train Underwater Acoustic Classifier')
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--model-type', type=str, choices=['classifier', 'detector'],
                       default='classifier', help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_dataset = UnderwaterAcousticDataset(args.data_dir, split='train', augment=True)
    val_dataset = UnderwaterAcousticDataset(args.data_dir, split='val', augment=False)
    
    if len(train_dataset) == 0:
        print("Error: No training data found!")
        return
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0  # Avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0  # Avoid multiprocessing issues
    )
    
    sample_dataset = UnderwaterAcousticDataset(args.data_dir, split='train', augment=False)
    num_classes = len(sample_dataset.class_to_id)
    print(f"Training with {num_classes} classes: {list(sample_dataset.class_to_id.keys())}")
    
    if args.model_type == 'classifier':
        model = UnderwaterAcousticClassifier(num_classes=num_classes)
        save_path = os.path.join(args.save_dir, 'best_classifier.pth')
    else:
        model = CNNBiLSTMDetector()
        save_path = os.path.join(args.save_dir, 'best_detector.pth')
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate
    )
    
    print("Starting training...")
    best_accuracy = trainer.train(args.epochs, save_path)
    
    plot_path = os.path.join(args.save_dir, f'training_history_{args.model_type}.png')
    trainer.plot_training_history(plot_path)
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {save_path}")

if __name__ == '__main__':
    main()
