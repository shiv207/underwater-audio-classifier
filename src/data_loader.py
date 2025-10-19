

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from sklearn.model_selection import train_test_split

class UnderwaterAcousticDataset(Dataset):
    
    
    def __init__(self, 
                 spectrograms: np.ndarray,
                 labels: np.ndarray,
                 metadata: List[Dict],
                 transform=None):
        
        self.spectrograms = spectrograms
        self.labels = labels
        self.metadata = metadata
        self.transform = transform
        
        assert len(spectrograms) == len(labels) == len(metadata), \
            "Spectrograms, labels, and metadata must have same length"
    
    def __len__(self) -> int:
        return len(self.spectrograms)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        spectrogram = self.spectrograms[idx]
        label = self.labels[idx]
        meta = self.metadata[idx]
        
        spectrogram = torch.FloatTensor(spectrogram)
        label = torch.LongTensor([label])
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label.squeeze(), meta

class DataLoaderManager:
    
    
    def __init__(self, data_dir: str = "data"):
        
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.annotations_file = self.data_dir / "training_annotations.json"
        
        self.category_mapping = {
            'aquatic_mammals': 0,
            'natural_sounds': 1,
            'other_anthropogenic': 2,
            'vessels': 3
        }
        
        self.num_classes = len(self.category_mapping)
    
    def load_processed_data(self) -> Tuple[np.ndarray, List[Dict]]:
        
        spectrograms_file = self.processed_dir / "spectrograms.npy"
        if not spectrograms_file.exists():
            raise FileNotFoundError(f"Spectrograms file not found: {spectrograms_file}")
        
        spectrograms = np.load(spectrograms_file)
        
        metadata_file = self.processed_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return spectrograms, metadata
    
    def prepare_labels(self, metadata: List[Dict]) -> np.ndarray:
        
        labels = []
        for meta in metadata:
            category_name = meta['category_name']
            label = self.category_mapping.get(category_name, -1)
            if label == -1:
                raise ValueError(f"Unknown category: {category_name}")
            labels.append(label)
        
        return np.array(labels)
    
    def create_train_val_split(self, 
                              spectrograms: np.ndarray,
                              labels: np.ndarray,
                              metadata: List[Dict],
                              val_size: float = 0.2,
                              random_state: int = 42) -> Tuple:
        
        indices = np.arange(len(spectrograms))
        
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=val_size, 
            random_state=random_state,
            stratify=labels  # Ensure balanced split across classes
        )
        
        train_spectrograms = spectrograms[train_idx]
        val_spectrograms = spectrograms[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_metadata = [metadata[i] for i in train_idx]
        val_metadata = [metadata[i] for i in val_idx]
        
        return (train_spectrograms, val_spectrograms, 
                train_labels, val_labels, 
                train_metadata, val_metadata)
    
    def get_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(labels)
        class_weights = compute_class_weight(
            'balanced', 
            classes=classes, 
            y=labels
        )
        
        return torch.FloatTensor(class_weights)
    
    def create_data_loaders(self, 
                           batch_size: int = 32,
                           val_size: float = 0.2,
                           num_workers: int = 4,
                           random_state: int = 42) -> Tuple[DataLoader, DataLoader, Dict]:
        
        spectrograms, metadata = self.load_processed_data()
        labels = self.prepare_labels(metadata)
        
        print(f"Loaded {len(spectrograms)} samples")
        print(f"Spectrogram shape: {spectrograms.shape}")
        print(f"Number of classes: {self.num_classes}")
        
        (train_spectrograms, val_spectrograms, 
         train_labels, val_labels, 
         train_metadata, val_metadata) = self.create_train_val_split(
            spectrograms, labels, metadata, val_size, random_state
        )
        
        print(f"Train samples: {len(train_spectrograms)}")
        print(f"Validation samples: {len(val_spectrograms)}")
        
        train_dataset = UnderwaterAcousticDataset(
            train_spectrograms, train_labels, train_metadata
        )
        val_dataset = UnderwaterAcousticDataset(
            val_spectrograms, val_labels, val_metadata
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        class_weights = self.get_class_weights(train_labels)
        
        info = {
            'num_classes': self.num_classes,
            'input_shape': spectrograms.shape[1:],  # (n_mels, n_frames)
            'train_samples': len(train_spectrograms),
            'val_samples': len(val_spectrograms),
            'class_weights': class_weights,
            'category_mapping': self.category_mapping,
            'class_distribution': {
                'train': {i: int(np.sum(train_labels == i)) for i in range(self.num_classes)},
                'val': {i: int(np.sum(val_labels == i)) for i in range(self.num_classes)}
            }
        }
        
        return train_loader, val_loader, info
    
    def get_sample_batch(self, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        
        spectrograms, metadata = self.load_processed_data()
        labels = self.prepare_labels(metadata)
        
        indices = random.sample(range(len(spectrograms)), min(batch_size, len(spectrograms)))
        
        sample_spectrograms = torch.FloatTensor(spectrograms[indices])
        sample_labels = torch.LongTensor(labels[indices])
        sample_metadata = [metadata[i] for i in indices]
        
        return sample_spectrograms, sample_labels, sample_metadata

def main():
    
    
    data_dir = "/Users/shivamsh/Desktop/under-water acoustic local/uda_model/data"
    manager = DataLoaderManager(data_dir)
    
    print("=== Testing Data Loader ===")
    
    train_loader, val_loader, info = manager.create_data_loaders(batch_size=16)
    
    print(f"\nDataset Info:")
    print(f"  Input shape: {info['input_shape']}")
    print(f"  Number of classes: {info['num_classes']}")
    print(f"  Train samples: {info['train_samples']}")
    print(f"  Validation samples: {info['val_samples']}")
    
    print(f"\nClass distribution (train):")
    for class_id, count in info['class_distribution']['train'].items():
        category_name = [k for k, v in info['category_mapping'].items() if v == class_id][0]
        print(f"  {category_name} (id={class_id}): {count} samples")
    
    print(f"\nClass weights: {info['class_weights']}")
    
    print(f"\n=== Testing Batch Loading ===")
    for batch_idx, (spectrograms, labels, metadata) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Spectrograms shape: {spectrograms.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels.tolist()}")
        
        if batch_idx >= 2:  # Test only first 3 batches
            break
    
    print("\n=== Data Loader Test Completed ===")

if __name__ == "__main__":
    main()
