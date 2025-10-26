import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from simple_config import *


def load_audio_files(data_dir):
    samples = []
    class_names = ['marine_animals', 'natural_sounds', 'other_anthropogenic', 'vessels']
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue
        
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.webm']:
            audio_files.extend(list(class_dir.rglob(ext)))
        
        print(f"{class_name}: {len(audio_files)} files")
        
        for audio_file in audio_files:
            samples.append((str(audio_file), class_to_id[class_name]))
    
    print(f"\nTotal: {len(samples)} files, {len(class_to_id)} classes")
    return samples, class_to_id


def process_audio(file_path, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=10.0)
        
        # Data augmentation for better generalization
        if augment and np.random.random() > 0.5:
            if np.random.random() > 0.5:
                stretch_factor = np.random.uniform(0.8, 1.2)
                audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
            
            if np.random.random() > 0.5:
                pitch_shift = np.random.uniform(-2, 2)
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            
            if np.random.random() > 0.5:
                noise_factor = np.random.uniform(0.001, 0.01)
                noise = np.random.normal(0, noise_factor, audio.shape)
                audio = audio + noise
        
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        if log_mel.shape[1] < TARGET_LENGTH:
            pad = TARGET_LENGTH - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode='constant')
        else:
            if augment and log_mel.shape[1] > TARGET_LENGTH:
                start = np.random.randint(0, log_mel.shape[1] - TARGET_LENGTH)
                log_mel = log_mel[:, start:start + TARGET_LENGTH]
            else:
                log_mel = log_mel[:, :TARGET_LENGTH]
        
        return log_mel
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((N_MELS, TARGET_LENGTH))


class EnhancedDataset(Dataset):
    
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
        
        # Calculate class weights for balanced training
        class_counts = {}
        for _, label in samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(samples)
        self.class_weights = {}
        for class_id, count in class_counts.items():
            self.class_weights[class_id] = total_samples / (len(class_counts) * count)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        spec = process_audio(file_path, augment=self.augment)
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)
        label_tensor = torch.LongTensor([label])[0]
        weight = torch.tensor(self.class_weights[label], dtype=torch.float32)
        
        return spec_tensor, label_tensor, weight


def create_dataloaders(data_dir):
    samples, class_to_id = load_audio_files(data_dir)
    
    if len(samples) == 0:
        raise ValueError("No samples found!")
    
    # Stratified split for balanced classes
    from collections import defaultdict
    class_samples = defaultdict(list)
    for sample in samples:
        class_samples[sample[1]].append(sample)
    
    train_samples = []
    val_samples = []
    
    np.random.seed(42)
    for class_id, class_sample_list in class_samples.items():
        np.random.shuffle(class_sample_list)
        split = int(0.8 * len(class_sample_list))
        train_samples.extend(class_sample_list[:split])
        val_samples.extend(class_sample_list[split:])
    
    print(f"\nTrain: {len(train_samples)}, Val: {len(val_samples)}")
    
    train_dataset = EnhancedDataset(train_samples, augment=True)
    val_dataset = EnhancedDataset(val_samples, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    return train_loader, val_loader, class_to_id
