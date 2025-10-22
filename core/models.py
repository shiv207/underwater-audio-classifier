"""
Core model definitions for underwater acoustic classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import math
from einops import rearrange


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class CNNFeatureExtractor(nn.Module):
    """CNN backbone for feature extraction from spectrograms."""
    
    def __init__(self, input_channels: int = 1, base_channels: int = 32):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                nn.BatchNorm2d(base_channels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
                nn.BatchNorm2d(base_channels*2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            nn.Sequential(
                nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
                nn.BatchNorm2d(base_channels*4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
                nn.BatchNorm2d(base_channels*4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ),
            nn.Sequential(
                nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
                nn.BatchNorm2d(base_channels*8),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
                nn.BatchNorm2d(base_channels*8),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4))
            )
        ])
        
        self.output_dim = base_channels * 8 * 16
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return x


class TransformerClassifier(nn.Module):
    """Transformer-based classifier head."""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 num_classes: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x


class UnderwaterAcousticClassifier(nn.Module):
    """Main underwater acoustic classification model."""
    
    def __init__(self, 
                 num_classes: int = 4,
                 input_channels: int = 1,
                 cnn_base_channels: int = 32,
                 transformer_dim: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.cnn_backbone = CNNFeatureExtractor(input_channels, cnn_base_channels)
        self.transformer_classifier = TransformerClassifier(
            input_dim=self.cnn_backbone.output_dim,
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            num_classes=num_classes,
            dropout=dropout
        )
        self.class_names = {
            0: 'vessels',
            1: 'marine_animals', 
            2: 'natural_sounds',
            3: 'other_anthropogenic'
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        if width > 1:
            segments = []
            segment_size = min(width, 128)
            for i in range(0, width, segment_size):
                end_idx = min(i + segment_size, width)
                segment = x[:, :, :, i:end_idx]
                if segment.size(-1) < segment_size:
                    pad_size = segment_size - segment.size(-1)
                    segment = F.pad(segment, (0, pad_size))
                segment_features = self.cnn_backbone(segment)
                segments.append(segment_features)
            if len(segments) > 1:
                sequence_features = torch.stack(segments, dim=1)
            else:
                sequence_features = segments[0].unsqueeze(1)
        else:
            cnn_features = self.cnn_backbone(x)
            sequence_features = cnn_features.unsqueeze(1)
        output = self.transformer_classifier(sequence_features)
        return output


class AcousticClassifier:
    """High-level classifier interface."""
    
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UnderwaterAcousticClassifier()
        self.model.to(self.device)
        if model_path:
            self.load_model(model_path)
        self.model.eval()
        self.id_to_class = {
            0: 'vessels',
            1: 'marine_animals',
            2: 'natural_sounds', 
            3: 'other_anthropogenic'
        }
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
    
    def load_model(self, model_path: str):
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded classifier model from {model_path}")
        except Exception as e:
            print(f"Could not load classifier model: {e}")
    
    def classify_spectrogram(self, log_mel_spec: np.ndarray) -> Dict:
        """Classify a log-mel spectrogram."""
        if log_mel_spec.size == 0:
            return {'category_id': 4, 'confidence': 0.0, 'probabilities': {}}
        spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0)
        spec_tensor = spec_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(spec_tensor)
            probabilities = F.softmax(logits, dim=1)
        probs = probabilities[0].cpu().numpy()
        predicted_class = np.argmax(probs)
        confidence = float(probs[predicted_class])
        
        category_id = predicted_class + 1
        
        prob_dict = {}
        for i, prob in enumerate(probs):
            class_name = self.model.class_names[i]
            prob_dict[class_name] = float(prob)
        return {
            'category_id': category_id,
            'confidence': confidence,
            'probabilities': prob_dict,
            'predicted_class_name': self.id_to_class[predicted_class]
        }
    
    def classify_event(self, log_mel_spec: np.ndarray, start_frame: int, end_frame: int) -> Dict:
        """Classify a specific event in the spectrogram."""
        if log_mel_spec.size == 0 or start_frame >= end_frame:
            return {'category_id': 4, 'confidence': 0.0, 'probabilities': {}}
        event_spec = log_mel_spec[:, start_frame:end_frame+1]
        return self.classify_spectrogram(event_spec)
