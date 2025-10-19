

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import librosa

class CNNBiLSTMDetector(nn.Module):
    
    
    def __init__(self, 
                 input_dim: int = 128,
                 cnn_channels: List[int] = [32, 64, 128],
                 lstm_hidden: int = 256,
                 lstm_layers: int = 2,
                 dropout: float = 0.3):
        
        super(CNNBiLSTMDetector, self).__init__()
        
        self.input_dim = input_dim
        
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in cnn_channels:
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(dropout)
                )
            )
            in_channels = out_channels
        
        self.cnn_output_dim = self._get_cnn_output_dim()
        
        self.bilstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 2)  # Binary: event/no-event
        )
        
    def _get_cnn_output_dim(self) -> int:
        
        x = torch.randn(1, 1, self.input_dim, 100)  # Dummy input
        for layer in self.cnn_layers:
            x = layer(x)
        return x.size(1) * x.size(2)  # channels * height
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        batch_size, _, n_mels, time_frames = x.size()
        
        for layer in self.cnn_layers:
            x = layer(x)
        
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, height)
        x = x.contiguous().view(batch_size, -1, self.cnn_output_dim)
        
        lstm_out, _ = self.bilstm(x)
        
        output = self.classifier(lstm_out)
        
        return output

class EventDetector:
    
    
    def __init__(self, 
                 model_path: str = None,
                 threshold: float = 0.5,
                 min_duration: float = 1.0,
                 max_gap: float = 0.5,
                 hop_length: int = 512,
                 sr: int = 16000):
        
        self.threshold = threshold
        self.min_duration = min_duration
        self.max_gap = max_gap
        self.hop_length = hop_length
        self.sr = sr
        
        self.model = CNNBiLSTMDetector()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path: str):
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
    
    def detect_events(self, log_mel_spec: np.ndarray) -> List[Dict]:
        
        if log_mel_spec.size == 0:
            return []
        
        spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0)
        spec_tensor = spec_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(spec_tensor)
            probabilities = F.softmax(outputs, dim=-1)
            event_probs = probabilities[0, :, 1].cpu().numpy()  # Event probability
        
        events = self._extract_events(event_probs)
        
        return events
    
    def _extract_events(self, event_probs: np.ndarray) -> List[Dict]:
        
        binary_events = event_probs > self.threshold
        
        events = []
        in_event = False
        start_frame = 0
        
        for i, is_event in enumerate(binary_events):
            if is_event and not in_event:
                start_frame = i
                in_event = True
            elif not is_event and in_event:
                end_frame = i - 1
                events.append((start_frame, end_frame))
                in_event = False
        
        if in_event:
            events.append((start_frame, len(binary_events) - 1))
        
        processed_events = []
        for start_frame, end_frame in events:
            start_time = self._frame_to_time(start_frame)
            end_time = self._frame_to_time(end_frame)
            duration = end_time - start_time
            
            if duration >= self.min_duration:
                score = np.mean(event_probs[start_frame:end_frame+1])
                
                processed_events.append({
                    'start_time': round(start_time),
                    'end_time': round(end_time),
                    'duration': round(duration, 2),
                    'score': float(score)
                })
        
        merged_events = self._merge_nearby_events(processed_events)
        
        return merged_events
    
    def _frame_to_time(self, frame: int) -> float:
        
        return frame * self.hop_length / self.sr
    
    def _merge_nearby_events(self, events: List[Dict]) -> List[Dict]:
        
        if not events:
            return events
        
        merged = []
        current_event = events[0].copy()
        
        for next_event in events[1:]:
            gap = next_event['start_time'] - current_event['end_time']
            
            if gap <= self.max_gap:
                current_event['end_time'] = next_event['end_time']
                current_event['duration'] = current_event['end_time'] - current_event['start_time']
                current_event['score'] = max(current_event['score'], next_event['score'])
            else:
                merged.append(current_event)
                current_event = next_event.copy()
        
        merged.append(current_event)
        
        return merged

class SimpleEnergyDetector:
    
    
    def __init__(self, 
                 threshold_db: float = -30,
                 min_duration: float = 1.0,
                 hop_length: int = 512,
                 sr: int = 16000):
        
        self.threshold_db = threshold_db
        self.min_duration = min_duration
        self.hop_length = hop_length
        self.sr = sr
    
    def detect_events(self, audio: np.ndarray) -> List[Dict]:
        
        if len(audio) == 0:
            return []
        
        frame_length = 2048
        hop_length = self.hop_length
        
        frames = librosa.util.frame(audio, frame_length=frame_length, 
                                  hop_length=hop_length, axis=0)
        energy = np.sum(frames**2, axis=0)
        energy_db = librosa.power_to_db(energy, ref=np.max)
        
        binary_events = energy_db > self.threshold_db
        
        events = []
        in_event = False
        start_frame = 0
        
        for i, is_event in enumerate(binary_events):
            if is_event and not in_event:
                start_frame = i
                in_event = True
            elif not is_event and in_event:
                end_frame = i - 1
                start_time = start_frame * hop_length / self.sr
                end_time = end_frame * hop_length / self.sr
                duration = end_time - start_time
                
                if duration >= self.min_duration:
                    events.append({
                        'start_time': round(start_time),
                        'end_time': round(end_time),
                        'duration': round(duration, 2),
                        'score': 0.8  # Fixed confidence for energy-based detection
                    })
                in_event = False
        
        return events
