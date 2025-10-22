"""
Data processing and augmentation utilities for underwater acoustic classification.
"""

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """Audio preprocessing pipeline for underwater acoustic data."""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 f_min: float = 20.0,
                 f_max: float = 8000.0):
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
    
    def load_and_convert_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and convert audio to target format."""
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {str(e)}")
    
    def normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude."""
        if len(audio) == 0:
            return audio
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def bandpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply bandpass filter to audio."""
        if len(audio) == 0:
            return audio
        
        nyquist = sr / 2
        low = max(self.f_min / nyquist, 0.01)
        high = min(self.f_max / nyquist, 0.99)
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_audio = signal.filtfilt(b, a, audio)
            return filtered_audio
        except Exception:
            return audio
    
    def extract_log_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract log-mel spectrogram from audio."""
        if len(audio) == 0:
            return np.zeros((self.n_mels, 1))
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    
    def process_audio_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Complete audio processing pipeline."""
        audio, sr = self.load_and_convert_audio(file_path)
        audio = self.normalize_amplitude(audio)
        audio = self.bandpass_filter(audio, sr)
        log_mel_spec = self.extract_log_mel_spectrogram(audio, sr)
        
        metadata = {
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'n_samples': len(audio),
            'spectrogram_shape': log_mel_spec.shape
        }
        
        return audio, log_mel_spec, metadata


class AdvancedAudioAugmentation:
    """Advanced audio augmentation techniques for underwater acoustics."""
    
    @staticmethod
    def time_stretch(audio: np.ndarray, rate_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Time stretching without changing pitch."""
        rate = np.random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    @staticmethod
    def pitch_shift(audio: np.ndarray, sr: int = 16000, 
                   semitones_range: Tuple[int, int] = (-2, 2)) -> np.ndarray:
        """Pitch shifting without changing tempo."""
        n_steps = np.random.randint(*semitones_range)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def add_ocean_noise(audio: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add realistic ocean ambient noise."""
        noise = np.random.randn(len(audio))
        
        # Apply pink noise filter (approximate)
        from scipy.signal import lfilter
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink_noise = lfilter(b, a, noise)
        
        # Normalize and scale
        pink_noise = pink_noise / np.max(np.abs(pink_noise)) * noise_level
        
        return audio + pink_noise
    
    @staticmethod
    def frequency_masking(spec: np.ndarray, max_mask_size: int = 10, 
                         num_masks: int = 2) -> np.ndarray:
        """Mask random frequency bands."""
        spec = spec.copy()
        freq_bins = spec.shape[0]
        
        for _ in range(num_masks):
            mask_size = np.random.randint(1, max_mask_size)
            mask_start = np.random.randint(0, freq_bins - mask_size)
            spec[mask_start:mask_start + mask_size, :] = 0
        
        return spec
    
    @staticmethod
    def time_masking(spec: np.ndarray, max_mask_size: int = 20, 
                    num_masks: int = 2) -> np.ndarray:
        """Mask random time segments."""
        spec = spec.copy()
        time_frames = spec.shape[1]
        
        for _ in range(num_masks):
            mask_size = np.random.randint(1, max_mask_size)
            mask_start = np.random.randint(0, time_frames - mask_size)
            spec[:, mask_start:mask_start + mask_size] = 0
        
        return spec


def apply_spec_augment(log_mel_spec: np.ndarray, 
                      freq_mask_param: int = 15,
                      time_mask_param: int = 35,
                      num_freq_masks: int = 1,
                      num_time_masks: int = 1) -> np.ndarray:
    """Apply SpecAugment to spectrogram."""
    spec = log_mel_spec.copy()
    n_mels, n_frames = spec.shape
    
    # Frequency masking
    for _ in range(num_freq_masks):
        f = min(freq_mask_param, n_mels - 1)
        if f > 0:
            f = np.random.randint(1, f + 1)
            f0 = np.random.randint(0, max(1, n_mels - f))
            spec[f0:f0+f, :] = 0
    
    # Time masking
    for _ in range(num_time_masks):
        t = min(time_mask_param, n_frames - 1)
        if t > 0:
            t = np.random.randint(1, t + 1)
            t0 = np.random.randint(0, max(1, n_frames - t))
            spec[:, t0:t0+t] = 0
    
    return spec


def add_noise(audio: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to audio."""
    noise = np.random.normal(0, noise_factor, audio.shape)
    return audio + noise
