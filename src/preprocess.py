import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
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
        if len(audio) == 0:
            return audio
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def bandpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
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
    
    def segment_audio(self, audio: np.ndarray, sr: int, segment_length: float = 10.0, 
                     overlap: float = 0.5) -> list:
        segment_samples = int(segment_length * sr)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        for start in range(0, len(audio) - segment_samples + 1, hop_samples):
            end = start + segment_samples
            segment = audio[start:end]
            segments.append({
                'audio': segment,
                'start_time': start / sr,
                'end_time': end / sr
            })
        
        if len(audio) % segment_samples != 0:
            segment = audio[-segment_samples:]
            segments.append({
                'audio': segment,
                'start_time': (len(audio) - segment_samples) / sr,
                'end_time': len(audio) / sr
            })
        
        return segments

def apply_spec_augment(log_mel_spec: np.ndarray, 
                      freq_mask_param: int = 15,
                      time_mask_param: int = 35,
                      num_freq_masks: int = 1,
                      num_time_masks: int = 1) -> np.ndarray:
    spec = log_mel_spec.copy()
    n_mels, n_frames = spec.shape
    
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, n_mels - f)
        spec[f0:f0+f, :] = 0
    
    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, n_frames - t)
        spec[:, t0:t0+t] = 0
    
    return spec

def add_noise(audio: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, noise_factor, audio.shape)
    return audio + noise
