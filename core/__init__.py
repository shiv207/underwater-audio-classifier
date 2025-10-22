"""
Core module for underwater acoustic classification system.
"""

from .models import UnderwaterAcousticClassifier, AcousticClassifier
from .data import AudioPreprocessor, AdvancedAudioAugmentation
from .training import ClassBalancedLoss, AdvancedTrainer

__all__ = [
    'UnderwaterAcousticClassifier',
    'AcousticClassifier', 
    'AudioPreprocessor',
    'AdvancedAudioAugmentation',
    'ClassBalancedLoss',
    'AdvancedTrainer'
]
