# Underwater Acoustic Classification System

A deep learning system for classifying underwater acoustic sounds into four categories:
- Vessels
- Marine Animals  
- Natural Sounds
- Other Anthropogenic

## Features

- **CNN-Transformer Architecture**: Combines CNN feature extraction with transformer classification
- **Advanced Data Augmentation**: SpecAugment, mixup, and noise injection
- **Class-Balanced Training**: Handles severe class imbalance with focal loss
- **Interactive Web App**: Streamlit-based interface for real-time classification
- **Comprehensive Training**: Support for transfer learning and fine-tuning

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Organize your audio files in the following structure:
```
data/training/
├── vessels/
├── marine_animals/
├── natural_sounds/
└── other_anthropogenic/
```

### 3. Train Model

```bash
python train.py --data-dir data/training --epochs 50
```

### 4. Run Web App

```bash
streamlit run app.py
```

## Architecture

The system uses a hybrid CNN-Transformer architecture:

- **CNN Backbone**: Extracts spatial features from log-mel spectrograms
- **Transformer Classifier**: Processes sequential features for final classification
- **Advanced Augmentation**: SpecAugment, mixup, and noise injection for robust training

## Training Features

- **Focal Loss**: Handles class imbalance effectively
- **Label Smoothing**: Improves generalization
- **Mixup Augmentation**: Increases robustness
- **Balanced Sampling**: Ensures equal representation of all classes
- **Early Stopping**: Prevents overfitting

## File Structure

```
uda_model/
├── core/                    # Core modules
│   ├── __init__.py
│   ├── models.py           # Model definitions
│   ├── data.py             # Data processing
│   └── training.py         # Training utilities
├── data/                   # Training data
├── models/                 # Saved models
├── train.py               # Training script
├── app.py                 # Streamlit app
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Usage Examples

### Training with Custom Parameters

```bash
python train.py \
    --data-dir data/training \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --no-focal-loss \
    --no-mixup
```

### Using the Classifier Programmatically

```python
from core import AcousticClassifier, AudioPreprocessor

# Load trained model
classifier = AcousticClassifier('models/best_model.pth')

# Process audio file
preprocessor = AudioPreprocessor()
audio, log_mel_spec, metadata = preprocessor.process_audio_file('audio.wav')

# Classify
result = classifier.classify_spectrogram(log_mel_spec)
print(f"Predicted: {result['predicted_class_name']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Performance

The system achieves high accuracy on underwater acoustic classification tasks:
- **Balanced Accuracy**: >85% on test set
- **Per-Class Performance**: Consistent across all four categories
- **Robustness**: Handles various audio qualities and durations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.