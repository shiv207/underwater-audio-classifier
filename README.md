# Underwater Audio Classifier

AI-powered underwater acoustic event detection and classification system.

## Overview

Detects and classifies underwater sounds into four categories: vessels, marine animals, natural sounds, and anthropogenic sources. Built for military applications and marine research.

## Quick Start

```bash
# Setup
python setup.py

# Process audio
python main.py --input audio.wav --output results.json

# Train model
python train.py --data-dir data/training --epochs 50
```

## Features

- Real-time audio processing (16kHz)
- CNN-Transformer architecture
- PS-12 compliant JSON output
- Docker deployment ready
- Energy and ML-based detection

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch, Librosa

## Usage

### Basic Classification
```bash
python main.py --input sonar.wav --output results.json
```

### Training
```bash
python train.py --data-dir data/training --batch-size 8
```

### Docker
```bash
docker build -t uda-classifier .
docker run -v /data:/app/data uda-classifier python main.py --input data/audio.wav
```

## Output Format

```json
{
  "annotations": [
    {
      "id": 1,
      "category_id": 4,
      "start_time": 2,
      "end_time": 8,
      "score": 0.95
    }
  ]
}
```

## Categories

1. **Vessels** - Ships, submarines
2. **Marine Animals** - Whale calls, dolphin sounds  
3. **Natural Sounds** - Earthquakes, ocean ambient
4. **Other Anthropogenic** - Sonar, torpedoes, motors

## Architecture

- **Preprocessing:** 16kHz conversion, mel-spectrograms
- **Detection:** CNN-BiLSTM temporal modeling
- **Classification:** CNN-Transformer hybrid
- **Evaluation:** IER metrics with pyannote

## Performance

- 100% accuracy on training data
- Real-time processing capability
- ~200MB model size
- CPU/GPU compatible

## Repository

```
src/           # Core modules
data/          # Training data
models/        # Trained models
main.py        # Inference
train.py       # Training
setup.py       # Installation
```

## Applications

** Military:** Sonar detection, vessel identification, threat assessment  
**Research:** Marine biology, environmental monitoring, seismic analysis

## License

MIT License
