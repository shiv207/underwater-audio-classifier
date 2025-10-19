# 🌊 Underwater Acoustic Classification System - Streamlit UI

## Overview

A military-grade underwater acoustic classification system with an intuitive Streamlit web interface. The system uses deep learning (CNN + Transformer hybrid architecture) to classify underwater sounds into 4 main categories.

## 🎯 Classification Categories

1. **Vessels** - Ships, boats, submarines, and watercraft
2. **Marine Animals** - Whales, dolphins, and other biological sounds
3. **Natural Sounds** - Earthquakes, underwater landslides, ice cracking
4. **Other Anthropogenic** - Sonar, torpedoes, underwater construction, turbines

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.28+
- Audio processing libraries (librosa, soundfile)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (First Time)

```bash
python retrain_model.py
```

This will:
- Scan the `data/training/` directory for audio files
- Train a CNN+Transformer model
- Save the best model to `models/best_classifier.pth`
- Take approximately 30-60 minutes depending on your hardware

### 3. Launch the Streamlit UI

```bash
# Option 1: Using the startup script
./start_ui.sh

# Option 2: Direct command
streamlit run streamlit_app.py
```

The application will open in your browser at: **http://localhost:8501**

## 💻 Using the Interface

### Main Features

#### 🎵 Upload & Classify Tab
1. Click "Load/Reload Model" in the sidebar (first time only)
2. Upload an audio file (WAV, MP3, FLAC, or M4A)
3. Click "Classify Audio" button
4. View results including:
   - Primary classification with confidence score
   - Detailed probability breakdown
   - Audio waveform visualization
   - Mel spectrogram analysis
   - Event detection details
5. Download results as JSON

#### 📊 Visualizations Provided
- **Audio Waveform**: Time-domain representation
- **Mel Spectrogram**: Frequency-time analysis with 128 mel bands
- **Confidence Chart**: Bar chart showing probabilities for all categories
- **Event Timeline**: Detected acoustic events with timestamps

#### 📥 Export Format

Results are exported in JSON format:

```json
{
  "info": {
    "file_name": "audio_sample.wav",
    "classification_date": "2025-10-19T12:14:00",
    "version": "1.0"
  },
  "audio_info": {
    "duration": 10.5,
    "sample_rate": 22050
  },
  "classification": {
    "category_id": 1,
    "category_name": "Vessels",
    "confidence": 0.952,
    "probabilities": {
      "vessels": 0.952,
      "marine_animals": 0.031,
      "natural_sounds": 0.008,
      "other_anthropogenic": 0.009
    }
  },
  "events": [
    {
      "start_time": 0.0,
      "end_time": 10.5,
      "duration": 10.5,
      "score": 0.85
    }
  ]
}
```

## 📁 Data Structure

Training data should be organized as:

```
uda_model/
├── data/
│   └── training/
│       ├── vessels/              # Ship/boat audio files
│       ├── acquatic_mammels/     # Marine animal sounds
│       ├── natural_sounds/       # Natural phenomena
│       └── other_anthropogenic/  # Man-made non-vessel sounds
├── models/
│   └── best_classifier.pth       # Trained model (generated)
├── streamlit_app.py              # Main UI application
├── retrain_model.py              # Training script
└── requirements.txt              # Dependencies
```

## 🔄 Retraining the Model

To retrain with new data:

1. Add audio files to appropriate category folders in `data/training/`
2. Run the retraining script:

```bash
python retrain_model.py
```

3. Monitor training progress (50 epochs by default)
4. Reload model in Streamlit UI

### Training Parameters

Default configuration:
- **Epochs**: 50
- **Batch Size**: 8
- **Learning Rate**: 0.001
- **Optimizer**: AdamW with Cosine Annealing
- **Validation Split**: 20%

Modify in `retrain_model.py` if needed.

## 🧠 Model Architecture

### CNN Feature Extractor
- 4 convolutional blocks
- Batch normalization and ReLU activation
- Progressive channel expansion: 32 → 64 → 128 → 256
- Adaptive average pooling

### Transformer Classifier
- 4 transformer encoder layers
- 8 attention heads
- 512-dimensional embeddings
- Positional encoding for temporal modeling

### Total Parameters: ~16 million

## 🎨 UI Features

### Responsive Design
- Modern gradient-based category cards
- Interactive Plotly visualizations
- Real-time confidence metrics
- Mobile-friendly layout

### Sidebar Information
- Model loading status
- Category descriptions with examples
- Quick reference instructions

## 📊 Performance Metrics

The model is evaluated on:
- Classification accuracy
- Per-class precision/recall
- Confidence calibration
- Processing time

Training history plots are saved to `models/training_history_classifier.png`

## 🔧 Troubleshooting

### Model Not Loading
- Ensure `models/best_classifier.pth` exists
- Run `python retrain_model.py` to train a new model
- Check file permissions

### Audio Processing Errors
- Supported formats: WAV, MP3, FLAC, M4A
- Maximum file size: 100MB (recommended)
- Sample rates are automatically resampled to 16kHz

### Slow Processing
- First-time processing initializes models (slower)
- CPU mode: ~10-30 seconds per file
- GPU mode (if available): ~2-5 seconds per file

## 🛠️ Advanced Configuration

### Using GPU

If CUDA is available, the system automatically uses GPU. To force CPU:

```python
# In streamlit_app.py, modify load_model function
classifier = AcousticClassifier(model_path=model_path, device='cpu')
```

### Custom Categories

To add new categories:

1. Add folder to `data/training/`
2. Update `CATEGORIES` dict in `streamlit_app.py`
3. Retrain model

## 📝 Example Usage

```python
# Programmatic usage (without UI)
from src.classifier import AcousticClassifier
from src.preprocess import AudioPreprocessor

# Initialize
preprocessor = AudioPreprocessor()
classifier = AcousticClassifier('models/best_classifier.pth')

# Process audio
audio, log_mel_spec, metadata = preprocessor.process_audio_file('sample.wav')
result = classifier.classify_spectrogram(log_mel_spec)

print(f"Category: {result['category_id']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📚 Additional Scripts

- `main.py`: Command-line interface for batch processing
- `train.py`: Low-level training utilities
- `test_system.py`: System validation and testing

## 🔐 Security & Privacy

- All processing is done locally
- No data is transmitted to external servers
- Audio files are temporarily cached during processing
- Results can be exported for audit trails

## 📞 Support

For issues or questions:
1. Check console logs in Streamlit
2. Review training output in terminal
3. Verify data structure matches requirements

## 🎯 Future Enhancements

- [ ] Batch processing for multiple files
- [ ] Real-time audio stream classification
- [ ] Advanced event detection algorithms
- [ ] Model ensemble support
- [ ] Database integration for results
- [ ] API endpoint deployment

## 📄 License

Military-grade underwater acoustic classification system for defense applications.

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Status**: Production Ready 🚀
