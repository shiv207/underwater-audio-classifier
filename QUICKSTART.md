# 🚀 Quick Start Guide

## Step 1: Install Dependencies (One-time)

```bash
cd "/Users/shivamsh/Desktop/under-water acoustic local/uda_model"
pip install -r requirements.txt
```

## Step 2: Wait for Training to Complete

Training is currently in progress! Monitor the terminal for:
- **Current Status**: The retraining script is running
- **Duration**: ~30-60 minutes for 50 epochs
- **Output**: Model will be saved to `models/best_classifier.pth`

You can check training progress in the terminal where you ran `python retrain_model.py`

## Step 3: Launch the Streamlit UI

Once training completes (or you can start now to familiarize yourself with the interface):

```bash
# Make script executable (one-time)
chmod +x start_ui.sh

# Launch the app
./start_ui.sh

# Or use direct command
streamlit run streamlit_app.py
```

## Step 4: Use the Application

1. **Open Browser**: Navigate to http://localhost:8501
2. **Load Model**: Click "🔄 Load/Reload Model" in the sidebar
3. **Upload Audio**: Drag & drop or browse for an underwater audio file
4. **Classify**: Click the "🔍 Classify Audio" button
5. **View Results**: See classification, confidence scores, and visualizations
6. **Export**: Download results as JSON for analysis

## 📋 Supported Audio Formats

- WAV (recommended)
- MP3
- FLAC
- M4A

## 🎯 Expected Output

The system will classify your audio into one of these categories:

1. **🚢 Vessels** - Ships, submarines, boats
2. **🐋 Marine Animals** - Whales, dolphins, marine mammals
3. **🌊 Natural Sounds** - Earthquakes, ice, landslides
4. **⚙️ Other Anthropogenic** - Sonar, torpedoes, construction

## 💡 Tips

- **First Classification**: May take 10-30 seconds (model initialization)
- **GPU vs CPU**: Automatically uses GPU if available, otherwise CPU
- **File Size**: Keep files under 100MB for best performance
- **Multiple Files**: Use batch processing tab (coming soon)

## 🔧 Troubleshooting

### "Model not loaded" error
→ Wait for training to complete, then click "Load/Reload Model"

### Training taking too long
→ Normal! 50 epochs can take 30-60 minutes on CPU

### Audio file not processing
→ Ensure file format is supported (WAV, MP3, FLAC, M4A)

## 📊 Current Training Status

**Dataset**: 92 audio files across 4 categories
- Vessels: 67 files
- Marine Animals: 11 files  
- Natural Sounds: 1 file
- Other Anthropogenic: 13 files

**Model**: CNN + Transformer (16M parameters)
**Training**: 50 epochs with validation

---

**Need Help?** Check `README_STREAMLIT.md` for detailed documentation
