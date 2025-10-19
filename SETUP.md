# Virtual Environment Setup Guide

## ✅ Setup Complete!

Your virtual environment has been created and all packages are installed.

## 📦 Installed Packages

- **PyTorch 2.9.0** - Deep learning framework
- **Streamlit** - Web UI framework
- **Librosa** - Audio processing
- **NumPy, SciPy, Scikit-learn** - Scientific computing
- **Matplotlib, Plotly** - Visualization
- **And 60+ other dependencies**

## 🚀 How to Use

### Option 1: Activate Virtual Environment (Recommended)

```bash
cd "/Users/shivamsh/Desktop/under-water acoustic local/uda_model"
source venv/bin/activate
```

Once activated, you can run:

```bash
# Train the model
python train.py --data-dir data/training --epochs 50

# Retrain with wrapper script
python retrain_model.py

# Launch Streamlit UI
streamlit run streamlit_app.py

# Run inference
python main.py --input audio_file.wav --classifier-model models/best_classifier.pth
```

To deactivate:
```bash
deactivate
```

### Option 2: Use Startup Scripts

```bash
# For Streamlit UI
./start_ui.sh

# For quick activation with info
./activate_venv.sh
```

## 🔧 Training Commands

### Full Training (from scratch)
```bash
source venv/bin/activate
python train.py \
    --data-dir data/training \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 0.001 \
    --device auto
```

### Quick Retraining
```bash
source venv/bin/activate
python retrain_model.py
```

### Streamlit UI
```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

## 📊 Check Training Status

If you still have training running from before:

```bash
# Check if training process is still running
ps aux | grep retrain_model.py

# To kill if needed (replace <PID> with actual process ID)
kill <PID>
```

## 🐍 Python Paths

**Virtual Environment Python:**
```bash
/Users/shivamsh/Desktop/under-water acoustic local/uda_model/venv/bin/python
```

**System Python:**
```bash
/opt/homebrew/bin/python3
```

⚠️ **Always use the venv Python** for this project!

## 🔍 Verify Installation

```bash
source venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print('Streamlit: OK')"
python -c "import librosa; print('Librosa: OK')"
```

## 💡 Tips

1. **Always activate venv** before running Python scripts
2. **Use `./start_ui.sh`** for the easiest UI launch
3. **Check training completion** - Model saved to `models/best_classifier.pth`
4. **GPU not available** - Training will use CPU (slower but works fine)

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
→ Make sure virtual environment is activated:
```bash
source venv/bin/activate
```

### "venv/bin/activate: No such file or directory"
→ Make sure you're in the correct directory:
```bash
cd "/Users/shivamsh/Desktop/under-water acoustic local/uda_model"
```

### Want to reinstall packages?
```bash
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Want to start fresh?
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📝 Version Control Notes

### Latest Update (Oct 19, 2025)

**Fixed Git Push Issues:**
- Removed large data files from version control (spectrograms.npy - 256.76 MB exceeded GitHub's 100 MB limit)
- Updated `.gitignore` to exclude processed data files:
  - `*.npy`, `*.npz` files
  - `data/processed/` directory
  - `data/training_annotations.json`
  - `data/dataset_summary_report.json`

**Important:** Processed data files remain on your local machine but are no longer tracked by git. This prevents repository bloat and allows for smoother collaboration.

**Files excluded from git:**
- Audio files (`.wav`, `.mp3`, `.flac`, `.m4a`)
- Trained models (`.pth`, `.pt` in models/)
- Processed spectrograms and metadata
- Virtual environment (`venv/`)

---

**Virtual Environment Location:**  
`/Users/shivamsh/Desktop/under-water acoustic local/uda_model/venv/`

**Status:** ✅ Ready to use!
