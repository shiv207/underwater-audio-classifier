from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data' / 'training'
MODELS_DIR = Path(__file__).parent / 'models'

# Audio processing
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 160
TARGET_LENGTH = 1024

# Training
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 7

# System
NUM_WORKERS = 2
PIN_MEMORY = False

# Model
DROPOUT = 0.3

MODELS_DIR.mkdir(exist_ok=True)
