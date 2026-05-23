"""
Global constants and configuration defaults.
"""
import torch

# Model & Training Defaults
SEED = 2026
SEQ_LEN = 24                # 24 hours of context
BATCH_SIZE = 64             # Batch size
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10

# Feature Configuration
FEATURE_COLS = [
    'WS50M', 'WS10M', 'WS100M', 'T2M', 'PS',
    'WindShear', 'AirDensity', 'TurbulenceIntensity',
    'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos'
]
TARGET_COL_NAME = 'WS50M'

# Data Split
TRAIN_TEST_SPLIT = 0.8

# Model Architecture
MODEL_HIDDEN_DIM = 64
MODEL_N_LAYERS = 2
MODEL_DROPOUT = 0.2

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths (will be overridden at runtime)
PROJECT_ROOT = None
DATA_RAW_DIR = None
DATA_PROCESSED_DIR = None
MODELS_DIR = None
RESULTS_DIR = None
