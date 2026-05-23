#!/bin/bash

# ============================================================================
# TRAINING SCRIPT
# Trains the DA-BiGRU model
# ============================================================================

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PROCESSED_DIR="$PROJECT_ROOT/data/processed"
MODELS_DIR="$PROJECT_ROOT/models"
RESULTS_DIR="$PROJECT_ROOT/results"

echo "=================================================="
echo "🚀 MODEL TRAINING SCRIPT"
echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Data Dir: $DATA_PROCESSED_DIR"
echo "Models Dir: $MODELS_DIR"
echo "Results Dir: $RESULTS_DIR"

# Create directories
mkdir -p "$MODELS_DIR/checkpoints"
mkdir -p "$RESULTS_DIR"

# Check if processed data exists
if [ ! -d "$DATA_PROCESSED_DIR" ] || [ -z "$(ls -A $DATA_PROCESSED_DIR)" ]; then
    echo ""
    echo "❌ Error: No processed data found at $DATA_PROCESSED_DIR"
    echo "   Please run: bash scripts/prepare_data.sh"
    exit 1
fi

echo "✅ Processed data found"
echo ""

# ============================================================================
# RUN TRAINING PYTHON SCRIPT
# ============================================================================

python "$PROJECT_ROOT/scripts/train.py" \
    --data_dir "$DATA_PROCESSED_DIR" \
    --models_dir "$MODELS_DIR" \
    --results_dir "$RESULTS_DIR" \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001

echo ""
echo "✅ Training complete!"
echo "   Model checkpoints: $MODELS_DIR/checkpoints/"
echo "   Results: $RESULTS_DIR/"
