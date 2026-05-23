#!/bin/bash

# ============================================================================
# EVALUATION SCRIPT
# Evaluates the trained DA-BiGRU model
# ============================================================================

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PROCESSED_DIR="$PROJECT_ROOT/data/processed"
MODELS_DIR="$PROJECT_ROOT/models"
RESULTS_DIR="$PROJECT_ROOT/results"

echo "=================================================="
echo "📊 MODEL EVALUATION SCRIPT"
echo "=================================================="
echo "Project Root: $PROJECT_ROOT"

# Check if model checkpoint exists
if [ ! -f "$MODELS_DIR/checkpoints/best_model.pth" ]; then
    echo ""
    echo "❌ Error: No trained model found at $MODELS_DIR/checkpoints/best_model.pth"
    echo "   Please run: bash scripts/train.sh"
    exit 1
fi

echo "✅ Model checkpoint found"
echo ""

# ============================================================================
# RUN EVALUATION PYTHON SCRIPT
# ============================================================================

python "$PROJECT_ROOT/scripts/evaluate.py" \
    --data_dir "$DATA_PROCESSED_DIR" \
    --models_dir "$MODELS_DIR" \
    --results_dir "$RESULTS_DIR"

echo ""
echo "✅ Evaluation complete!"
echo "   Results saved to: $RESULTS_DIR/"
