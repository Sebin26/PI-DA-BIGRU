#!/bin/bash

# ============================================================================
# MAIN ORCHESTRATION SCRIPT
# Runs the complete pipeline: prepare data -> train -> evaluate
# ============================================================================

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  DA-BiGRU WIND SPEED FORECASTING PIPELINE                  ║"
echo "║  Complete End-to-End Execution                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================
echo ""
echo "STEP 1/3: DATA PREPARATION"
echo "─────────────────────────────────────────────────────────────"

if bash "$PROJECT_ROOT/scripts/prepare_data.sh"; then
    echo "✅ Data preparation successful"
else
    echo "❌ Data preparation failed"
    exit 1
fi

# ============================================================================
# STEP 2: MODEL TRAINING
# ============================================================================
echo ""
echo "STEP 2/3: MODEL TRAINING"
echo "─────────────────────────────────────────────────────────────"

if bash "$PROJECT_ROOT/scripts/train.sh"; then
    echo "✅ Training successful"
else
    echo "❌ Training failed"
    exit 1
fi

# ============================================================================
# STEP 3: MODEL EVALUATION
# ============================================================================
echo ""
echo "STEP 3/3: MODEL EVALUATION"
echo "─────────────────────────────────────────────────────────────"

if bash "$PROJECT_ROOT/scripts/evaluate.sh"; then
    echo "✅ Evaluation successful"
else
    echo "❌ Evaluation failed"
    exit 1
fi

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ PIPELINE COMPLETE                                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Output Directories:"
echo "   Data: $PROJECT_ROOT/data/processed/"
echo "   Models: $PROJECT_ROOT/models/checkpoints/"
echo "   Results: $PROJECT_ROOT/results/"
echo ""
