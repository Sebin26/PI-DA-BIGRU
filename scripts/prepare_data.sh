#!/bin/bash

# ============================================================================
# DATA PREPARATION SCRIPT
# Downloads and prepares data for model training
# ============================================================================

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_RAW_DIR="$PROJECT_ROOT/data/raw"
DATA_PROCESSED_DIR="$PROJECT_ROOT/data/processed"

echo "=================================================="
echo "📥 DATA PREPARATION SCRIPT"
echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Raw Data Dir: $DATA_RAW_DIR"
echo "Processed Data Dir: $DATA_PROCESSED_DIR"

# Create directories
mkdir -p "$DATA_RAW_DIR"
mkdir -p "$DATA_PROCESSED_DIR"

# ============================================================================
# OPTION 1: Download from Google Drive (requires gdrive or similar setup)
# ============================================================================
# Uncomment and modify the following for Google Drive downloads:

# echo ""
# echo "📥 Downloading data from Google Drive..."
# DRIVE_FILE_ID="YOUR_FILE_ID_HERE"  # Replace with actual file ID
# gdrive download $DRIVE_FILE_ID --path $DATA_RAW_DIR

# ============================================================================
# OPTION 2: Download from URL (if available)
# ============================================================================
# Uncomment the following if data is available via HTTP:

# echo ""
# echo "📥 Downloading data from URL..."
# DATA_URL="https://example.com/merged_weather_data.csv"
# wget -O "$DATA_RAW_DIR/merged_weather_data.csv" "$DATA_URL"

# ============================================================================
# OPTION 3: Local file setup for testing
# ============================================================================
# If you have the CSV file locally, copy it here:

if [ ! -f "$DATA_RAW_DIR/merged_weather_data.csv" ]; then
    echo ""
    echo "⚠️  No data file found at: $DATA_RAW_DIR/merged_weather_data.csv"
    echo ""
    echo "📋 To proceed, please:"
    echo "   1. Download merged_weather_data.csv from your data source"
    echo "   2. Place it in: $DATA_RAW_DIR/"
    echo ""
    echo "   Or run one of the following:"
    echo "   - Uncomment Google Drive download section above"
    echo "   - Uncomment HTTP download section above"
    echo ""
    exit 1
else
    echo "✅ Data file found!"
fi

# ============================================================================
# RUN PREPROCESSING PYTHON SCRIPT
# ============================================================================

echo ""
echo "⚙️  Running preprocessing pipeline..."
python "$PROJECT_ROOT/scripts/preprocess.py" \
    --input_file "$DATA_RAW_DIR/merged_weather_data.csv" \
    --output_dir "$DATA_PROCESSED_DIR"

echo ""
echo "✅ Data preparation complete!"
echo "   Processed data saved to: $DATA_PROCESSED_DIR"
