#!/usr/bin/env python
"""
Data preprocessing script.
Loads raw data, applies feature engineering, and saves processed data.
"""

import os
import sys
import argparse
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.seeding import seed_everything
from utils.constants import SEED, TRAIN_TEST_SPLIT, FEATURE_COLS, TARGET_COL_NAME
from data.loader import (
    load_data, split_data, engineer_features, scale_data, save_processed_data
)
from data.preprocessor import create_sequences, get_baseline_metrics


def main(args):
    """Main preprocessing pipeline."""
    
    # Apply seeding
    seed_everything(SEED)
    
    print("\n" + "="*60)
    print("⚙️  DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Load data
    df = load_data(args.input_file)
    
    # 2. Split data
    train_df, test_df = split_data(df, train_ratio=TRAIN_TEST_SPLIT)
    
    # 3. Feature engineering
    print("\n⚙️  Engineering Features...")
    train_df_features = engineer_features(train_df)
    test_df_features = engineer_features(test_df)
    
    # 4. Select features and scale
    train_df_final = train_df_features[FEATURE_COLS].copy()
    test_df_final = test_df_features[FEATURE_COLS].copy()
    
    train_scaled, test_scaled, scaler = scale_data(
        train_df_final, test_df_final, FEATURE_COLS
    )
    
    # 5. Baseline metrics
    target_col_idx = FEATURE_COLS.index(TARGET_COL_NAME)
    baseline_metrics = get_baseline_metrics(test_scaled, target_col_idx)
    
    # 6. Create sequences
    seq_len = 24  # from constants
    print("\n🔄 Creating Sequences...")
    X_train, y_train = create_sequences(train_scaled, seq_len, target_col_idx)
    X_test, y_test = create_sequences(test_scaled, seq_len, target_col_idx)
    
    # 7. Configuration
    config = {
        'seq_len': seq_len,
        'batch_size': 64,
        'feature_cols': FEATURE_COLS,
        'target_col_name': TARGET_COL_NAME,
        'target_col_idx': target_col_idx,
        'train_size': len(train_df),
        'baseline_metrics': baseline_metrics
    }
    
    # 8. Save processed data
    save_processed_data(X_train, y_train, X_test, y_test, scaler, config, args.output_dir)
    
    print("\n✅ Data preprocessing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing pipeline')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed data')
    
    args = parser.parse_args()
    main(args)
