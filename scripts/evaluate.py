#!/usr/bin/env python
"""
Evaluation script.
Loads trained model and generates evaluation metrics and visualizations.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.constants import DEVICE
from data.loader import load_processed_data
from data.preprocessor import create_dataloaders
from models.da_bigru import DA_BiGRU
from evaluation.evaluator import Evaluator
from evaluation.visualizer import plot_diagnostic_dashboard


def main(args):
    """Main evaluation pipeline."""
    
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION PIPELINE")
    print("="*60)
    
    # 1. Load processed data
    X_train, y_train, X_test, y_test, scaler, config = load_processed_data(args.data_dir)
    
    # 2. Create dataloaders
    print("\n🔄 Creating DataLoaders...")
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test,
        batch_size=config['batch_size'],
        shuffle_train=False  # No shuffling for evaluation
    )
    
    # 3. Initialize model
    print("\n🏗️  Building DA-BiGRU Model...")
    input_dim = X_train.shape[2]
    model = DA_BiGRU(
        input_dim=input_dim,
        hidden_dim=64,
        n_layers=2,
        dropout=0.2
    ).to(DEVICE)
    
    # 4. Load best checkpoint
    checkpoint_path = os.path.join(args.models_dir, 'checkpoints', 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from {checkpoint_path}")
    
    # 5. Generate predictions
    print("\n🔮 Generating Predictions...")
    evaluator = Evaluator(
        model=model,
        device=DEVICE,
        scaler=scaler,
        target_col_idx=config['target_col_idx']
    )
    
    preds, actuals = evaluator.generate_predictions(test_loader)
    inv_preds, inv_actuals = evaluator.inverse_transform_predictions(
        preds, actuals, config['feature_cols']
    )
    
    # 6. Calculate metrics
    print("\n📈 Calculating Metrics...")
    metrics = evaluator.calculate_metrics(inv_actuals, inv_preds)
    error_by_cat = evaluator.calculate_error_by_wind_category(inv_actuals, inv_preds)
    
    evaluator.print_metrics(metrics, error_by_cat)
    
    # 7. Save metrics to file
    metrics_file = os.path.join(args.results_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("DA-BiGRU EVALUATION METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"MAPE:  {metrics['mape']:.2f}%\n")
        f.write(f"MAE:   {metrics['mae']:.4f} m/s\n")
        f.write(f"RMSE:  {metrics['rmse']:.4f} m/s\n")
        f.write(f"R²:    {metrics['r2']:.4f}\n\n")
        f.write("Error by Wind Category:\n")
        for cat_name, mae_val in error_by_cat.items():
            f.write(f"   {cat_name:<15}: MAE = {mae_val:.4f} m/s\n")
    
    print(f"✅ Metrics saved to {metrics_file}")
    
    # 8. Load training history
    history_file = os.path.join(args.results_dir, 'training_history.npz')
    history = np.load(history_file)
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    # 9. Generate visualizations
    print("\n📊 Generating Diagnostic Dashboard...")
    residuals = inv_actuals - inv_preds
    dashboard_path = os.path.join(args.results_dir, 'diagnostic_dashboard.png')
    
    plot_diagnostic_dashboard(
        train_losses=train_losses,
        val_losses=val_losses,
        inv_preds=inv_preds,
        inv_actuals=inv_actuals,
        residuals=residuals,
        save_path=dashboard_path
    )
    
    # 10. Save predictions to CSV
    pred_df = pd.DataFrame({
        'actual': inv_actuals,
        'predicted': inv_preds,
        'error': residuals,
        'abs_error': np.abs(residuals)
    })
    pred_csv = os.path.join(args.results_dir, 'predictions.csv')
    pred_df.to_csv(pred_csv, index=False)
    print(f"✅ Predictions saved to {pred_csv}")
    
    print("\n✅ Evaluation complete!")
    print(f"\n📁 Results saved to: {args.results_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model evaluation pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with processed data')
    parser.add_argument('--models_dir', type=str, required=True,
                       help='Directory with model checkpoints')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    main(args)
