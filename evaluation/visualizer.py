"""
Visualization utilities for model evaluation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional


def plot_diagnostic_dashboard(train_losses, val_losses, inv_preds, inv_actuals, 
                              timestamps=None, residuals=None, save_path: Optional[str] = None):
    """
    Generate 6-panel diagnostic dashboard.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        inv_preds (np.ndarray): Inverse-scaled predictions
        inv_actuals (np.ndarray): Inverse-scaled actuals
        timestamps (np.ndarray, optional): Timestamps for time-based analysis
        residuals (np.ndarray, optional): Residuals (actuals - preds)
        save_path (str, optional): Path to save figure
    """
    if residuals is None:
        residuals = inv_actuals - inv_preds
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # Plot 1: Time Series Zoom (First 100 hrs)
    ax = axes[0, 0]
    ax.plot(inv_actuals[:100], label='Actual', color='black', lw=2, alpha=0.7)
    ax.plot(inv_preds[:100], label='Predicted', color='#e74c3c', linestyle='--', lw=2)
    ax.set_title('1. Time Series Tracking (First 100 Hours)', fontweight='bold')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_xlabel('Time (Hours)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Scatter Plot
    ax = axes[0, 1]
    r2 = np.corrcoef(inv_actuals, inv_preds)[0, 1] ** 2
    ax.scatter(inv_actuals, inv_preds, alpha=0.1, s=10, color='#2980b9')
    max_val = max(inv_actuals.max(), inv_preds.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Fit')
    ax.set_title(f'2. Actual vs. Predicted (R²={r2:.3f})', fontweight='bold')
    ax.set_xlabel('Actual Wind Speed (m/s)')
    ax.set_ylabel('Predicted Wind Speed (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Residual Distribution
    ax = axes[1, 0]
    sns.histplot(residuals, kde=True, ax=ax, color='#8e44ad', bins=50)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title('3. Error (Residual) Distribution', fontweight='bold')
    ax.set_xlabel('Error (Actual - Predicted) [m/s]')
    
    # Plot 4: Error by Wind Speed
    ax = axes[1, 1]
    ax.scatter(inv_actuals, np.abs(residuals), alpha=0.2, s=10, color='#d35400')
    z = np.polyfit(inv_actuals, np.abs(residuals), 1)
    p = np.poly1d(z)
    ax.plot(inv_actuals, p(inv_actuals), "r--", lw=2, label='Trend')
    ax.set_title('4. Absolute Error vs. Wind Speed', fontweight='bold')
    ax.set_xlabel('Actual Wind Speed (m/s)')
    ax.set_ylabel('Absolute Error (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Diurnal Error Profile (Error by Hour)
    ax = axes[2, 0]
    if timestamps is not None:
        df_viz = pd.DataFrame({
            'hour': pd.to_datetime(timestamps).hour,
            'abs_error': np.abs(residuals)
        })
        hourly_error = df_viz.groupby('hour')['abs_error'].mean()
        ax.plot(hourly_error.index, hourly_error.values, marker='o', 
                linestyle='-', color='#16a085', lw=2)
        ax.set_title('5. Mean Absolute Error by Hour of Day', fontweight='bold')
        ax.set_xlabel('Hour of Day (0-23)')
        ax.set_ylabel('Mean Abs Error (m/s)')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Timestamps not available', ha='center', va='center')
        ax.set_title('5. Diurnal Error Profile', fontweight='bold')
    
    # Plot 6: Training History
    ax = axes[2, 1]
    ax.plot(train_losses, label='Train Loss', color='#2980b9')
    ax.plot(val_losses, label='Val Loss', color='#c0392b')
    ax.set_title('6. Training & Validation Loss', fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Dashboard saved to {save_path}")
    
    plt.show()


def plot_time_series(actuals, preds, start_idx=0, end_idx=100, 
                     title='Wind Speed Forecast', save_path: Optional[str] = None):
    """
    Plot time series comparison.
    
    Args:
        actuals (np.ndarray): Actual values
        preds (np.ndarray): Predicted values
        start_idx (int): Start index
        end_idx (int): End index
        title (str): Plot title
        save_path (str, optional): Path to save figure
    """
    plt.figure(figsize=(14, 6))
    plt.plot(actuals[start_idx:end_idx], label='Actual', marker='o', lw=2)
    plt.plot(preds[start_idx:end_idx], label='Predicted', linestyle='--', marker='s', lw=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time Step')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()
