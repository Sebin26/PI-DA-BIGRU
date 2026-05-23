#!/usr/bin/env python
"""
Training script.
Loads processed data and trains the DA-BiGRU model.
"""

import os
import sys
import argparse
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.seeding import seed_everything
from utils.constants import SEED, DEVICE
from data.loader import load_processed_data
from data.preprocessor import create_dataloaders
from models.da_bigru import DA_BiGRU
from training.trainer import Trainer
from training.config import TrainingConfig


def main(args):
    """Main training pipeline."""
    
    # Apply seeding
    seed_everything(SEED)
    
    print("\n" + "="*60)
    print("🚀 MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"🖥️  Using Device: {DEVICE}")
    
    # 1. Load processed data
    X_train, y_train, X_test, y_test, scaler, config = load_processed_data(args.data_dir)
    
    # 2. Create dataloaders
    print("\n🔄 Creating DataLoaders...")
    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test,
        batch_size=args.batch_size,
        shuffle_train=True
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
    
    print(f"✅ Model Initialized on {DEVICE}")
    print(f"   Input Dim: {input_dim}")
    
    # 4. Training configuration
    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=os.path.join(args.models_dir, 'checkpoints')
    )
    
    # 5. Train model
    trainer = Trainer(
        model=model,
        device=DEVICE,
        learning_rate=train_config.learning_rate
    )
    
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=train_config.epochs,
        patience=train_config.patience,
        checkpoint_dir=train_config.checkpoint_dir,
        log_interval=train_config.log_interval
    )
    
    # 6. Save training history
    import numpy as np
    history_path = os.path.join(args.results_dir, 'training_history.npz')
    np.savez(history_path, train_losses=train_losses, val_losses=val_losses)
    print(f"✅ Training history saved to {history_path}")
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with processed data')
    parser.add_argument('--models_dir', type=str, required=True,
                       help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(os.path.join(args.models_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    main(args)
