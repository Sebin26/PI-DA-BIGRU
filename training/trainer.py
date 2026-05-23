"""
Model training loop and utilities.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, List
import numpy as np


class Trainer:
    """Trainer class for DA-BiGRU model."""
    
    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): Model to train
            device (torch.device): Device to use
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(X_batch).squeeze()
            loss = self.criterion(pred, y_batch)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        
        return train_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                pred = self.model(X_batch).squeeze()
                val_loss += self.criterion(pred, y_batch).item()
        
        return val_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 50, patience: int = 10, 
              checkpoint_dir: str = None, log_interval: int = 5) -> Tuple[List[float], List[float]]:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs
            patience (int): Early stopping patience
            checkpoint_dir (str): Directory to save checkpoints
            log_interval (int): Logging interval
            
        Returns:
            Tuple[List[float], List[float]]: (train_losses, val_losses)
        """
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        start_time = time.time()
        
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        
        for epoch in range(epochs):
            avg_train = self.train_epoch(train_loader)
            avg_val = self.validate(val_loader)
            
            self.train_losses.append(avg_train)
            self.val_losses.append(avg_val)
            
            self.scheduler.step(avg_val)
            
            # Logging
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1:>3}/{epochs}] | Train: {avg_train:.5f} | Val: {avg_val:.5f} | Time: {elapsed:.1f}s")
            
            # Early stopping
            if avg_val < self.best_val_loss:
                self.best_val_loss = avg_val
                self.patience_counter = 0
                
                if checkpoint_dir:
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_loss': self.best_val_loss
                    }, checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n🛑 Early Stopping at Epoch {epoch+1}")
                    break
        
        print(f"✅ Training Finished. Best Val Loss: {self.best_val_loss:.5f}")
        return self.train_losses, self.val_losses
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {checkpoint_path}")
