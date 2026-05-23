"""
Sequence creation and data preparation utilities.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(data: np.ndarray, seq_len: int, 
                    target_col_idx: int = 0) -> tuple:
    """
    Create sequences for time series prediction.
    
    Args:
        data (np.ndarray): Scaled data array
        seq_len (int): Sequence length (context window)
        target_col_idx (int): Index of target column
        
    Returns:
        tuple: (X, y) arrays of sequences
    """
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i+seq_len)]
        y = data[i+seq_len, target_col_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def create_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 64,
                       shuffle_train: bool = True) -> tuple:
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        X_train, y_train, X_test, y_test (np.ndarray): Data arrays
        batch_size (int): Batch size
        shuffle_train (bool): Whether to shuffle training data
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=shuffle_train
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=batch_size, shuffle=False
    )
    
    print(f"✅ DataLoaders Ready: Batch Size = {batch_size}")
    
    return train_loader, test_loader


def get_baseline_metrics(scaled_test_data: np.ndarray, target_col_idx: int) -> dict:
    """
    Calculate baseline metrics using persistence model.
    
    Args:
        scaled_test_data (np.ndarray): Scaled test data
        target_col_idx (int): Index of target column
        
    Returns:
        dict: Baseline metrics
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    y_actual = scaled_test_data[1:, target_col_idx]
    y_naive = scaled_test_data[:-1, target_col_idx]
    
    mse = mean_squared_error(y_actual, y_naive)
    r2 = r2_score(y_actual, y_naive)
    
    print(f"\n📊 Baseline (Persistence Model):")
    print(f"   MSE: {mse:.5f}")
    print(f"   R²:  {r2:.5f}")
    
    return {'mse': mse, 'r2': r2}
