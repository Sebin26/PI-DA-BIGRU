"""
DA-BiGRU model architecture with attention mechanisms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAttention(nn.Module):
    """
    Feature-level attention mechanism.
    Learns which features are important for prediction.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            Attention-weighted input
        """
        weights = self.attn(x)
        return x * weights


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism.
    Learns which time steps are important for prediction.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
        Returns:
            Context vector from temporal attention
        """
        weights = F.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1)


class DA_BiGRU(nn.Module):
    """
    Dual-Attention Bidirectional GRU for time series forecasting.
    
    Architecture:
    1. Feature Attention: Learn feature importance
    2. Bidirectional GRU: Capture temporal patterns
    3. Temporal Attention: Learn temporal importance
    4. Fully Connected Layer: Final prediction
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 n_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Hidden dimension size. Default 64
            n_layers (int): Number of GRU layers. Default 2
            dropout (float): Dropout rate. Default 0.2
        """
        super().__init__()
        self.feat_attn = FeatureAttention(input_dim)
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.temp_attn = TemporalAttention(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            Prediction of shape (batch, 1)
        """
        # Feature Attention
        x = self.feat_attn(x)
        
        # Bidirectional GRU
        out, _ = self.gru(x)
        
        # Temporal Attention
        ctx = self.temp_attn(out)
        
        # Fully Connected Layer
        return self.fc(ctx)
