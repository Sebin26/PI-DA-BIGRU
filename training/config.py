"""
Training configuration and utilities.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 10
    seed: int = 2026
    checkpoint_dir: str = "models/checkpoints"
    log_interval: int = 5
    weight_decay: float = 1e-5
    
    def to_dict(self):
        """Convert to dictionary."""
        return self.__dict__
