"""
Global seeding for reproducibility.
"""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 2026) -> None:
    """
    Sets seeds for all random number generators to ensure reproducible results.
    
    Args:
        seed (int): Seed value for reproducibility. Default is 2026.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior on CuDNN (GPU)
    # Note: This guarantees reproducibility but might be slightly slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🔒 Global Seed set to: {seed}")
