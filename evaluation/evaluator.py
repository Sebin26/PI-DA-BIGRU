"""
Model evaluation and metrics calculation.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from typing import Tuple, Dict


class Evaluator:
    """Evaluator class for DA-BiGRU model."""
    
    def __init__(self, model: nn.Module, device: torch.device, scaler=None, 
                 target_col_idx: int = 0):
        """
        Initialize evaluator.
        
        Args:
            model (nn.Module): Model to evaluate
            device (torch.device): Device to use
            scaler: Fitted scaler for inverse transformation
            target_col_idx (int): Index of target column in scaled data
        """
        self.model = model
        self.device = device
        self.scaler = scaler
        self.target_col_idx = target_col_idx
        self.n_features = None
    
    def generate_predictions(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test data.
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (predictions, actuals)
        """
        self.model.eval()
        preds, actuals = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                p = self.model(X_batch).squeeze().cpu().numpy()
                preds.extend(p if isinstance(p, np.ndarray) and p.ndim > 0 else [p])
                actuals.extend(y_batch.numpy() if y_batch.ndim > 0 else [y_batch.item()])
        
        return np.array(preds), np.array(actuals)
    
    def inverse_transform_predictions(self, preds: np.ndarray, actuals: np.ndarray, 
                                      feature_cols: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse scale predictions back to original units.
        
        Args:
            preds (np.ndarray): Scaled predictions
            actuals (np.ndarray): Scaled actuals
            feature_cols (list): List of feature column names
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (inv_preds, inv_actuals)
        """
        if self.scaler is None:
            return preds, actuals
        
        self.n_features = len(feature_cols)
        n_samples = len(preds)
        
        # Create dummy arrays with all features
        dummy_pred = np.zeros((n_samples, self.n_features))
        dummy_act = np.zeros((n_samples, self.n_features))
        
        dummy_pred[:, self.target_col_idx] = preds
        dummy_act[:, self.target_col_idx] = actuals
        
        inv_pred = self.scaler.inverse_transform(dummy_pred)[:, self.target_col_idx]
        inv_act = self.scaler.inverse_transform(dummy_act)[:, self.target_col_idx]
        
        return inv_pred, inv_act
    
    def calculate_metrics(self, actuals: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            actuals (np.ndarray): Actual values
            preds (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        r2 = r2_score(actuals, preds)
        
        # MAPE for valid points
        valid_mask = actuals > 0.5
        if valid_mask.sum() > 0:
            mape = np.mean(np.abs((actuals[valid_mask] - preds[valid_mask]) / actuals[valid_mask])) * 100
        else:
            mape = 0.0
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        return metrics
    
    def calculate_error_by_wind_category(self, actuals: np.ndarray, 
                                        preds: np.ndarray) -> Dict[str, float]:
        """
        Calculate error metrics by wind speed category.
        
        Args:
            actuals (np.ndarray): Actual values
            preds (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: MAE for each wind speed category
        """
        residuals = np.abs(actuals - preds)
        
        categories = {
            'Calm (0-2)': (0, 2),
            'Light (2-5)': (2, 5),
            'Moderate (5-8)': (5, 8),
            'High (8+)': (8, 100)
        }
        
        error_by_cat = {}
        for cat_name, (min_ws, max_ws) in categories.items():
            mask = (actuals >= min_ws) & (actuals < max_ws)
            if mask.sum() > 0:
                error_by_cat[cat_name] = residuals[mask].mean()
            else:
                error_by_cat[cat_name] = 0.0
        
        return error_by_cat
    
    def print_metrics(self, metrics: Dict[str, float], 
                     error_by_cat: Dict[str, float] = None) -> None:
        """
        Print evaluation metrics.
        
        Args:
            metrics (Dict[str, float]): Metrics dictionary
            error_by_cat (Dict[str, float]): Error by wind category
        """
        print("\n" + "="*60)
        print("📊 EVALUATION METRICS")
        print("="*60)
        print(f"   MAPE:  {metrics['mape']:.2f}%")
        print(f"   MAE:   {metrics['mae']:.4f} m/s")
        print(f"   RMSE:  {metrics['rmse']:.4f} m/s")
        print(f"   R²:    {metrics['r2']:.4f}")
        
        if error_by_cat:
            print("\n📈 Performance by Wind Category:")
            for cat_name, mae_val in error_by_cat.items():
                print(f"   {cat_name:<15}: MAE = {mae_val:.4f} m/s")
