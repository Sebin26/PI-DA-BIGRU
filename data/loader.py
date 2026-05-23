"""
Data loading and preprocessing utilities.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import joblib


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load CSV data and create timestamp index.
    
    Args:
        data_path (str): Path to CSV file with columns YEAR, MO, DY, HR
        
    Returns:
        pd.DataFrame: DataFrame with timestamp index
    """
    print(f"\n📉 Loading Dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Create Timestamp Index (Critical for Hourly Data)
    df['timestamp'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].rename(
        columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}
    ))
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)  # Ensure time is sorted
    
    # Drop the breakdown columns now that we have index
    df.drop(columns=['YEAR', 'MO', 'DY', 'HR'], inplace=True)
    
    print(f"✅ Data Loaded. Shape: {df.shape}")
    print(f"   Date Range: {df.index.min()} to {df.index.max()}")
    print(f"   Total Hours: {len(df)}")
    
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        train_ratio (float): Ratio for train/test split. Default 0.8
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    print(f"\n🔪 Splitting Data ({int(train_ratio*100)}% Train / {int((1-train_ratio)*100)}% Test)...")
    train_size = int(len(df) * train_ratio)
    
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    print(f"   Train: {train_df.index[0]} to {train_df.index[-1]}")
    print(f"   Test:  {test_df.index[0]} to {test_df.index[-1]}")
    
    return train_df, test_df


def engineer_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Apply physics-based feature engineering.
    Rolling operations must be done AFTER split to prevent leakage.
    
    Args:
        df_input (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    df = df_input.copy()
    eps = 1e-6
    
    # A. Wind Shear (Logarithmic Wind Profile)
    df['WindShear'] = np.log((df['WS100M'] + eps) / (df['WS10M'] + eps)) / np.log(100 / 10)
    
    # B. Air Density (Ideal Gas Law: ρ = P/(R·T))
    T_kelvin = df['T2M'] + 273.15
    df['AirDensity'] = df['PS'] / (287.05 * T_kelvin)
    
    # C. Turbulence Intensity (Rolling window)
    roll = df['WS50M'].rolling(window=3, min_periods=1)
    df['TurbulenceIntensity'] = roll.std() / (roll.mean() + eps)
    df['TurbulenceIntensity'] = df['TurbulenceIntensity'].fillna(0)
    
    # D. Cyclical Time Encoding
    df['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['Month_Sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    return df


def scale_data(train_df: pd.DataFrame, test_df: pd.DataFrame, 
               feature_cols: list) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale train and test data using MinMaxScaler (fit on train only).
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        feature_cols (list): List of feature column names
        
    Returns:
        Tuple[np.ndarray, np.ndarray, MinMaxScaler]: (train_scaled, test_scaled, scaler)
    """
    print("\n⚖️  Scaling Data...")
    
    train_data = train_df[feature_cols].copy()
    test_data = test_df[feature_cols].copy()
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)   # ✅ Fit on train
    test_scaled = scaler.transform(test_data)         # ✅ Transform test
    
    print(f"✅ Data Processed.")
    print(f"   Train Shape: {train_scaled.shape}")
    print(f"   Test Shape:  {test_scaled.shape}")
    
    return train_scaled, test_scaled, scaler


def save_processed_data(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        scaler: MinMaxScaler, config: dict, save_dir: str) -> None:
    """
    Save processed data and configuration.
    
    Args:
        X_train, y_train, X_test, y_test (np.ndarray): Data arrays
        scaler (MinMaxScaler): Fitted scaler
        config (dict): Configuration dictionary
        save_dir (str): Directory to save files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n💾 Saving Processed Data to: {save_dir}")
    
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(config, os.path.join(save_dir, 'config.pkl'))
    
    print("✅ Data and Configuration Saved Successfully!")


def load_processed_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, 
                                                  np.ndarray, np.ndarray, 
                                                  MinMaxScaler, dict]:
    """
    Load preprocessed data and configuration.
    
    Args:
        data_dir (str): Directory containing processed data
        
    Returns:
        Tuple: (X_train, y_train, X_test, y_test, scaler, config)
    """
    print(f"\n📦 Loading Processed Data from: {data_dir}")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    config = joblib.load(os.path.join(data_dir, 'config.pkl'))
    
    print(f"✅ Data Loaded Successfully!")
    print(f"   X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"   X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler, config
