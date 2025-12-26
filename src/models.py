"""
Model Training for SPX Option Pricing
======================================

Contains all model training functions:
- Neural Network with Two-Pass Training
- Random Forest
- XGBoost

All models use walk-forward validation (5 folds: 2021-2025)
"""

# ============================================================
# IMPORTS
# ============================================================

import sys
import os
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Import config
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import (
    PROJECT_ROOT, DATA_DIR, RESULTS_DIR, MODELS_DIR,
    FEATURES_BASIC
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_bs_baseline():
    """Load BS baseline from preprocessing results."""
    bs_path = os.path.join(RESULTS_DIR, 'bs_walk_forward_results.csv')
    if os.path.exists(bs_path):
        bs_df = pd.read_csv(bs_path)
        return bs_df['mae'].mean()
    return None


def check_and_clean_array(name, X):
    """Clean NaNs/Infs from arrays."""
    X = np.asarray(X, dtype=np.float32)
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  ⚠️ {name}: cleaning NaNs={n_nan} Infs={n_inf}")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def evaluate_model_metrics(y_true, y_pred, set_name="Test"):
    """Calculate evaluation metrics."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    return {
        'set': set_name,
        'mae': float(np.mean(abs_errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'median_ae': float(np.median(abs_errors)),
        'bias': float(np.mean(errors)),
        'max_error': float(np.max(abs_errors)),
    }


# ============================================================
# NEURAL NETWORK MODEL
# ============================================================

class OptionPricingMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
        super(OptionPricingMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


def train_neural_network(df, feature_columns):
    """
    Train Neural Network with two-pass training on walk-forward folds.
    
    Returns: results_df
    """
    
    print("\n" + "="*80)
    print("NEURAL NETWORK WITH TWO-PASS TRAINING (5 FOLDS)")
    print("="*80)
    
    fold_configs = [
        {'name': 'Fold 1', 'train_end': '2020-12-31', 'test_end': '2021-12-31'},
        {'name': 'Fold 2', 'train_end': '2021-12-31', 'test_end': '2022-12-31'},
        {'name': 'Fold 3', 'train_end': '2022-12-31', 'test_end': '2023-12-31'},
        {'name': 'Fold 4', 'train_end': '2023-12-31', 'test_end': '2024-12-31'},
        {'name': 'Fold 5', 'train_end': '2024-12-31', 'test_end': '2025-08-29'},
    ]
    
    all_fold_results = []
    
    for fold_idx, fold_config in enumerate(fold_configs):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx+1}/5: {fold_config['name']}")
        print(f"{'='*80}")
        
        train_mask = df['date'] <= fold_config['train_end']
        test_mask = (df['date'] > fold_config['train_end']) & (df['date'] <= fold_config['test_end'])
        
        df_train_val = df[train_mask].copy()
        df_test_fold = df[test_mask].copy()
        
        # Split into train (85%) and val (15%)
        n_total = len(df_train_val)
        n_train = int(0.85 * n_total)
        df_train_fold = df_train_val.iloc[:n_train].copy()
        df_val_fold = df_train_val.iloc[n_train:].copy()
        
        print(f"\nTrain: {len(df_train_fold):,} options")
        print(f"Val:   {len(df_val_fold):,} options")
        print(f"Test:  {len(df_test_fold):,} options")
        
        if len(df_test_fold) < 100:
            print(f"⚠️ Insufficient test data, skipping fold")
            continue
        
        # Pass A: Selection
        print(f"\n--- Pass A (Selection) Fold {fold_idx+1} ---")
        trained_sel = _train_nn_single_pass(
            df_train_fold, df_val_fold, df_test_fold, feature_columns,
            model_name=f"NN_Basic_Fold{fold_idx+1}_SEL",
            early_stopping=True
        )
        
        best_epoch = trained_sel["best_epoch"]
        print(f"🏁 Selected best_epoch={best_epoch}")
        
        # Pass B: Final retrain
        print(f"\n--- Pass B (Final Retrain) Fold {fold_idx+1} ---")
        trained_final = _train_nn_single_pass(
            df_train_val, None, df_test_fold, feature_columns,
            model_name=f"NN_Basic_Fold{fold_idx+1}_FINAL",
            fixed_epochs=best_epoch
        )
        
        results = _evaluate_nn(trained_final, df_test_fold)
        
        # Store results
        train_metrics = results["metrics"]["train"]
        test_metrics = results["metrics"]["test"]
        type_metrics = results["metrics"].get("type_metrics") or {}
        
        all_fold_results.append({
            "fold": fold_config["name"],
            "test_year": fold_config["test_end"][:4],
            "train_size": len(df_train_val),
            "test_size": len(df_test_fold),
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_bias": train_metrics["bias"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_bias": test_metrics["bias"],
            "mae_atm": results["atm_metrics"]["mae"] if results.get("atm_metrics") else np.nan,
            "mae_call": type_metrics.get("call", {}).get("mae", np.nan),
            "mae_put": type_metrics.get("put", {}).get("mae", np.nan),
        })
    
    # Summary
    results_df = pd.DataFrame(all_fold_results)
    avg_mae = results_df['test_mae'].mean()
    
    print("\n" + "="*80)
    print("📊 NEURAL NETWORK AVERAGE RESULTS")
    print("="*80)
    
    bs_baseline = load_bs_baseline()
    print(f"\n{'Model':<30} {'Avg Test MAE':<15} {'vs BS':<15}")
    print("-" * 60)
    print(f"{'BS (Historical Vol)':<30} ${bs_baseline:<14.2f} Baseline")
    print(f"{'NN (Two-Pass Training)':<30} ${avg_mae:<14.2f} {(bs_baseline-avg_mae)/bs_baseline*100:+.1f}%")
    
    results_df.to_csv(os.path.join(RESULTS_DIR, 'nn_walk_forward_results.csv'), index=False)
    print(f"\n✅ Results saved to: nn_walk_forward_results.csv")
    
    return results_df


def _train_nn_single_pass(df_train, df_val, df_test, feature_columns, model_name,
                          early_stopping=True, fixed_epochs=None):
    """Helper: Train NN for one pass."""
    
    X_train = df_train[feature_columns].to_numpy(dtype=np.float32)
    y_train = df_train['mid_price'].to_numpy(dtype=np.float32)
    
    X_val = df_val[feature_columns].to_numpy(dtype=np.float32) if df_val is not None else None
    y_val = df_val['mid_price'].to_numpy(dtype=np.float32) if df_val is not None else None
    
    X_test = df_test[feature_columns].to_numpy(dtype=np.float32)
    y_test = df_test['mid_price'].to_numpy(dtype=np.float32)
    
    # Clean
    X_train = check_and_clean_array("X_train", X_train)
    X_test = check_and_clean_array("X_test", X_test)
    y_train = check_and_clean_array("y_train", y_train)
    y_test = check_and_clean_array("y_test", y_test)
    
    if X_val is not None:
        X_val = check_and_clean_array("X_val", X_val)
        y_val = check_and_clean_array("y_val", y_val)
    
    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    if X_val is not None:
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    else:
        X_val_scaled, y_val_scaled = None, None
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp_model = OptionPricingMLP(input_size=len(feature_columns)).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(mlp_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).reshape(-1, 1).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    if X_val_scaled is not None:
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).reshape(-1, 1).to(device)
    else:
        X_val_tensor, y_val_tensor = None, None
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    
    # Training
    num_epochs = fixed_epochs if fixed_epochs else 300
    patience = 60
    best_val_loss = float('inf')
    best_epoch = None
    patience_counter = 0
    
    start_time = time.perf_counter()
    
    for epoch in range(num_epochs):
        mlp_model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = mlp_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        if X_val_tensor is None:
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f}")
            continue
        
        # Validation
        mlp_model.eval()
        with torch.no_grad():
            val_outputs = mlp_model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            model_path = os.path.join(MODELS_DIR, f'best_{model_name}.pth')
            torch.save(mlp_model.state_dict(), model_path)
        else:
            patience_counter += 1
            if early_stopping and patience_counter >= patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.perf_counter() - start_time
    print(f"\n⏱️ Training time: {total_time:.1f}s")
    
    # Load best model
    model_path = os.path.join(MODELS_DIR, f'best_{model_name}.pth')
    if X_val_tensor is not None and os.path.exists(model_path):
        mlp_model.load_state_dict(torch.load(model_path, map_location=device))
    
    if X_val_tensor is None:
        best_epoch = num_epochs
    
    return {
        'model': mlp_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'best_epoch': best_epoch,
        'X_train_tensor': X_train_tensor,
        'X_val_tensor': X_val_tensor,
        'X_test_tensor': X_test_tensor,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'device': device,
    }


def _evaluate_nn(trained_dict, df_test_fold):
    """Helper: Evaluate trained NN."""
    
    model = trained_dict['model']
    scaler_y = trained_dict['scaler_y']
    
    X_train_tensor = trained_dict['X_train_tensor']
    X_val_tensor = trained_dict.get('X_val_tensor')
    X_test_tensor = trained_dict['X_test_tensor']
    
    y_train = trained_dict['y_train']
    y_val = trained_dict.get('y_val')
    y_test = trained_dict['y_test']
    
    model.eval()
    with torch.no_grad():
        y_train_scaled = model(X_train_tensor).cpu().numpy().flatten()
        y_train_pred = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
        
        if X_val_tensor is not None:
            y_val_scaled = model(X_val_tensor).cpu().numpy().flatten()
            y_val_pred = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
        else:
            y_val_pred = None
        
        y_test_scaled = model(X_test_tensor).cpu().numpy().flatten()
        y_test_pred = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    train_metrics = evaluate_model_metrics(y_train, y_train_pred, "Train")
    val_metrics = evaluate_model_metrics(y_val, y_val_pred, "Val") if y_val is not None else None
    test_metrics = evaluate_model_metrics(y_test, y_test_pred, "Test")
    
    # Type metrics
    type_metrics = {}
    if df_test_fold is not None and 'cp_flag' in df_test_fold.columns:
        is_call = (df_test_fold['cp_flag'].values == 'C')
        is_put = (df_test_fold['cp_flag'].values == 'P')
        
        if is_call.sum() > 0:
            type_metrics['call'] = evaluate_model_metrics(y_test[is_call], y_test_pred[is_call], "Call")
        if is_put.sum() > 0:
            type_metrics['put'] = evaluate_model_metrics(y_test[is_put], y_test_pred[is_put], "Put")
    
    # ATM metrics
    atm_metrics = None
    if df_test_fold is not None and 'moneyness' in df_test_fold.columns:
        atm_mask = (df_test_fold['moneyness'] >= 0.95) & (df_test_fold['moneyness'] <= 1.05)
        if atm_mask.sum() > 0:
            atm_metrics = evaluate_model_metrics(y_test[atm_mask.values], y_test_pred[atm_mask.values], "ATM")
    
    print(f"\n{'Set':<12} {'MAE':<12} {'RMSE':<12} {'Bias':<12}")
    print("-" * 50)
    print(f"{'Train':<12} ${train_metrics['mae']:<11.2f} ${train_metrics['rmse']:<11.2f} ${train_metrics['bias']:<11.2f}")
    if val_metrics:
        print(f"{'Validation':<12} ${val_metrics['mae']:<11.2f} ${val_metrics['rmse']:<11.2f} ${val_metrics['bias']:<11.2f}")
    print(f"{'Test':<12} ${test_metrics['mae']:<11.2f} ${test_metrics['rmse']:<11.2f} ${test_metrics['bias']:<11.2f}")
    
    return {
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'type_metrics': type_metrics,
        },
        'atm_metrics': atm_metrics,
    }


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

def train_random_forest(df, feature_columns):
    """
    Train Random Forest on walk-forward folds.
    
    Returns: results_df, feature_importance
    """
    
    print("\n" + "="*80)
    print("RANDOM FOREST TRAINING (5 FOLDS)")
    print("="*80)
    
    folds = [
        {'name': 'Fold 1', 'train_end': '2020-12-31', 'test_end': '2021-12-31', 'test_year': '2021'},
        {'name': 'Fold 2', 'train_end': '2021-12-31', 'test_end': '2022-12-31', 'test_year': '2022'},
        {'name': 'Fold 3', 'train_end': '2022-12-31', 'test_end': '2023-12-31', 'test_year': '2023'},
        {'name': 'Fold 4', 'train_end': '2023-12-31', 'test_end': '2024-12-31', 'test_year': '2024'},
        {'name': 'Fold 5', 'train_end': '2024-12-31', 'test_end': '2025-08-29', 'test_year': '2025'},
    ]
    
    results = []
    all_feature_importance = []
    
    for i, fold in enumerate(folds):
        print(f"\n{'='*80}")
        print(f"FOLD {i+1}/5: {fold['name']}")
        print(f"{'='*80}")
        
        train_mask = df['date'] <= fold['train_end']
        test_mask = (df['date'] > fold['train_end']) & (df['date'] <= fold['test_end'])
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        print(f"\nTrain: {len(df_train):,} options")
        print(f"Test:  {len(df_test):,} options")
        
        # Prepare data
        X_train = df_train[feature_columns].to_numpy(dtype=np.float32)
        y_train = df_train['mid_price'].to_numpy(dtype=np.float32)
        X_test = df_test[feature_columns].to_numpy(dtype=np.float32)
        y_test = df_test['mid_price'].to_numpy(dtype=np.float32)
        
        X_train = check_and_clean_array("X_train", X_train)
        X_test = check_and_clean_array("X_test", X_test)
        y_train = check_and_clean_array("y_train", y_train)
        y_test = check_and_clean_array("y_test", y_test)
        
        # Train
        print(f"\n--- Training Random Forest ({fold['name']}) ---")
        start_time = time.perf_counter()
        
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        rf_model.fit(X_train, y_train)
        
        train_time = time.perf_counter() - start_time
        print(f"✅ Training completed in {train_time:.1f}s")
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"RF_Basic_{fold['name'].replace(' ', '')}.joblib")
        joblib.dump(rf_model, model_path)
        
        # Evaluate
        train_pred = rf_model.predict(X_train)
        test_pred = rf_model.predict(X_test)
        
        train_metrics = evaluate_model_metrics(y_train, train_pred, "Train")
        test_metrics = evaluate_model_metrics(y_test, test_pred, "Test")
        
        # Type metrics
        type_metrics = {}
        if 'cp_flag' in df_test.columns:
            is_call = (df_test['cp_flag'].values == 'C')
            is_put = (df_test['cp_flag'].values == 'P')
            
            if is_call.sum() > 0:
                type_metrics['call'] = evaluate_model_metrics(y_test[is_call], test_pred[is_call], "Call")
            if is_put.sum() > 0:
                type_metrics['put'] = evaluate_model_metrics(y_test[is_put], test_pred[is_put], "Put")
        
        # ATM
        atm_mask = df_test['money_bucket'] == 'ATM' if 'money_bucket' in df_test.columns else None
        mae_atm = mean_absolute_error(df_test.loc[atm_mask, 'mid_price'], test_pred[atm_mask.values]) if atm_mask is not None and atm_mask.sum() > 0 else np.nan
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance['fold'] = fold['name']
        all_feature_importance.append(importance)
        
        print(f"\n{'Set':<12} {'MAE':<12} {'RMSE':<12} {'Bias':<12}")
        print("-" * 50)
        print(f"{'Train':<12} ${train_metrics['mae']:<11.2f} ${train_metrics['rmse']:<11.2f} ${train_metrics['bias']:<11.2f}")
        print(f"{'Test':<12} ${test_metrics['mae']:<11.2f} ${test_metrics['rmse']:<11.2f} ${test_metrics['bias']:<11.2f}")
        
        results.append({
            'fold': fold['name'],
            'test_year': fold['test_year'],
            'train_size': len(df_train),
            'test_size': len(df_test),
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'train_bias': train_metrics['bias'],
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_bias': test_metrics['bias'],
            'mae_atm': mae_atm,
            'mae_call': type_metrics.get('call', {}).get('mae', np.nan),
            'mae_put': type_metrics.get('put', {}).get('mae', np.nan),
        })
    
    # Summary
    results_df = pd.DataFrame(results)
    avg_mae = results_df['test_mae'].mean()
    
    print("\n" + "="*80)
    print("📊 RANDOM FOREST AVERAGE RESULTS")
    print("="*80)
    
    bs_baseline = load_bs_baseline()
    print(f"\n{'Model':<30} {'Avg Test MAE':<15} {'vs BS':<15}")
    print("-" * 60)
    print(f"{'BS (Historical Vol)':<30} ${bs_baseline:<14.2f} Baseline")
    print(f"{'Random Forest':<30} ${avg_mae:<14.2f} {(bs_baseline-avg_mae)/bs_baseline*100:+.1f}%")
    
    results_df.to_csv(os.path.join(RESULTS_DIR, 'rf_walk_forward_results.csv'), index=False)
    
    # Feature importance
    all_importance_df = pd.concat(all_feature_importance)
    avg_importance = all_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    avg_importance.to_frame().to_csv(os.path.join(RESULTS_DIR, 'rf_feature_importance.csv'))
    
    print(f"\n✅ Results saved")
    
    return results_df, avg_importance


# ============================================================
# XGBOOST MODEL
# ============================================================

def train_xgboost(df, feature_columns):
    """
    Train XGBoost on walk-forward folds.
    
    Returns: results_df, feature_importance
    """
    
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost not available")
        return None, None
    
    print("\n" + "="*80)
    print("XGBOOST TRAINING (5 FOLDS)")
    print("="*80)
    
    folds = [
        {'name': 'Fold 1', 'train_end': '2020-12-31', 'test_end': '2021-12-31', 'test_year': '2021'},
        {'name': 'Fold 2', 'train_end': '2021-12-31', 'test_end': '2022-12-31', 'test_year': '2022'},
        {'name': 'Fold 3', 'train_end': '2022-12-31', 'test_end': '2023-12-31', 'test_year': '2023'},
        {'name': 'Fold 4', 'train_end': '2023-12-31', 'test_end': '2024-12-31', 'test_year': '2024'},
        {'name': 'Fold 5', 'train_end': '2024-12-31', 'test_end': '2025-08-29', 'test_year': '2025'},
    ]
    
    results = []
    all_feature_importance = []
    
    for i, fold in enumerate(folds):
        print(f"\n{'='*80}")
        print(f"FOLD {i+1}/5: {fold['name']}")
        print(f"{'='*80}")
        
        train_mask = df['date'] <= fold['train_end']
        test_mask = (df['date'] > fold['train_end']) & (df['date'] <= fold['test_end'])
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        print(f"\nTrain: {len(df_train):,} options")
        print(f"Test:  {len(df_test):,} options")
        
        # Prepare data
        X_train = df_train[feature_columns].to_numpy(dtype=np.float32)
        y_train = df_train['mid_price'].to_numpy(dtype=np.float32)
        X_test = df_test[feature_columns].to_numpy(dtype=np.float32)
        y_test = df_test['mid_price'].to_numpy(dtype=np.float32)
        
        X_train = check_and_clean_array("X_train", X_train)
        X_test = check_and_clean_array("X_test", X_test)
        y_train = check_and_clean_array("y_train", y_train)
        y_test = check_and_clean_array("y_test", y_test)
        
        # Train
        print(f"\n--- Training XGBoost ({fold['name']}) ---")
        start_time = time.perf_counter()
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=1,
            reg_alpha=1.0,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            objective='reg:squarederror',
            tree_method='hist',
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        
        train_time = time.perf_counter() - start_time
        print(f"✅ Training completed in {train_time:.1f}s")
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"XGB_Basic_{fold['name'].replace(' ', '')}.joblib")
        joblib.dump(xgb_model, model_path)
        
        # Evaluate
        train_pred = xgb_model.predict(X_train)
        test_pred = xgb_model.predict(X_test)
        
        train_metrics = evaluate_model_metrics(y_train, train_pred, "Train")
        test_metrics = evaluate_model_metrics(y_test, test_pred, "Test")
        
        # Type metrics
        type_metrics = {}
        if 'cp_flag' in df_test.columns:
            is_call = (df_test['cp_flag'].values == 'C')
            is_put = (df_test['cp_flag'].values == 'P')
            
            if is_call.sum() > 0:
                type_metrics['call'] = evaluate_model_metrics(y_test[is_call], test_pred[is_call], "Call")
            if is_put.sum() > 0:
                type_metrics['put'] = evaluate_model_metrics(y_test[is_put], test_pred[is_put], "Put")
        
        # ATM
        atm_mask = df_test['money_bucket'] == 'ATM' if 'money_bucket' in df_test.columns else None
        mae_atm = mean_absolute_error(df_test.loc[atm_mask, 'mid_price'], test_pred[atm_mask.values]) if atm_mask is not None and atm_mask.sum() > 0 else np.nan
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance['fold'] = fold['name']
        all_feature_importance.append(importance)
        
        print(f"\n{'Set':<12} {'MAE':<12} {'RMSE':<12} {'Bias':<12}")
        print("-" * 50)
        print(f"{'Train':<12} ${train_metrics['mae']:<11.2f} ${train_metrics['rmse']:<11.2f} ${train_metrics['bias']:<11.2f}")
        print(f"{'Test':<12} ${test_metrics['mae']:<11.2f} ${test_metrics['rmse']:<11.2f} ${test_metrics['bias']:<11.2f}")
        
        results.append({
            'fold': fold['name'],
            'test_year': fold['test_year'],
            'train_size': len(df_train),
            'test_size': len(df_test),
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'train_bias': train_metrics['bias'],
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_bias': test_metrics['bias'],
            'mae_atm': mae_atm,
            'mae_call': type_metrics.get('call', {}).get('mae', np.nan),
            'mae_put': type_metrics.get('put', {}).get('mae', np.nan),
        })
    
    # Summary
    results_df = pd.DataFrame(results)
    avg_mae = results_df['test_mae'].mean()
    
    print("\n" + "="*80)
    print("📊 XGBOOST AVERAGE RESULTS")
    print("="*80)
    
    bs_baseline = load_bs_baseline()
    print(f"\n{'Model':<30} {'Avg Test MAE':<15} {'vs BS':<15}")
    print("-" * 60)
    print(f"{'BS (Historical Vol)':<30} ${bs_baseline:<14.2f} Baseline")
    print(f"{'XGBoost':<30} ${avg_mae:<14.2f} {(bs_baseline-avg_mae)/bs_baseline*100:+.1f}%")
    
    results_df.to_csv(os.path.join(RESULTS_DIR, 'xgb_walk_forward_results.csv'), index=False)
    
    # Feature importance
    all_importance_df = pd.concat(all_feature_importance)
    avg_importance = all_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    avg_importance.to_frame().to_csv(os.path.join(RESULTS_DIR, 'xgb_feature_importance.csv'))
    
    print(f"\n✅ Results saved")
    
    return results_df, avg_importance
