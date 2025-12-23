"""
XGBoost for SPX Option Pricing
===============================

Walk-forward validation (5 folds: 2021-2025)
Uses FEATURES_BASIC from config.py (created in preprocessing)

NOTE: This file does NOT create features - they come from data_preprocessing.py
"""

import os
import sys
import time
import csv
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not installed. Run: pip install xgboost")


# ============================================================
# IMPORTS FROM CONFIG
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import (
        PROJECT_ROOT, DATA_DIR, RESULTS_DIR, MODELS_DIR,
        FEATURES_BASIC
    )
    print("✅ Configuration loaded from config.py")
except ImportError:
    print("⚠️ Could not import from config.py, using defaults")
    PROJECT_ROOT = project_root
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    
    FEATURES_BASIC = [
        'moneyness', 'log_moneyness', 'T', 'log_T', 'sqrt_T',
        'is_call', 'forward_price_norm', 'moneyness_T',
        'log_moneyness_sqrt_T', 'log_volume', 'log_open_interest',
        'bid_ask_spread', 'historical_vol', 'historical_vol_sqrt_T',
    ]

for d in [RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

def load_bs_baseline():
    """Load BS baseline from preprocessing results."""
    bs_path = os.path.join(RESULTS_DIR, 'bs_walk_forward_results.csv')
    if os.path.exists(bs_path):
        bs_df = pd.read_csv(bs_path)
        return bs_df['mae'].mean()
    return None

# ============================================================
# XGBOOST MODEL
# ============================================================

def train_xgboost(X_train, y_train, X_val=None, y_val=None,
                  n_estimators=1000, max_depth=5, learning_rate=0.02,
                  min_child_weight=200, reg_alpha=1.0, reg_lambda=10.0, gamma=1.0, 
                  subsample=0.6, colsample_bytree=0.6, n_jobs=-1, random_state=42):
    """Train XGBoost model with regularization."""
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_jobs=n_jobs,
        random_state=random_state,
        objective='reg:squarederror',
        tree_method='hist',
        verbosity=0
    )
    
    if X_val is not None and y_val is not None:
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        xgb_model.fit(X_train, y_train)
    
    return xgb_model


def get_feature_importance(xgb_model, feature_names):
    """Extract feature importance from trained model."""
    return pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)


def evaluate_model(y_true, y_pred, set_name="Test"):
    """Calculate evaluation metrics. Bias = mean error."""
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

def check_and_clean_array(name, X):
    """Defensive cleaning for NaNs/Infs."""
    X = np.asarray(X)
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  ⚠️ {name}: cleaning NaNs={n_nan} Infs={n_inf}")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def run_walk_forward_validation(df, feature_columns, target_col='mid_price'):
    """Run 5-fold walk-forward validation."""
    
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost not available")
        return None, None
    
    print("\n" + "="*80)
    print("WALK-FORWARD TRAINING (5 FOLDS) - XGBOOST")
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
        
        # Get all data up to train_end for train+val
        train_val_mask = df['date'] <= fold['train_end']
        test_mask = (df['date'] > fold['train_end']) & (df['date'] <= fold['test_end'])
        
        df_train_val = df[train_val_mask].copy()
        df_test = df[test_mask].copy()
        
        # Use 100% of pre-test data for training (no validation split)
        df_train = df_train_val
        df_val = None

        print(f"\nTrain: {len(df_train):,} options (100% up to {fold['train_end']})")
        print(f"Val:   0 options (no validation split)")
        print(f"Test:  {len(df_test):,} options ({fold['test_year']})")

        # Prepare data
        X_train = df_train[feature_columns].to_numpy(dtype=np.float32)
        y_train = df_train[target_col].to_numpy(dtype=np.float32)

        X_val, y_val = None, None


        X_test = df_test[feature_columns].to_numpy(dtype=np.float32)
        y_test = df_test[target_col].to_numpy(dtype=np.float32)

        X_train = check_and_clean_array("X_train", X_train)
        y_train = check_and_clean_array("y_train", y_train)
        X_test  = check_and_clean_array("X_test", X_test)
        y_test  = check_and_clean_array("y_test", y_test)

        # Train model
        print(f"\n--- Training XGBoost ({fold['name']}) ---")
        start_time = time.perf_counter()
        xgb_model = train_xgboost(X_train, y_train)
        train_time = time.perf_counter() - start_time
        print(f"✅ Training completed in {train_time:.1f}s")
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"XGB_Basic_{fold['name'].replace(' ', '')}.joblib")
        joblib.dump(xgb_model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save training time
        runtime_path = os.path.join(RESULTS_DIR, "training_times.csv")
        header = ["model_name", "n_train", "n_val", "n_test", "seconds"]
        row = [f"XGB_Basic_{fold['name'].replace(' ', '')}", len(df_train), 0, len(df_test), train_time]
        
        file_exists = os.path.exists(runtime_path)
        with open(runtime_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        
        # Evaluate
        train_pred = xgb_model.predict(X_train)
        test_pred = xgb_model.predict(X_test)
        
        train_metrics = evaluate_model(y_train, train_pred, "Train")
        test_metrics = evaluate_model(y_test, test_pred, "Test")

        # Call vs Put split (Test)
        type_metrics = {}
        if 'cp_flag' in df_test.columns:
            is_call = (df_test['cp_flag'].values == 'C')
            is_put  = (df_test['cp_flag'].values == 'P')

            if is_call.sum() > 0:
                type_metrics['call'] = evaluate_model(y_test[is_call], test_pred[is_call], "Test_Call")
                print(f"\n📌 Call Test MAE: ${type_metrics['call']['mae']:.2f} | Bias: ${type_metrics['call']['bias']:.2f} (n={is_call.sum():,})")

            if is_put.sum() > 0:
                type_metrics['put'] = evaluate_model(y_test[is_put], test_pred[is_put], "Test_Put")
                print(f"📌 Put  Test MAE: ${type_metrics['put']['mae']:.2f} | Bias: ${type_metrics['put']['bias']:.2f} (n={is_put.sum():,})")

        
        print(f"\n{'Set':<12} {'MAE':<12} {'RMSE':<12} {'Median AE':<12} {'Bias':<12}")
        print("-" * 70)
        print(f"{'Train':<12} ${train_metrics['mae']:<11.2f} ${train_metrics['rmse']:<11.2f} "
            f"${train_metrics['median_ae']:<11.2f} ${train_metrics['bias']:<11.2f}")
        print(f"{'Test':<12} ${test_metrics['mae']:<11.2f} ${test_metrics['rmse']:<11.2f} "
            f"${test_metrics['median_ae']:<11.2f} ${test_metrics['bias']:<11.2f}")


        # Feature importance
        importance = get_feature_importance(xgb_model, feature_columns)
        importance['fold'] = fold['name']
        all_feature_importance.append(importance)
        
        # Performance by moneyness
        if 'money_bucket' in df_test.columns:
            print(f"\n📊 Performance by Moneyness:")
            df_test_eval = df_test.copy()
            df_test_eval['xgb_pred'] = test_pred
            for bucket in ['Low M', 'ATM', 'High M']:
                bucket_mask = df_test_eval['money_bucket'] == bucket
                if bucket_mask.sum() > 0:
                    bucket_mae = mean_absolute_error(
                        df_test_eval.loc[bucket_mask, target_col],
                        df_test_eval.loc[bucket_mask, 'xgb_pred']
                    )
                    print(f"   {bucket}: MAE=${bucket_mae:.2f} (n={bucket_mask.sum():,})")
        
        atm_mask = (df_test['money_bucket'] == 'ATM') if 'money_bucket' in df_test.columns else None
        if atm_mask is not None and atm_mask.sum() > 0:
            mae_atm = mean_absolute_error(df_test.loc[atm_mask, target_col].values, test_pred[atm_mask.values])
        else:
            mae_atm = np.nan

        val_size = 0 if df_val is None else len(df_val)
        
        results.append({
            'fold': fold['name'],
            'test_year': fold['test_year'],
            'train_size': len(df_train),
            'val_size': val_size,
            'test_size': len(df_test),

            # Train metrics
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'train_median_ae': train_metrics['median_ae'],
            'train_bias': train_metrics['bias'],

            # Test metrics
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_median_ae': test_metrics['median_ae'],
            'test_bias': test_metrics['bias'],

            # ATM + Type splits
            'mae_atm': mae_atm,
            'mae_call': type_metrics['call']['mae'] if 'call' in type_metrics else np.nan,
            'bias_call': type_metrics['call']['bias'] if 'call' in type_metrics else np.nan,
            'mae_put': type_metrics['put']['mae'] if 'put' in type_metrics else np.nan,
            'bias_put': type_metrics['put']['bias'] if 'put' in type_metrics else np.nan,
        })


    
    # Summary
    results_df = pd.DataFrame(results)
    avg_test_mae = results_df['test_mae'].mean()
    
    print("\n" + "="*80)
    print("📊 WALK-FORWARD AVERAGE RESULTS (5 FOLDS) - XGBOOST")
    print("="*80)
    
    bs_baseline = load_bs_baseline()

    print(f"\n{'Model':<30} {'Avg Test MAE':<15} {'vs BS':<15}")
    print("-" * 60)
    print(f"{'BS (Historical Vol)':<30} ${bs_baseline:<14.2f} Baseline")
    print(f"{'XGBoost (Basic + Hist Vol)':<30} ${avg_test_mae:<14.2f} {(bs_baseline-avg_test_mae)/bs_baseline*100:+.1f}%")
    
    print(f"\n📊 Fold-by-Fold Results:")
    print(f"\n{'Fold':<12} {'Test Year':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 48)
    for _, row in results_df.iterrows():
        print(f"{row['fold']:<12} {row['test_year']:<12} ${row['test_mae']:<11.2f} ${row['test_rmse']:<11.2f}")
    print("-" * 48)
    print(f"{'Average':<12} {'All':<12} ${avg_test_mae:<11.2f}")
    
    # Save results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'xgb_walk_forward_results.csv'), index=False)
    print(f"\n✅ Results saved to: xgb_walk_forward_results.csv")
    
    # Feature importance summary
    print("\n" + "="*60)
    print("📊 FEATURE IMPORTANCE (Average across folds)")
    print("="*60)
    
    all_importance_df = pd.concat(all_feature_importance)
    avg_importance = all_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    print(f"\n{'Feature':<30} {'Avg Importance':<15}")
    print("-" * 45)
    for feat, imp in avg_importance.items():
        print(f"{feat:<30} {imp:.4f}")
    
    avg_importance.to_frame().to_csv(os.path.join(RESULTS_DIR, 'xgb_feature_importance.csv'))
    
    return results_df, avg_importance


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_training():
    """Main entry point."""
    
    print("="*60)
    print("XGBOOST FOR SPX OPTIONS")
    print("Using WRDS OptionMetrics Data")
    print("="*60)
    
    if not XGBOOST_AVAILABLE:
        print("\n❌ Install XGBoost: pip install xgboost")
        return None
    
    # Load data
    print("\n📂 Loading data...")
    data_file = os.path.join(RESULTS_DIR, 'SPX_with_BS_Historical.csv')
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("   Run data_preprocessing.py first!")
        return None
    
    df = pd.read_csv(data_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    print(f"✅ Loaded: {len(df):,} options")
    
    # Verify features exist (created in preprocessing)
    print("\n📊 Verifying features...")
    missing_features = [f for f in FEATURES_BASIC if f not in df.columns]
    
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        print("   Run data_preprocessing.py first!")
        return None
    
    print(f"✅ All {len(FEATURES_BASIC)} features available")
    
    # Data already filtered in preprocessing - just remove NaNs
    df_clean = df.dropna(subset=FEATURES_BASIC + ['mid_price'])
    print(f"After removing NaNs: {len(df_clean):,} options")
    
    # Run walk-forward validation
    results_df, feature_importance = run_walk_forward_validation(
        df_clean, 
        feature_columns=FEATURES_BASIC, 
        target_col='mid_price'
    )
    
    print("\n" + "="*60)
    print("✅ XGBOOST TRAINING COMPLETE!")
    print("="*60)
    
    return results_df, feature_importance


if __name__ == "__main__":
    run_training()