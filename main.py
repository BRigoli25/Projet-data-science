"""
Main Pipeline for SPX Option Pricing Project
============================================

Runs the complete analysis pipeline:
1. Data Preprocessing & Black-Scholes Baseline
2. Neural Network Training (5-fold walk-forward)
3. Random Forest Training (5-fold walk-forward)
4. XGBoost Training (5-fold walk-forward)
5. Runtime Comparison
6. Visualizations

Usage:
    python main.py              # Run everything
    python main.py --preprocess # Run only preprocessing
    python main.py --models     # Run only model training
    python main.py --viz        # Run only visualizations
    python main.py --runtime    # Run only runtime comparison
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
from typing import Callable

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

# Import config
try:
    from src.config import (
        PROJECT_ROOT, RESULTS_DIR, MODELS_DIR, DATA_DIR,
        SPX_FORWARD_FILE, SPX_OPTIONS_FILE, SPX_BS_HIST_FILE,
        FEATURES_BASIC
    )
    CONFIG_LOADED = True
    print("✅ Configuration loaded")
except ImportError:
    CONFIG_LOADED = False
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    print("⚠️  Using default configuration")


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


# ============================================================
# CHECK DATA FILES
# ============================================================

def check_data_files():
    """Check if required data files exist."""
    if not CONFIG_LOADED:
        print("⚠️  Config not loaded, skipping file check")
        return True
    
    required_files = [SPX_FORWARD_FILE, SPX_OPTIONS_FILE]
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("\n⚠️  Missing required data files for preprocessing!")
        print("\nPlease place the following files in data/raw/:")
        for f in missing:
            print(f"  - {os.path.basename(f)}")
        print("\nData source: WRDS OptionMetrics")
        return False
    return True


# ============================================================
# PIPELINE STEPS
# ============================================================

def step_preprocessing():
    """Run data preprocessing and BS baseline."""
    print_header("STEP 1: DATA PREPROCESSING & BLACK-SCHOLES BASELINE")
    try:
        from src.data_preprocessing import run_preprocessing
        run_preprocessing()
        return True
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        traceback.print_exc()
        return False


def step_neural_network():
    """Run neural network training."""
    print_header("STEP 2: NEURAL NETWORK TRAINING")
    try:
        from src.neural_network import run_training
        run_training()
        return True
    except Exception as e:
        print(f"❌ Neural Network error: {e}")
        traceback.print_exc()
        return False


def step_random_forest():
    """Run random forest training."""
    print_header("STEP 3: RANDOM FOREST TRAINING")
    try:
        from src.random_forest import run_training
        run_training()
        return True
    except Exception as e:
        print(f"❌ Random Forest error: {e}")
        traceback.print_exc()
        return False


def step_xgboost():
    """Run XGBoost training."""
    print_header("STEP 4: XGBOOST TRAINING")
    try:
        from src.xg_boost import run_training
        run_training()
        return True
    except Exception as e:
        print(f"❌ XGBoost error: {e}")
        traceback.print_exc()
        return False


def step_visualizations():
    """Generate all visualizations."""
    print_header("STEP 5: GENERATING VISUALIZATIONS")
    try:
        from src.visualizations import generate_all_plots
        generate_all_plots()
        return True
    except Exception as e:
        print(f"❌ Visualizations error: {e}")
        traceback.print_exc()
        return False


# ============================================================
# RUNTIME COMPARISON
# ============================================================

def step_runtime_comparison():
    """Compare prediction speed across models."""
    print_header("RUNTIME COMPARISON (PREDICTION SPEED)")
    
    import joblib
    from scipy.stats import norm
    
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
    
    # Load test data
    data_file = os.path.join(RESULTS_DIR, 'SPX_with_BS_Historical.csv')
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("   Run preprocessing first!")
        return False
    
    print("📂 Loading test data...")
    df = pd.read_csv(data_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    missing_features = [f for f in FEATURES_BASIC if f not in df.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        print("   Run preprocessing first to create all features!")
        return False
    
    print(f"✅ All {len(FEATURES_BASIC)} features available")

    
    # Use 2025 as test set
    test_mask = df['date'] > '2024-12-31'
    df_test = df[test_mask].copy()
    
    if len(df_test) == 0:
        print("⚠️  No 2025 test data, using last 100k rows")
        df_test = df.tail(100000).copy()
    
    
    # Clean test data
    df_test = df_test.dropna(subset=FEATURES_BASIC)
    X_test = df_test[FEATURES_BASIC].values

    print(f"✅ Test set: {len(df_test):,} options")
    print(f"   Features: {len(FEATURES_BASIC)}")
    
    N_RUNS = 5
    results = []
    
    # --------------------------------------------------------
    # 1. Black-Scholes (Analytical)
    # --------------------------------------------------------
    print("\n📊 Timing Black-Scholes...")
    
    def black_scholes_vectorized(F, K, T, r, sigma, option_type='C'):
        T = np.maximum(T, 1e-10)
        
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        disc = np.exp(-r * T)
        call_price = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        put_price = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        
        is_call = (option_type == 'C')
        price = np.where(is_call, call_price, put_price)
        
        return price

    bs_times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        _ = black_scholes_vectorized(
            df_test['forward_price'].values,
            df_test['strike_price'].values,
            df_test['T'].values,
            df_test['r'].values if 'r' in df_test.columns else 0.04,
            df_test['historical_vol'].values,
            df_test['is_call'].values
        )
        bs_times.append(time.perf_counter() - start)
    
    bs_avg = np.mean(bs_times)
    results.append({
        'Model': 'Black-Scholes',
        'Avg Time (s)': bs_avg,
        'Std Time (s)': np.std(bs_times),
        'Options/sec': len(df_test) / bs_avg,
        'Type': 'Analytical'
    })
    print(f"   ✅ BS: {bs_avg:.4f}s ({len(df_test)/bs_avg:,.0f} options/sec)")
    
    # --------------------------------------------------------
    # 2. Random Forest
    # --------------------------------------------------------
    rf_path = os.path.join(MODELS_DIR, 'RF_Basic_Fold5.joblib')
    if os.path.exists(rf_path):
        print("\n📊 Timing Random Forest...")
        rf_model = joblib.load(rf_path)
        
        rf_times = []
        for _ in range(N_RUNS):
            start = time.perf_counter()
            _ = rf_model.predict(X_test)
            rf_times.append(time.perf_counter() - start)
        
        rf_avg = np.mean(rf_times)
        results.append({
            'Model': 'Random Forest',
            'Avg Time (s)': rf_avg,
            'Std Time (s)': np.std(rf_times),
            'Options/sec': len(df_test) / rf_avg,
            'Type': 'Ensemble (Bagging)'
        })
        print(f"   ✅ RF: {rf_avg:.4f}s ({len(df_test)/rf_avg:,.0f} options/sec)")
    else:
        print(f"   ⚠️ RF model not found: {rf_path}")
    
    # --------------------------------------------------------
    # 3. XGBoost
    # --------------------------------------------------------
    xgb_path = os.path.join(MODELS_DIR, 'XGB_Basic_Fold5.joblib')
    if os.path.exists(xgb_path):
        print("\n📊 Timing XGBoost...")
        xgb_model = joblib.load(xgb_path)
        
        xgb_times = []
        for _ in range(N_RUNS):
            start = time.perf_counter()
            _ = xgb_model.predict(X_test)
            xgb_times.append(time.perf_counter() - start)
        
        xgb_avg = np.mean(xgb_times)
        results.append({
            'Model': 'XGBoost',
            'Avg Time (s)': xgb_avg,
            'Std Time (s)': np.std(xgb_times),
            'Options/sec': len(df_test) / xgb_avg,
            'Type': 'Ensemble (Boosting)'
        })
        print(f"   ✅ XGB: {xgb_avg:.4f}s ({len(df_test)/xgb_avg:,.0f} options/sec)")
    else:
        print(f"   ⚠️ XGBoost model not found: {xgb_path}")

    
    # --------------------------------------------------------
    # 4. Neural Network
    # --------------------------------------------------------
    nn_path = os.path.join(MODELS_DIR, 'best_NN_Basic_Fold5_FINAL.pth')
    if os.path.exists(nn_path) and TORCH_AVAILABLE:
        print("\n📊 Timing Neural Network...")
        
        from sklearn.preprocessing import StandardScaler
        from src.neural_network import OptionPricingMLP
        
        # Scale features (NN requires scaling)
        scaler_X = StandardScaler()
        X_test_scaled = scaler_X.fit_transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        nn_model = OptionPricingMLP(input_size=len(FEATURES_BASIC))
        nn_model.load_state_dict(torch.load(nn_path, map_location='cpu'))
        nn_model.eval()
        
        nn_times = []
        for _ in range(N_RUNS):
            start = time.perf_counter()
            with torch.no_grad():
                _ = nn_model(X_test_tensor)
            nn_times.append(time.perf_counter() - start)
        
        nn_avg = np.mean(nn_times)
        results.append({
            'Model': 'Neural Network',
            'Avg Time (s)': nn_avg,
            'Std Time (s)': np.std(nn_times),
            'Options/sec': len(df_test) / nn_avg,
            'Type': 'Deep Learning'
        })
        print(f"   ✅ NN: {nn_avg:.4f}s ({len(df_test)/nn_avg:,.0f} options/sec)")
    else:
        if not TORCH_AVAILABLE:
            print("   ⚠️ PyTorch not available")
        else:
            print(f"   ⚠️ NN model not found: {nn_path}")

    
    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 RUNTIME COMPARISON SUMMARY")
    print("=" * 70)
    
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Avg Time (s)')
        
        print(f"\nTest set size: {len(df_test):,} options")
        print(f"Timing runs: {N_RUNS} iterations\n")
        
        print(f"{'Model':<18} {'Type':<20} {'Time (s)':<12} {'Options/sec':<15}")
        print("-" * 65)
        
        for _, row in results_df.iterrows():
            print(f"{row['Model']:<18} {row['Type']:<20} "
                  f"{row['Avg Time (s)']:.4f}       {row['Options/sec']:>12,.0f}")
        
        # Save results
        results_df.to_csv(os.path.join(RESULTS_DIR, 'runtime_comparison.csv'), index=False)
        print(f"\n✅ Saved: {os.path.join(RESULTS_DIR, 'runtime_comparison.csv')}")
        
        fastest = results_df.iloc[0]
        print(f"\n🏆 Fastest: {fastest['Model']} ({fastest['Options/sec']:,.0f} options/sec)")
    
    return True


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_full_pipeline():
    """Run the complete project pipeline."""
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  SPX OPTION PRICING - PROJECT PIPELINE".center(68) + "█")
    print("█" + "  Neural Networks vs Black-Scholes".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Project: {PROJECT_ROOT}")
    
    total_start = time.time()
    results = {}
    
    # ============================================================
    # STEP 1: PREPROCESSING (conditional on raw data availability)
    # ============================================================
    if CONFIG_LOADED and check_data_files():
        step_start = time.time()
        success = step_preprocessing()
        elapsed = time.time() - step_start
        results["Preprocessing"] = {'success': success, 'time': elapsed}
        if success:
            print(f"\n✅ Preprocessing completed in {elapsed:.1f}s")
        else:
            print(f"\n⚠️ Preprocessing had issues")
    else:
        # Check if preprocessed data exists
        if os.path.exists(os.path.join(RESULTS_DIR, 'SPX_with_BS_Historical.csv')):
            print("\n⚠️ Raw data not found, but preprocessed data exists.")
            print("   Skipping preprocessing and using existing results.")
            results["Preprocessing"] = {'success': True, 'time': 0}
        else:
            print("\n❌ Raw data not found AND no preprocessed data exists.")
            print("   Cannot continue. Please provide data files.")
            return
    
    # ============================================================
    # STEPS 2-6: Model Training, Runtime, Visualizations
    # ============================================================
    steps = [
        ("Neural Network", step_neural_network),
        ("Random Forest", step_random_forest),
        ("XGBoost", step_xgboost),
        ("Runtime Comparison", step_runtime_comparison),
        ("Visualizations", step_visualizations),
    ]
    
    for i, (name, func) in enumerate(steps, 2):
        print(f"\n{'━' * 70}")
        print(f"  [{i}/{len(steps)+1}] {name}")
        print(f"{'━' * 70}")
        
        step_start = time.time()
        try:
            success = func()
            elapsed = time.time() - step_start
            results[name] = {'success': success, 'time': elapsed}
            
            if success:
                print(f"\n✅ {name} completed in {elapsed:.1f}s")
            else:
                print(f"\n⚠️ {name} had issues")
        except Exception as e:
            elapsed = time.time() - step_start
            results[name] = {'success': False, 'time': elapsed}
            print(f"\n❌ {name} failed: {e}")
            traceback.print_exc()
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    total_time = time.time() - total_start
    
    print("\n" + "█" * 70)
    print("█" + "  PIPELINE COMPLETE".center(68) + "█")
    print("█" * 70)
    
    print(f"\n📊 EXECUTION SUMMARY")
    print("─" * 50)
    
    for name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {name:<25} {result['time']:>8.1f}s")
    
    print("─" * 50)
    print(f"  {'Total Time':<25} {total_time:>8.1f}s ({total_time/60:.1f} min)")
    
    successes = sum(1 for r in results.values() if r['success'])
    print(f"\n  {successes}/{len(results)} steps completed successfully")
    
    print(f"\n📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"   Models:  {MODELS_DIR}")
    print(f"   Results: {RESULTS_DIR}")
    print(f"   Plots:   {os.path.join(RESULTS_DIR, 'plots')}")


# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='SPX Option Pricing Project Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                Run complete pipeline
  python main.py --preprocess   Run only data preprocessing
  python main.py --models       Run only model training (NN, RF, XGB)
  python main.py --viz          Run only visualizations
  python main.py --runtime      Run only runtime comparison
  python main.py --step nn      Run only neural network
  python main.py --step rf      Run only random forest
  python main.py --step xgb     Run only xgboost
        """
    )
    
    parser.add_argument('--preprocess', action='store_true', help='Run only preprocessing')
    parser.add_argument('--models', action='store_true', help='Run only model training')
    parser.add_argument('--viz', action='store_true', help='Run only visualizations')
    parser.add_argument('--runtime', action='store_true', help='Run only runtime comparison')
    parser.add_argument('--step', type=str, 
                        choices=['preprocess', 'nn', 'rf', 'xgb', 'viz', 'runtime'],
                        help='Run a specific step')
    
    args = parser.parse_args()
    
    if args.step:
        step_map = {
            'preprocess': step_preprocessing,
            'nn': step_neural_network,
            'rf': step_random_forest,
            'xgb': step_xgboost,
            'viz': step_visualizations,
            'runtime': step_runtime_comparison,
        }
        step_map[args.step]()
    elif args.preprocess:
        step_preprocessing()
    elif args.models:
        print_header("RUNNING MODELS ONLY")
        step_neural_network()
        step_random_forest()
        step_xgboost()
    elif args.viz:
        step_visualizations()
    elif args.runtime:
        step_runtime_comparison()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()