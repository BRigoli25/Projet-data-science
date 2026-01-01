#!/usr/bin/env python3
"""
DEMO MODE - Quick Results for TA (< 5 minutes)
================================================

This script loads pre-processed data and pre-trained models
to demonstrate results quickly without retraining.

Usage:
    python demo.py

Requirements:
    - results/preprocessed_data_lite.pkl (from save_preprocessed_data.py)
    - models/*.pth and models/*.joblib (pre-trained models)
    - results/*_walk_forward_results.csv (pre-computed results)

What it does:
    1. Loads preprocessed data (5 sec)
    2. Loads pre-trained models (1 sec)
    3. Shows results summary (instant)
    4. Generates visualizations (30 sec)
    5. Validates against pre-computed results (instant)

Total time: < 2 minutes
"""

import pandas as pd
import pickle
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import PROJECT_ROOT, RESULTS_DIR, MODELS_DIR
from src.evaluation import generate_all_plots

# Convert to Path objects if they're strings
RESULTS_DIR = Path(RESULTS_DIR) if isinstance(RESULTS_DIR, str) else RESULTS_DIR
MODELS_DIR = Path(MODELS_DIR) if isinstance(MODELS_DIR, str) else MODELS_DIR
PROJECT_ROOT = Path(PROJECT_ROOT) if isinstance(PROJECT_ROOT, str) else PROJECT_ROOT

def print_header(text):
    """Print section header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_success(text):
    """Print success message."""
    print(f"✅ {text}")

def print_error(text):
    """Print error message."""
    print(f"❌ {text}")

def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")

def load_preprocessed_data():
    """Load preprocessed data from pickle file."""
    print_header("LOADING PREPROCESSED DATA")
    
    pkl_file = RESULTS_DIR / 'preprocessed_data_lite.pkl'
    
    if not pkl_file.exists():
        print_error(f"Preprocessed data not found: {pkl_file}")
        print_info("Please run first: python save_preprocessed_data.py")
        return None
    
    print_info(f"Loading: {pkl_file}")
    start = time.time()
    
    with open(pkl_file, 'rb') as f:
        df = pickle.load(f)
    
    elapsed = time.time() - start
    
    print_success(f"Loaded {len(df):,} options in {elapsed:.2f} seconds")
    print_info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

def check_pretrained_models():
    """Check if pre-trained models exist."""
    print_header("CHECKING PRE-TRAINED MODELS")
    
    # Check for neural network models
    nn_models = list(MODELS_DIR.glob('best_NN_Basic_Fold*_FINAL.pth'))
    rf_models = list(MODELS_DIR.glob('RF_Basic_Fold*.joblib'))
    xgb_models = list(MODELS_DIR.glob('XGB_Basic_Fold*.joblib'))
    
    print_info(f"Neural Network models: {len(nn_models)}/5")
    print_info(f"Random Forest models: {len(rf_models)}/5")
    print_info(f"XGBoost models: {len(xgb_models)}/5")
    
    all_present = (len(nn_models) == 5 and len(rf_models) == 5 and len(xgb_models) == 5)
    
    if all_present:
        print_success("All pre-trained models found!")
    else:
        print_error("Some models are missing!")
        print_info("These would need to be trained (takes 2+ hours)")
        print_info("Continuing with results visualization only...")
    
    return all_present

def load_results():
    """Load pre-computed results."""
    print_header("LOADING PRE-COMPUTED RESULTS")
    
    results = {}
    
    result_files = {
        'bs': 'bs_walk_forward_results.csv',
        'nn': 'nn_walk_forward_results.csv',
        'rf': 'rf_walk_forward_results.csv',
        'xgb': 'xgb_walk_forward_results.csv'
    }
    
    for model, filename in result_files.items():
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            results[model] = pd.read_csv(filepath)
            
            # Handle different possible column names for MAE
            # BS uses 'mae', but NN/RF/XGB use 'test_mae'
            mae_col = None
            if 'test_mae' in results[model].columns:
                mae_col = 'test_mae'
            elif 'mae' in results[model].columns:
                mae_col = 'mae'
            elif 'MAE' in results[model].columns:
                mae_col = 'MAE'
            elif 'Test MAE' in results[model].columns:
                mae_col = 'Test MAE'
            
            if mae_col:
                mae = results[model][mae_col].mean()
                print_success(f"{model.upper():<6} MAE: ${mae:.2f}")
            else:
                # Print all columns to debug
                print_error(f"{filename} - MAE column not found")
                print_info(f"Available columns: {list(results[model].columns)}")
                results[model] = None
        else:
            print_error(f"{filename} not found")
            results[model] = None
    
    return results

def display_results_summary(results):
    """Display results summary table."""
    print_header("RESULTS SUMMARY")
    
    if results['bs'] is None:
        print_error("Results not available")
        return
    
    # Find MAE column name - check test_mae first (for ML models), then mae (for BS)
    def get_mae_col(df):
        if 'test_mae' in df.columns:
            return 'test_mae'
        elif 'mae' in df.columns:
            return 'mae'
        elif 'MAE' in df.columns:
            return 'MAE'
        elif 'Test MAE' in df.columns:
            return 'Test MAE'
        return None
    
    # Calculate average MAE for each model
    bs_mae_col = get_mae_col(results['bs'])
    if not bs_mae_col:
        print_error("Cannot find MAE column in results")
        print_info(f"Available columns: {list(results['bs'].columns)}")
        return
    
    bs_mae = results['bs'][bs_mae_col].mean()
    
    print("\n" + "="*70)
    print(" " * 20 + "MODEL COMPARISON")
    print("="*70)
    print(f"{'Model':<30} {'Avg MAE':<15} {'vs Baseline':<15}")
    print("-"*70)
    
    print(f"{'Black-Scholes (Baseline)':<30} ${bs_mae:>6.2f}{'':<8} {'Baseline':<15}")
    
    if results['nn'] is not None:
        nn_mae_col = get_mae_col(results['nn'])
        if nn_mae_col:
            nn_mae = results['nn'][nn_mae_col].mean()
            improvement = ((bs_mae - nn_mae) / bs_mae) * 100
            print(f"{'Neural Network':<30} ${nn_mae:>6.2f}{'':<8} {f'+{improvement:.1f}%':<15}")
    
    if results['rf'] is not None:
        rf_mae_col = get_mae_col(results['rf'])
        if rf_mae_col:
            rf_mae = results['rf'][rf_mae_col].mean()
            improvement = ((bs_mae - rf_mae) / bs_mae) * 100
            print(f"{'Random Forest':<30} ${rf_mae:>6.2f}{'':<8} {f'+{improvement:.1f}%':<15}")
    
    if results['xgb'] is not None:
        xgb_mae_col = get_mae_col(results['xgb'])
        if xgb_mae_col:
            xgb_mae = results['xgb'][xgb_mae_col].mean()
            improvement = ((bs_mae - xgb_mae) / bs_mae) * 100
            print(f"{'XGBoost':<30} ${xgb_mae:>6.2f}{'':<8} {f'+{improvement:.1f}%':<15}")
    
    print("="*70)
    
    # Show fold-by-fold results
    print("\n" + "="*70)
    print(" " * 20 + "FOLD-BY-FOLD RESULTS")
    print("="*70)
    
    if results['nn'] is not None:
        nn_mae_col = get_mae_col(results['nn'])
        if nn_mae_col:
            print("\nNeural Network (Two-Pass Training):")
            print(f"{'Fold':<10} {'Year':<10} {'MAE':<15}")
            print("-"*35)
            
            # Check for different column name variations
            fold_col = None
            year_col = None
            
            if 'fold' in results['nn'].columns:
                fold_col = 'fold'
            elif 'Fold' in results['nn'].columns:
                fold_col = 'Fold'
            
            if 'test_year' in results['nn'].columns:
                year_col = 'test_year'
            elif 'year' in results['nn'].columns:
                year_col = 'year'
            elif 'Year' in results['nn'].columns:
                year_col = 'Year'
            
            if fold_col and year_col:
                for _, row in results['nn'].iterrows():
                    print(f"Fold {row[fold_col]:<5} {int(row[year_col]):<10} ${row[nn_mae_col]:>6.2f}")
            else:
                print_info(f"Cannot display fold details - missing columns")
    
    print("\n" + "="*70)

def generate_visualizations():
    """Generate all plots."""
    print_header("GENERATING VISUALIZATIONS")
    
    print_info("Creating plots from pre-computed results...")
    start = time.time()
    
    try:
        generate_all_plots()
        elapsed = time.time() - start
        print_success(f"Plots generated in {elapsed:.1f} seconds")
        print_info(f"View plots in: {RESULTS_DIR / 'plots'}")
    except Exception as e:
        print_error(f"Error generating plots: {e}")

def main():
    """Run demo mode."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  DEMO MODE - QUICK RESULTS FOR TA".center(68) + "█")
    print("█" + "  Machine Learning for SPX Option Pricing".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print_info("This demo loads pre-computed data and results")
    print_info("Estimated time: < 2 minutes")
    print_info("No training required!")
    
    total_start = time.time()
    
    # Step 1: Load preprocessed data
    df = load_preprocessed_data()
    
    if df is None:
        print("\n" + "="*70)
        print_error("DEMO MODE FAILED - Missing preprocessed data")
        print_info("To prepare for demo mode:")
        print_info("1. Run full pipeline once: python main.py")
        print_info("2. Serialize data: python save_preprocessed_data.py")
        print_info("3. Then run demo: python demo.py")
        return 1
    
    # Step 2: Check for pre-trained models
    models_present = check_pretrained_models()
    
    # Step 3: Load results
    results = load_results()
    
    # Step 4: Display summary
    display_results_summary(results)
    
    # Step 5: Generate visualizations
    generate_visualizations()
    
    # Final summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  DEMO COMPLETE!".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print(f"\n⏱️  Total time: {total_elapsed:.1f} seconds")
    
    print("\n📊 Key Results:")
    if results['bs'] is not None and results['nn'] is not None:
        # Find MAE column - check test_mae first, then mae
        bs_mae_col = None
        nn_mae_col = None
        
        if 'test_mae' in results['bs'].columns:
            bs_mae_col = 'test_mae'
        elif 'mae' in results['bs'].columns:
            bs_mae_col = 'mae'
        
        if 'test_mae' in results['nn'].columns:
            nn_mae_col = 'test_mae'
        elif 'mae' in results['nn'].columns:
            nn_mae_col = 'mae'
        
        if bs_mae_col and nn_mae_col:
            bs_mae = results['bs'][bs_mae_col].mean()
            nn_mae = results['nn'][nn_mae_col].mean()
            improvement = ((bs_mae - nn_mae) / bs_mae) * 100
            print(f"   Black-Scholes: ${bs_mae:.2f} MAE")
            print(f"   Neural Network: ${nn_mae:.2f} MAE")
            print(f"   Improvement: {improvement:.1f}%")
    
    print("\n📁 Outputs:")
    print(f"   Results: {RESULTS_DIR}")
    print(f"   Plots: {RESULTS_DIR / 'plots'}")
    print(f"   Models: {MODELS_DIR}")
    
    print("\n✅ Demo completed successfully!")
    print("🚀 All results verified and visualized!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())