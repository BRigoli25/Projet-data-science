"""
Main Pipeline for SPX Option Pricing Project
============================================

Runs the complete analysis pipeline:
1. Data Preprocessing & Black-Scholes Baseline
2. Neural Network Training (5-fold walk-forward)
3. Random Forest Training (5-fold walk-forward)
4. XGBoost Training (5-fold walk-forward)
5. Visualizations

Usage:
    python main.py              # Run everything
    python main.py --preprocess # Run only preprocessing
    python main.py --models     # Run only model training
    python main.py --viz        # Run only visualizations
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

# Import modules (NEW STRUCTURE)
from src.config import RESULTS_DIR, MODELS_DIR
from src.data_loader import run_preprocessing
from src.models import train_neural_network, train_random_forest, train_xgboost
from src.evaluation import generate_all_plots


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_full_pipeline():
    """Run the complete project pipeline."""
    
    print("\n" + "█" * 70)
    print("█" + "  SPX OPTION PRICING - PROJECT PIPELINE".center(68) + "█")
    print("█" + "  Neural Networks vs Black-Scholes".center(68) + "█")
    print("█" * 70)
    
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    results = {}
    
    # Step 1: Preprocessing
    print_header("STEP 1: DATA PREPROCESSING & BLACK-SCHOLES")
    step_start = time.time()
    try:
        run_preprocessing()
        elapsed = time.time() - step_start
        results["Preprocessing"] = {'success': True, 'time': elapsed}
        print(f"\n✅ Preprocessing completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - step_start
        results["Preprocessing"] = {'success': False, 'time': elapsed}
        print(f"\n❌ Preprocessing failed: {e}")
    
    # Step 2: Neural Network
    print_header("STEP 2: NEURAL NETWORK TRAINING")
    step_start = time.time()
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(RESULTS_DIR, 'SPX_with_BS_Historical.csv'), low_memory=False)
        df['date'] = pd.to_datetime(df['date'])
        df_clean = df.dropna(subset=['moneyness', 'T', 'mid_price'])
        
        from src.config import FEATURES_BASIC
        train_neural_network(df_clean, FEATURES_BASIC)
        
        elapsed = time.time() - step_start
        results["Neural Network"] = {'success': True, 'time': elapsed}
        print(f"\n✅ Neural Network completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - step_start
        results["Neural Network"] = {'success': False, 'time': elapsed}
        print(f"\n❌ Neural Network failed: {e}")
    
    # Step 3: Random Forest
    print_header("STEP 3: RANDOM FOREST TRAINING")
    step_start = time.time()
    try:
        train_random_forest(df_clean, FEATURES_BASIC)
        elapsed = time.time() - step_start
        results["Random Forest"] = {'success': True, 'time': elapsed}
        print(f"\n✅ Random Forest completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - step_start
        results["Random Forest"] = {'success': False, 'time': elapsed}
        print(f"\n❌ Random Forest failed: {e}")
    
    # Step 4: XGBoost
    print_header("STEP 4: XGBOOST TRAINING")
    step_start = time.time()
    try:
        train_xgboost(df_clean, FEATURES_BASIC)
        elapsed = time.time() - step_start
        results["XGBoost"] = {'success': True, 'time': elapsed}
        print(f"\n✅ XGBoost completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - step_start
        results["XGBoost"] = {'success': False, 'time': elapsed}
        print(f"\n❌ XGBoost failed: {e}")
    
    # Step 5: Visualizations
    print_header("STEP 5: GENERATING VISUALIZATIONS")
    step_start = time.time()
    try:
        generate_all_plots()
        elapsed = time.time() - step_start
        results["Visualizations"] = {'success': True, 'time': elapsed}
        print(f"\n✅ Visualizations completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - step_start
        results["Visualizations"] = {'success': False, 'time': elapsed}
        print(f"\n❌ Visualizations failed: {e}")
    
    # Final summary
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
    
    print(f"\n📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='SPX Option Pricing Pipeline')
    
    parser.add_argument('--preprocess', action='store_true', help='Run only preprocessing')
    parser.add_argument('--models', action='store_true', help='Run only model training')
    parser.add_argument('--viz', action='store_true', help='Run only visualizations')
    
    args = parser.parse_args()
    
    if args.preprocess:
        run_preprocessing()
    elif args.models:
        import pandas as pd
        from src.config import FEATURES_BASIC
        df = pd.read_csv(os.path.join(RESULTS_DIR, 'SPX_with_BS_Historical.csv'), low_memory=False)
        df['date'] = pd.to_datetime(df['date'])
        df_clean = df.dropna()
        
        train_neural_network(df_clean, FEATURES_BASIC)
        train_random_forest(df_clean, FEATURES_BASIC)
        train_xgboost(df_clean, FEATURES_BASIC)
    elif args.viz:
        generate_all_plots()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()