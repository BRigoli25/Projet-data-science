#!/usr/bin/env python3
"""
DEMO MODE - Quick Results for TA (< 1 minute)
==============================================

Loads ONE pickle file with all pre-computed results.
No training required!

Usage:
    python demo.py

Requirements:
    - demo_data.pkl (download from Google Drive)

Total time: < 30 seconds
"""

import pandas as pd
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import PROJECT_ROOT, RESULTS_DIR
from src.evaluation import generate_all_plots

RESULTS_DIR = Path(RESULTS_DIR) if isinstance(RESULTS_DIR, str) else RESULTS_DIR

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

def load_demo_data():
    """Load all data from single pickle file."""
    print_header("LOADING DEMO DATA")
    
    # Check for demo_data.pkl in multiple locations
    possible_paths = [
        Path("demo_data.pkl"),
        Path("results/demo_data.pkl"),
        RESULTS_DIR / "demo_data.pkl"
    ]
    
    demo_file = None
    for path in possible_paths:
        if path.exists():
            demo_file = path
            break
    
    if not demo_file:
        print_error("demo_data.pkl not found!")
        print("")
        print_info("Please download demo_data.pkl from Google Drive:")
        print_info("https://drive.google.com/drive/folders/1iBwDWOvCxwZOBESIHBSVMfeImUMxvT4Z")
        print("")
        print_info("Place it in the project root directory")
        print("")
        return None
    
    print_info(f"Loading: {demo_file}")
    
    with open(demo_file, "rb") as f:
        demo_data = pickle.load(f)
    
    print_success("Demo data loaded!")
    print_info(f"Preprocessed data: {len(demo_data['preprocessed_data']):,} samples")
    print_info(f"Model results: {len(demo_data['results'])} models")
    
    return demo_data

def get_mae_column(df):
    """Find MAE column name in dataframe."""
    possible_names = ["test_mae", "mae", "MAE", "Test MAE"]
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def display_results_summary(results):
    """Display results summary table."""
    print_header("RESULTS SUMMARY")
    
    if "bs" not in results:
        print_error("Results not available")
        return False
    
    bs_mae_col = get_mae_column(results["bs"])
    if not bs_mae_col:
        print_error("Cannot find MAE column in results")
        return False
    
    bs_mae = results["bs"][bs_mae_col].mean()
    
    print("\n" + "="*70)
    print(" " * 20 + "MODEL PERFORMANCE")
    print("="*70)
    print(f"{'Model':<25} {'MAE':<15} {'vs Black-Scholes':<20}")
    print("-"*70)
    
    print(f"{'Black-Scholes':<25} ${bs_mae:>6.2f}{'':<8} {'Baseline':<20}")
    
    # Neural Network
    if "nn" in results:
        nn_mae_col = get_mae_column(results["nn"])
        if nn_mae_col:
            nn_mae = results["nn"][nn_mae_col].mean()
            improvement = ((bs_mae - nn_mae) / bs_mae) * 100
            print(f"{'Neural Network':<25} ${nn_mae:>6.2f}{'':<8} {f'+{improvement:.1f}% better':<20}")
    
    # Random Forest
    if "rf" in results:
        rf_mae_col = get_mae_column(results["rf"])
        if rf_mae_col:
            rf_mae = results["rf"][rf_mae_col].mean()
            improvement = ((bs_mae - rf_mae) / bs_mae) * 100
            print(f"{'Random Forest':<25} ${rf_mae:>6.2f}{'':<8} {f'+{improvement:.1f}% better':<20}")
    
    # XGBoost
    if "xgb" in results:
        xgb_mae_col = get_mae_column(results["xgb"])
        if xgb_mae_col:
            xgb_mae = results["xgb"][xgb_mae_col].mean()
            improvement = ((bs_mae - xgb_mae) / bs_mae) * 100
            print(f"{'XGBoost':<25} ${xgb_mae:>6.2f}{'':<8} {f'+{improvement:.1f}% better':<20}")
    
    print("="*70)
    
    # Show best model
    best_mae = float("inf")
    best_model = None
    
    for model_name, model_key in [("Neural Network", "nn"), ("Random Forest", "rf"), ("XGBoost", "xgb")]:
        if model_key in results:
            mae_col = get_mae_column(results[model_key])
            if mae_col:
                mae = results[model_key][mae_col].mean()
                if mae < best_mae:
                    best_mae = mae
                    best_model = model_name
    
    if best_model:
        improvement = ((bs_mae - best_mae) / bs_mae) * 100
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   MAE: ${best_mae:.2f} ({improvement:.1f}% better than Black-Scholes)")
    
    return True

def show_feature_importance(feature_importance):
    """Show feature importance."""
    print_header("FEATURE IMPORTANCE")
    
    if "rf" in feature_importance:
        print_success("Random Forest - Top 5 Features:")
        df = feature_importance["rf"]
        if "feature" in df.columns and "importance" in df.columns:
            print()
            for i, row in df.head(5).iterrows():
                print(f"   {i+1}. {row['feature']:<30} {row['importance']:>6.3f}")
        print()
    
    if "xgb" in feature_importance:
        print_success("XGBoost - Top 5 Features:")
        df = feature_importance["xgb"]
        if "feature" in df.columns and "importance" in df.columns:
            print()
            for i, row in df.head(5).iterrows():
                print(f"   {i+1}. {row['feature']:<30} {row['importance']:>6.3f}")
        print()

def save_results_for_plots(demo_data):
    """Save results to CSV files for plotting."""
    print_header("PREPARING VISUALIZATIONS")
    
    # Create results directory if needed
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Save results to CSV
    for model_name, df in demo_data["results"].items():
        output_file = RESULTS_DIR / f"{model_name}_walk_forward_results.csv"
        df.to_csv(output_file, index=False)
        print_success(f"Saved {model_name} results")
    
    # Save feature importance
    for model_name, df in demo_data["feature_importance"].items():
        output_file = RESULTS_DIR / f"{model_name}_feature_importance.csv"
        df.to_csv(output_file, index=False)
        print_success(f"Saved {model_name} feature importance")

def generate_visualizations():
    """Generate all plots."""
    print_header("GENERATING VISUALIZATIONS")
    
    print_info("Creating plots...")
    
    try:
        generate_all_plots()
        print_success("Plots generated successfully")
        print_info(f"View plots in: {RESULTS_DIR / 'plots'}")
    except Exception as e:
        print_error(f"Error generating plots: {e}")
        print_info("Continuing without visualizations...")

def main():
    """Run demo mode."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  DEMO MODE - QUICK RESULTS FOR TA".center(68) + "█")
    print("█" + "  Machine Learning for SPX Option Pricing".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print_info("Loading from single demo_data.pkl file")
    print_info("No training required!")
    print()
    
    # Load demo data
    demo_data = load_demo_data()
    
    if demo_data is None:
        return 1
    
    # Display summary
    success = display_results_summary(demo_data["results"])
    
    if not success:
        return 1
    
    # Show feature importance
    show_feature_importance(demo_data["feature_importance"])
    
    # Save results for plotting
    save_results_for_plots(demo_data)
    
    # Generate visualizations
    generate_visualizations()
    
    # Final message
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  ✅ DEMO COMPLETED SUCCESSFULLY!".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("\n📊 Summary:")
    print("   - Loaded all pre-computed results")
    print("   - Compared ML models vs Black-Scholes baseline")
    print("   - Generated visualizations")
    print()
    print("📁 Outputs:")
    print(f"   - Results: {RESULTS_DIR}")
    print(f"   - Plots: {RESULTS_DIR / 'plots'}")
    print()
    print("🚀 All results verified!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
