"""
Visualizations for SPX Option Pricing Project
=============================================

Publication-ready plots for thesis:
- Fold-by-fold MAE comparison (overall + ATM)
- Error vs Moneyness (key diagnostic)
- Error vs Bid-Ask Spread (microstructure insight)
- RMSE comparison
- Call/Put performance
- Feature importance
- Model comparison bar chart
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Publication style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25

# ============================================================
# CONFIGURATION
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.config import RESULTS_DIR, PROJECT_ROOT
    print("✅ Configuration loaded")
except ImportError:
    PROJECT_ROOT = project_root
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

YEARS = ['2021', '2022', '2023', '2024', '2025']

# Color scheme
COLORS = {
    'BS (Historical Vol)': '#E74C3C',
    'Neural Network': '#3498DB',
    'Random Forest': '#27AE60',
    'XGBoost': '#9B59B6'
}

MARKERS = {
    'BS (Historical Vol)': 's',
    'Neural Network': 'o',
    'Random Forest': '^',
    'XGBoost': 'd'
}


# ============================================================
# DATA LOADING
# ============================================================

def load_results():
    """Load all model results from CSV files."""
    print("\n📂 Loading results from CSV files...")
    
    results = {}
    
    # Black-Scholes
    bs_path = os.path.join(RESULTS_DIR, 'bs_walk_forward_results.csv')
    if os.path.exists(bs_path):
        bs_df = pd.read_csv(bs_path)
        # Handle different column name variations
        mae_col = next((c for c in ['mae', 'test_mae', 'MAE'] if c in bs_df.columns), None)
        rmse_col = next((c for c in ['rmse', 'test_rmse', 'RMSE'] if c in bs_df.columns), None)
        
        results['BS (Historical Vol)'] = {
            'df': bs_df,
            'avg_mae': bs_df[mae_col].mean() if mae_col else np.nan,
            'avg_rmse': bs_df[rmse_col].mean() if rmse_col else np.nan,
            'fold_mae': bs_df[mae_col].tolist() if mae_col else [],
            'fold_rmse': bs_df[rmse_col].tolist() if rmse_col else [],
            'avg_mae_atm': bs_df['mae_atm'].mean() if 'mae_atm' in bs_df.columns else np.nan,
            'fold_mae_atm': bs_df['mae_atm'].tolist() if 'mae_atm' in bs_df.columns else [],
            'avg_mae_call': bs_df['mae_call'].mean() if 'mae_call' in bs_df.columns else np.nan,
            'avg_mae_put': bs_df['mae_put'].mean() if 'mae_put' in bs_df.columns else np.nan,
        }
        print(f"   ✓ BS: MAE=${results['BS (Historical Vol)']['avg_mae']:.2f}")
    
    # Neural Network
    nn_path = os.path.join(RESULTS_DIR, 'nn_walk_forward_results.csv')
    if os.path.exists(nn_path):
        nn_df = pd.read_csv(nn_path)
        mae_col = next((c for c in ['test_mae', 'mae', 'MAE'] if c in nn_df.columns), None)
        rmse_col = next((c for c in ['test_rmse', 'rmse', 'RMSE'] if c in nn_df.columns), None)
        
        results['Neural Network'] = {
            'df': nn_df,
            'avg_mae': nn_df[mae_col].mean() if mae_col else np.nan,
            'avg_rmse': nn_df[rmse_col].mean() if rmse_col else np.nan,
            'fold_mae': nn_df[mae_col].tolist() if mae_col else [],
            'fold_rmse': nn_df[rmse_col].tolist() if rmse_col else [],
            'avg_mae_atm': nn_df['mae_atm'].mean() if 'mae_atm' in nn_df.columns else np.nan,
            'fold_mae_atm': nn_df['mae_atm'].tolist() if 'mae_atm' in nn_df.columns else [],
            'avg_mae_call': nn_df['mae_call'].mean() if 'mae_call' in nn_df.columns else np.nan,
            'avg_mae_put': nn_df['mae_put'].mean() if 'mae_put' in nn_df.columns else np.nan,
        }
        print(f"   ✓ NN: MAE=${results['Neural Network']['avg_mae']:.2f}")
    
    # Random Forest
    rf_path = os.path.join(RESULTS_DIR, 'rf_walk_forward_results.csv')
    if os.path.exists(rf_path):
        rf_df = pd.read_csv(rf_path)
        mae_col = next((c for c in ['test_mae', 'mae', 'MAE'] if c in rf_df.columns), None)
        rmse_col = next((c for c in ['test_rmse', 'rmse', 'RMSE'] if c in rf_df.columns), None)
        
        results['Random Forest'] = {
            'df': rf_df,
            'avg_mae': rf_df[mae_col].mean() if mae_col else np.nan,
            'avg_rmse': rf_df[rmse_col].mean() if rmse_col else np.nan,
            'fold_mae': rf_df[mae_col].tolist() if mae_col else [],
            'fold_rmse': rf_df[rmse_col].tolist() if rmse_col else [],
            'avg_mae_atm': rf_df['mae_atm'].mean() if 'mae_atm' in rf_df.columns else np.nan,
            'fold_mae_atm': rf_df['mae_atm'].tolist() if 'mae_atm' in rf_df.columns else [],
            'avg_mae_call': rf_df['mae_call'].mean() if 'mae_call' in rf_df.columns else np.nan,
            'avg_mae_put': rf_df['mae_put'].mean() if 'mae_put' in rf_df.columns else np.nan,
        }
        print(f"   ✓ RF: MAE=${results['Random Forest']['avg_mae']:.2f}")
    
    # XGBoost
    xgb_path = os.path.join(RESULTS_DIR, 'xgb_walk_forward_results.csv')
    if os.path.exists(xgb_path):
        xgb_df = pd.read_csv(xgb_path)
        mae_col = next((c for c in ['test_mae', 'mae', 'MAE'] if c in xgb_df.columns), None)
        rmse_col = next((c for c in ['test_rmse', 'rmse', 'RMSE'] if c in xgb_df.columns), None)
        
        results['XGBoost'] = {
            'df': xgb_df,
            'avg_mae': xgb_df[mae_col].mean() if mae_col else np.nan,
            'avg_rmse': xgb_df[rmse_col].mean() if rmse_col else np.nan,
            'fold_mae': xgb_df[mae_col].tolist() if mae_col else [],
            'fold_rmse': xgb_df[rmse_col].tolist() if rmse_col else [],
            'avg_mae_atm': xgb_df['mae_atm'].mean() if 'mae_atm' in xgb_df.columns else np.nan,
            'fold_mae_atm': xgb_df['mae_atm'].tolist() if 'mae_atm' in xgb_df.columns else [],
            'avg_mae_call': xgb_df['mae_call'].mean() if 'mae_call' in xgb_df.columns else np.nan,
            'avg_mae_put': xgb_df['mae_put'].mean() if 'mae_put' in xgb_df.columns else np.nan,
        }
        print(f"   ✓ XGB: MAE=${results['XGBoost']['avg_mae']:.2f}")
    
    return results


def load_feature_importance():
    """Load feature importance from CSV files."""
    rf_importance, xgb_importance = {}, {}
    
    rf_path = os.path.join(RESULTS_DIR, 'rf_feature_importance.csv')
    if os.path.exists(rf_path):
        rf_df = pd.read_csv(rf_path)
        rf_importance = dict(zip(rf_df['feature'], rf_df['importance']))
        print(f"   ✓ RF feature importance loaded")
    
    xgb_path = os.path.join(RESULTS_DIR, 'xgb_feature_importance.csv')
    if os.path.exists(xgb_path):
        xgb_df = pd.read_csv(xgb_path)
        xgb_importance = dict(zip(xgb_df['feature'], xgb_df['importance']))
        print(f"   ✓ XGB feature importance loaded")
    
    return rf_importance, xgb_importance


def load_full_data():
    """Load full dataset with predictions for diagnostic plots."""
    # Try multiple possible file locations
    possible_paths = [
        os.path.join(RESULTS_DIR, 'SPX_features.csv'),  # Preprocessed with features
        os.path.join(RESULTS_DIR, 'SPX_with_BS_Historical.csv'),
        os.path.join(RESULTS_DIR, 'SPX_Clean_Merged.csv'),
    ]
    
    for data_path in possible_paths:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, low_memory=False)
            print(f"   ✓ Full data loaded from {os.path.basename(data_path)}: {len(df):,} options")
            
            # Check for required columns
            required = ['moneyness', 'mid_price', 'bid_ask_spread']
            available = [c for c in required if c in df.columns]
            missing = [c for c in required if c not in df.columns]
            
            if missing:
                print(f"   ⚠️ Missing columns: {missing}")
                # Try to compute missing columns
                if 'mid_price' not in df.columns and 'best_bid' in df.columns and 'best_offer' in df.columns:
                    df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2
                    print(f"   ✓ Computed mid_price")
                if 'bid_ask_spread' not in df.columns and 'best_bid' in df.columns and 'best_offer' in df.columns:
                    df['bid_ask_spread'] = df['best_offer'] - df['best_bid']
                    print(f"   ✓ Computed bid_ask_spread")
            
            # Check for BS price column
            bs_cols = [c for c in df.columns if 'bs_price' in c.lower() or 'bs_hist' in c.lower()]
            if bs_cols:
                print(f"   ✓ BS price columns: {bs_cols}")
            
            print(f"   Available columns: {df.columns.tolist()[:15]}...")
            return df
    
    print(f"   ⚠️ No data file found")
    return None


# ============================================================
# PLOT 1: FOLD-BY-FOLD MAE COMPARISON (Overall)
# ============================================================

def plot_fold_comparison(results):
    """Line chart showing MAE across all folds for each model."""
    print("\n📊 Generating fold-by-fold MAE comparison...")
    
    if not results:
        print("   ⚠️ No results, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model, data in results.items():
        fold_mae = data.get('fold_mae', [])
        if fold_mae and len(fold_mae) > 0:
            years = YEARS[:len(fold_mae)]
            ax.plot(years, fold_mae,
                    marker=MARKERS.get(model, 'o'),
                    color=COLORS.get(model, 'gray'),
                    linewidth=2.5,
                    markersize=10,
                    label=f"{model} (Avg: ${data['avg_mae']:.2f})")
    
    ax.set_xlabel('Test Year', fontsize=12)
    ax.set_ylabel('Mean Absolute Error ($)', fontsize=12)
    ax.set_title('Walk-Forward Validation: MAE by Test Year', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_fold_mae_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, '01_fold_mae_comparison.pdf'), bbox_inches='tight')
    print("   ✅ Saved: 01_fold_mae_comparison.png/pdf")
    plt.close()


# ============================================================
# PLOT 2: ATM COMPARISON (Overall vs ATM)
# ============================================================

def plot_atm_comparison(results):
    """Bar chart and line chart comparing Overall vs ATM performance."""
    print("\n📊 Generating ATM comparison plot...")
    
    if not results:
        print("   ⚠️ No results, skipping")
        return
    
    # Check ATM data availability
    has_atm = any(not np.isnan(d.get('avg_mae_atm', np.nan)) for d in results.values())
    if not has_atm:
        print("   ⚠️ No ATM data, skipping")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = list(results.keys())
    
    # (a) Bar chart: Overall vs ATM
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.35
    
    overall_mae = [results[m]['avg_mae'] for m in models]
    atm_mae = [results[m].get('avg_mae_atm', np.nan) for m in models]
    colors = [COLORS.get(m, 'gray') for m in models]
    
    bars1 = ax.bar(x - width/2, overall_mae, width, label='Overall', 
                   color=colors, edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, atm_mae, width, label='ATM Only',
                   color=colors, edgecolor='black', hatch='//', alpha=0.6)
    
    ax.set_ylabel('Mean Absolute Error ($)', fontsize=12)
    ax.set_title('(a) Overall vs ATM Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' (Historical Vol)', '') for m in models], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Value labels
    for bar, mae in zip(bars1, overall_mae):
        ax.annotate(f'${mae:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    for bar, mae in zip(bars2, atm_mae):
        if not np.isnan(mae):
            ax.annotate(f'${mae:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)
    
    # (b) ATM by year
    ax = axes[1]
    for model, data in results.items():
        fold_atm = data.get('fold_mae_atm', [])
        if fold_atm and len(fold_atm) > 0 and not all(np.isnan(fold_atm)):
            years = YEARS[:len(fold_atm)]
            ax.plot(years, fold_atm,
                    marker=MARKERS.get(model, 'o'),
                    color=COLORS.get(model, 'gray'),
                    linewidth=2.5,
                    markersize=10,
                    label=f"{model} (ATM Avg: ${data.get('avg_mae_atm', 0):.2f})")
    
    ax.set_xlabel('Test Year', fontsize=12)
    ax.set_ylabel('ATM Mean Absolute Error ($)', fontsize=12)
    ax.set_title('(b) ATM Performance by Year', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('At-The-Money (ATM) Options Analysis\n(0.95 ≤ Moneyness ≤ 1.05)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_atm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, '02_atm_comparison.pdf'), bbox_inches='tight')
    print("   ✅ Saved: 02_atm_comparison.png/pdf")
    plt.close()


# ============================================================
# PLOT 5: RMSE COMPARISON
# ============================================================

def plot_rmse_comparison(results):
    """Bar chart and line chart comparing RMSE across models."""
    print("\n📊 Generating RMSE comparison plot...")
    
    if not results:
        print("   ⚠️ No results, skipping")
        return
    
    # Check RMSE availability
    has_rmse = any(not np.isnan(d.get('avg_rmse', np.nan)) for d in results.values())
    if not has_rmse:
        print("   ⚠️ No RMSE data, skipping")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = list(results.keys())
    
    # (a) Bar chart: Average RMSE
    ax = axes[0]
    x = np.arange(len(models))
    
    rmse_values = [results[m].get('avg_rmse', np.nan) for m in models]
    colors = [COLORS.get(m, 'gray') for m in models]
    
    bars = ax.bar(x, rmse_values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, rmse in zip(bars, rmse_values):
        if not np.isnan(rmse):
            ax.annotate(f'${rmse:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # BS baseline
    bs_rmse = results.get('BS (Historical Vol)', {}).get('avg_rmse', np.nan)
    if not np.isnan(bs_rmse):
        ax.axhline(y=bs_rmse, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_ylabel('Root Mean Squared Error ($)', fontsize=12)
    ax.set_title('(a) Average RMSE by Model', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' (Historical Vol)', '') for m in models], rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # (b) RMSE by fold
    ax = axes[1]
    for model, data in results.items():
        fold_rmse = data.get('fold_rmse', [])
        if fold_rmse and len(fold_rmse) > 0:
            years = YEARS[:len(fold_rmse)]
            ax.plot(years, fold_rmse,
                    marker=MARKERS.get(model, 'o'),
                    color=COLORS.get(model, 'gray'),
                    linewidth=2.5,
                    markersize=10,
                    label=f"{model} (Avg: ${data.get('avg_rmse', 0):.2f})")
    
    ax.set_xlabel('Test Year', fontsize=12)
    ax.set_ylabel('Root Mean Squared Error ($)', fontsize=12)
    ax.set_title('(b) RMSE by Test Year', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('RMSE Comparison Across Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_rmse_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, '05_rmse_comparison.pdf'), bbox_inches='tight')
    print("   ✅ Saved: 05_rmse_comparison.png/pdf")
    plt.close()


# ============================================================
# PLOT 6: CALL VS PUT PERFORMANCE (ALL MODELS)
# ============================================================

def plot_call_put_comparison(results = None):
    """Compare Call vs Put MAE across all models (including Black-Scholes)."""
    print("\n📊 Generating Call vs Put MAE comparison...")

    rows = []

    # ------------------------------------------------------------
    # 1) Black-Scholes (Historical Vol) from main dataset
    # ------------------------------------------------------------
    bs_path = os.path.join(RESULTS_DIR, "SPX_with_BS_Historical.csv")
    if os.path.exists(bs_path):
        df_bs = pd.read_csv(bs_path, low_memory=False)

        bs_call_mae = df_bs[df_bs["cp_flag"] == "C"]["abs_error_hist"].mean()
        bs_put_mae  = df_bs[df_bs["cp_flag"] == "P"]["abs_error_hist"].mean()

        rows.append(["Black-Scholes", bs_call_mae, bs_put_mae])
        print(f"   ✓ Loaded BS Call/Put MAE")

    # ------------------------------------------------------------
    # 2) ML models from walk-forward result CSVs
    # ------------------------------------------------------------
    paths = {
        "Neural Network": os.path.join(RESULTS_DIR, "nn_walk_forward_results.csv"),
        "Random Forest":  os.path.join(RESULTS_DIR, "rf_walk_forward_results.csv"),
        "XGBoost":        os.path.join(RESULTS_DIR, "xgb_walk_forward_results.csv"),
    }

    for model, p in paths.items():
        if not os.path.exists(p):
            continue

        df = pd.read_csv(p)
        if {"mae_call", "mae_put"}.issubset(df.columns):
            rows.append([
                model,
                df["mae_call"].mean(),
                df["mae_put"].mean()
            ])
            print(f"   ✓ Loaded {model} Call/Put MAE")

    if len(rows) == 0:
        print("   ⚠️ No Call/Put MAE data found. Skipping.")
        return

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    df_cp = pd.DataFrame(rows, columns=["Model", "MAE Call", "MAE Put"])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df_cp))
    width = 0.35

    b1 = ax.bar(x - width/2, df_cp["MAE Call"], width,
                label="Calls", edgecolor="black")
    b2 = ax.bar(x + width/2, df_cp["MAE Put"],  width,
                label="Puts", edgecolor="black", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(df_cp["Model"], rotation=15)
    ax.set_ylabel("Mean Absolute Error ($)")
    ax.set_title("Performance by Option Type (Calls vs Puts)", fontweight="bold")
    ax.legend()

    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"${bar.get_height():.2f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "15_call_put_comparison.png"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "15_call_put_comparison.pdf"))
    print("✅ Saved: 15_call_put_comparison.png/pdf")
    plt.close()


# ============================================================
# PLOT 7: MODEL COMPARISON BAR CHART
# ============================================================

def plot_model_comparison(results):
    """Bar chart comparing average MAE across all models."""
    print("\n📊 Generating model comparison bar chart...")
    
    if not results:
        print("   ⚠️ No results, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    maes = [results[m]['avg_mae'] for m in models]
    colors = [COLORS.get(m, 'gray') for m in models]
    
    bars = ax.bar(models, maes, color=colors, edgecolor='black', linewidth=1.5)
    
    # Value labels
    for bar, mae in zip(bars, maes):
        ax.annotate(f'${mae:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # BS baseline
    bs_mae = results['BS (Historical Vol)']['avg_mae']
    ax.axhline(y=bs_mae, color='red', linestyle='--', linewidth=2, alpha=0.7, label='BS Baseline')
    
    # Improvement percentages
    for i, (model, mae) in enumerate(zip(models[1:], maes[1:]), 1):
        improvement = (bs_mae - mae) / bs_mae * 100
        ax.annotate(f'+{improvement:.1f}%',
                    xy=(i, mae / 2),
                    ha='center', va='center',
                    fontsize=11, color='white', fontweight='bold')
    
    ax.set_ylabel('Mean Absolute Error ($)', fontsize=12)
    ax.set_title('Model Comparison: Average Test MAE\n(5-Fold Walk-Forward Validation)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(maes) * 1.15)
    ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, '07_model_comparison.pdf'), bbox_inches='tight')
    print("   ✅ Saved: 07_model_comparison.png/pdf")
    plt.close()


# ============================================================
# PLOT 8: FEATURE IMPORTANCE (RF / XGB / NN)
# ============================================================

def plot_feature_importance(rf_importance=None, xgb_importance=None, nn_importance=None, top_k=12):
    """Horizontal bar chart of feature importance."""
    print("\n📊 Generating feature importance plot...")

    rf_importance = rf_importance or {}
    xgb_importance = xgb_importance or {}
    nn_importance = nn_importance or {}

    if not rf_importance and not xgb_importance and not nn_importance:
        print("   ⚠️ No feature importance data, skipping")
        return

    # Combine all features
    all_features = set(rf_importance.keys()) | set(xgb_importance.keys()) | set(nn_importance.keys())

    rows = []
    for f in all_features:
        rf_val = rf_importance.get(f, 0.0)
        xgb_val = xgb_importance.get(f, 0.0)
        nn_val = nn_importance.get(f, 0.0)

        # Average only across models that have the feature
        vals = [v for v in [rf_val, xgb_val, nn_val] if v is not None]
        avg_val = np.mean(vals) if len(vals) else 0.0

        rows.append({"feature": f, "RF": rf_val, "XGB": xgb_val, "NN": nn_val, "avg": avg_val})

    df = pd.DataFrame(rows).sort_values("avg", ascending=True).tail(top_k)

    fig, ax = plt.subplots(figsize=(10, 8))

    y = np.arange(len(df))
    height = 0.25

    # Use only what exists
    has_rf = df["RF"].abs().sum() > 0
    has_xgb = df["XGB"].abs().sum() > 0
    has_nn = df["NN"].abs().sum() > 0

    shift = 0
    if has_rf:
        ax.barh(y - height, df["RF"] * 100, height, label="Random Forest",
                color=COLORS["Random Forest"], edgecolor="black")
    if has_xgb:
        ax.barh(y, df["XGB"] * 100, height, label="XGBoost",
                color=COLORS["XGBoost"], edgecolor="black")
    if has_nn:
        # If you don't have a NN color yet, define COLORS["Neural Network"]
        nn_color = COLORS.get("Neural Network", "gray")
        ax.barh(y + height, df["NN"] * 100, height, label="Neural Network",
                color=nn_color, edgecolor="black")

    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Feature Importance (%)", fontsize=12)
    ax.set_title("Feature Importance: RF vs XGB vs NN", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    if "bid_ask_spread" in df["feature"].values:
        idx = df["feature"].tolist().index("bid_ask_spread")
        ax.axhspan(idx - 0.5, idx + 0.5, alpha=0.2, color="yellow")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(PLOTS_DIR, "08_feature_importance.pdf"), bbox_inches="tight")
    print("   ✅ Saved: 08_feature_importance.png/pdf")
    plt.close()

# ============================================================
# PLOT 9-10: ERROR BY MONEyness REGIME (OTM/ATM/ITM) SEPARATE FOR CALLS & PUTS
# ============================================================

def _infer_pred_cols(df_full):
    """
    Try to infer prediction column names in df_full for each model.
    Adjust these aliases to match your pipeline if needed.
    """
    aliases = {
        "BS (Historical Vol)": ["bs_price_hist", "bs_hist", "bs_price", "bs_historical"],
        "Neural Network": ["nn_pred", "pred_nn", "y_pred_nn", "nn_prediction"],
        "Random Forest": ["rf_pred", "pred_rf", "y_pred_rf", "rf_prediction"],
        "XGBoost": ["xgb_pred", "pred_xgb", "y_pred_xgb", "xgb_prediction"],
    }

    pred_cols = {}
    for model, cols in aliases.items():
        for c in cols:
            if c in df_full.columns:
                pred_cols[model] = c
                break
    return pred_cols


def plot_error_by_moneyness_regime(results, df_full, atm_band=0.05):
    """
    Creates two plots:
    - Calls: MAE by {ITM, ATM, OTM}
    - Puts : MAE by {OTM, ATM, ITM} (reversed definition)
    Uses REAL per-option errors if prediction columns exist in df_full.
    """
    print("\n📊 Generating Call/Put MAE by moneyness regime (OTM/ATM/ITM)...")

    if df_full is None:
        print("   ⚠️ df_full is None, cannot compute regime errors.")
        return

    # Required columns
    if "cp_flag" not in df_full.columns:
        print("   ⚠️ Missing cp_flag in df_full.")
        return

    # Moneyness column
    moneyness_col = None
    for col in ["moneyness", "Moneyness", "K_F", "strike_price_ratio"]:
        if col in df_full.columns:
            moneyness_col = col
            break
    if moneyness_col is None:
        print("   ⚠️ No moneyness column found.")
        return

    # Price column
    price_col = None
    for col in ["mid_price", "midprice", "option_price", "price"]:
        if col in df_full.columns:
            price_col = col
            break
    if price_col is None and "best_bid" in df_full.columns and "best_offer" in df_full.columns:
        df_full["mid_price"] = (df_full["best_bid"] + df_full["best_offer"]) / 2
        price_col = "mid_price"

    if price_col is None:
        print("   ⚠️ No price column found (mid_price).")
        return

    # Prediction columns
    pred_cols = _infer_pred_cols(df_full)
    if "BS (Historical Vol)" not in pred_cols:
        print("   ⚠️ No BS prediction column found in df_full (e.g., bs_price_hist).")
        return

    # We only plot models that we can compute REAL errors for
    models_to_plot = [m for m in results.keys() if m in pred_cols]
    if len(models_to_plot) == 0:
        print("   ⚠️ No models have prediction columns in df_full. Add NN/RF/XGB preds to df_full first.")
        return

    # Clean valid rows
    use_cols = [moneyness_col, price_col, "cp_flag"] + [pred_cols[m] for m in models_to_plot]
    df = df_full[use_cols].dropna()

    # Define regimes
    def regime_call(kf):
        if kf < 1 - atm_band:
            return "ITM"
        elif kf > 1 + atm_band:
            return "OTM"
        return "ATM"

    def regime_put(kf):
        # reversed for puts
        if kf < 1 - atm_band:
            return "OTM"
        elif kf > 1 + atm_band:
            return "ITM"
        return "ATM"

    # --- helper to compute MAE table for a cp type ---
    def compute_table(cp):
        sub = df[df["cp_flag"] == cp].copy()
        if sub.empty:
            return None

        if cp == "C":
            sub["regime"] = sub[moneyness_col].apply(regime_call)
            order = ["ITM", "ATM", "OTM"]
            title = "Calls"
            out_png = "09_calls_moneyness_regimes.png"
            out_pdf = "09_calls_moneyness_regimes.pdf"
        else:
            sub["regime"] = sub[moneyness_col].apply(regime_put)
            order = ["OTM", "ATM", "ITM"]
            title = "Puts"
            out_png = "10_puts_moneyness_regimes.png"
            out_pdf = "10_puts_moneyness_regimes.pdf"

        rows = []
        for model in models_to_plot:
            pred = pred_cols[model]
            sub[f"abs_err_{model}"] = (sub[pred] - sub[price_col]).abs()

            grouped = sub.groupby("regime", observed=True)[f"abs_err_{model}"].mean()
            rows.append({
                "Model": model,
                **{r: float(grouped.get(r, np.nan)) for r in order}
            })

        table = pd.DataFrame(rows)
        return title, order, table, out_png, out_pdf

    # --- plotting ---
    def plot_table(title, order, table, out_png, out_pdf):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(order))
        width = 0.18

        for i, model in enumerate(table["Model"]):
            vals = [table.loc[table["Model"] == model, r].values[0] for r in order]
            ax.bar(x + (i - (len(table)-1)/2)*width, vals, width,
                   label=model, color=COLORS.get(model, "gray"), edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_ylabel("Mean Absolute Error ($)")
        ax.set_title(f"{title}: MAE by Moneyness Regime (ATM band ±{atm_band:.2f})", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, out_png), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(PLOTS_DIR, out_pdf), bbox_inches="tight")
        print(f"   ✅ Saved: {out_png}/{out_pdf}")
        plt.close()

    calls = compute_table("C")
    puts  = compute_table("P")

    if calls:
        plot_table(*calls)
    else:
        print("   ⚠️ No call data found (cp_flag=='C').")

    if puts:
        plot_table(*puts)
    else:
        print("   ⚠️ No put data found (cp_flag=='P').")


# ============================================================
# MAIN EXECUTION
# ============================================================

def generate_all_plots():
    """Generate all visualization plots."""
    
    print("=" * 60)
    print("GENERATING PROJECT VISUALIZATIONS")
    print("=" * 60)
    print(f"\nOutput directory: {PLOTS_DIR}\n")
    
    # Load data
    results = load_results()
    rf_importance, xgb_importance = load_feature_importance()
    df_full = load_full_data()
    
    if not results:
        print("\n❌ Cannot generate plots without results")
        print("   Run the full pipeline first: python main.py")
        return None
    
    print(f"\n✅ Loaded results for {len(results)} models")
    
    # Generate all plots
    plot_fold_comparison(results)           # 01 - Fold MAE by year
    plot_atm_comparison(results)            # 02 - ATM comparison
    plot_rmse_comparison(results)           # 05 - RMSE comparison
    plot_call_put_comparison(results)       # 06 - Call vs Put (all models)
    plot_model_comparison(results)          # 07 - Model comparison bar
    plot_feature_importance(rf_importance, xgb_importance)  # 08 - Feature importance
    plot_error_by_moneyness_regime(results, df_full, atm_band=0.05) 

    print("\n" + "=" * 60)
    print("✅ ALL VISUALIZATIONS GENERATED!")
    print("=" * 60)
    print(f"\n📁 Plots saved to: {PLOTS_DIR}")
    
    # List generated files
    print(f"\n📊 Generated files:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        if f.endswith('.png'):
            print(f"   ✓ {f}")
    
    return PLOTS_DIR


if __name__ == "__main__":
    generate_all_plots()