"""
SPX OPTIONS DATA PREPROCESSING AND BLACK-SCHOLES BASELINE
=========================================================

Centralizes ALL feature engineering for the project.

Steps:
1. Load and merge options with forward prices
2. Basic calculations (moneyness, T, mid_price)
3. Risk-free rates from FRED
4. Historical volatility calculation (60-day rolling)
5. Feature engineering (ALL features created here)
6. Data filtering (includes r and vol sanity checks)
7. Black-Scholes baseline pricing
8. Walk-forward validation splits
9. Save processed data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from scipy.stats import norm
from pandas_datareader import data as pdr

# ============================================================
# CONFIGURATION
# ============================================================

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Main directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Input data files
SPX_FORWARD_FILE = os.path.join(DATA_DIR, 'SPX_Forward_Prices_Complete_2018-2025.csv')
SPX_OPTIONS_FILE = os.path.join(DATA_DIR, 'SPX_Options_raw_2018-2025.csv')
TREASURY_RATES_FILE = os.path.join(DATA_DIR, 'treasury_3month_rates.csv')

# Output files
SPX_MERGED_FILE = os.path.join(RESULTS_DIR, 'SPX_MERGED_TO_USE.csv')
SPX_CLEAN_FILE = os.path.join(RESULTS_DIR, 'SPX_Clean_Merged.csv')
SPX_BS_BOTH_FILE = os.path.join(RESULTS_DIR, 'SPX_with_BS_Both_Vols.csv')
SPX_BS_HIST_FILE = os.path.join(RESULTS_DIR, 'SPX_with_BS_Historical.csv')
SPX_FEATURES_FILE = os.path.join(RESULTS_DIR, 'SPX_features.csv')

# Feature sets 
FEATURES_BASIC = [
    'moneyness',
    'log_moneyness',
    'T',
    'log_T',
    'sqrt_T',
    'is_call',
    'forward_price_norm',
    'moneyness_T',
    'log_moneyness_sqrt_T',
    'log_volume',
    'log_open_interest',
    'bid_ask_spread',
    'historical_vol',
    'historical_vol_sqrt_T'
]

# Treasury rates configuration
TREASURY_CONFIG = {
    'symbol': 'DTB3',
    'source': 'fred',
    'start_date': '2017-12-01',
    'end_date': '2025-12-31',
    'fallback_rate': 0.04,
}

# Black-Scholes configuration
BS_CONFIG = {
    'historical_vol_window': 60,
    'trading_days_per_year': 252,
}

FILTER_CONFIG = {
    'min_moneyness': 0.80,
    'max_moneyness': 1.20,
    'min_days_to_expiry': 2,
    'max_days_to_expiry': 730,
    'min_volume': 10,
    'min_open_interest': 100,
    'min_implied_vol': 0.05,
    'max_implied_vol': 1.00,
}

print(f"✅ Configuration loaded")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Data directory: {DATA_DIR}")
print(f"   Results directory: {RESULTS_DIR}")
def load_treasury_rates():
    """Load or download risk-free rates from FRED."""
    
    if os.path.exists(TREASURY_RATES_FILE):
        print(f"✅ Loading existing treasury rates from: {TREASURY_RATES_FILE}")
        treasury_rates = pd.read_csv(TREASURY_RATES_FILE, index_col=0, parse_dates=True)
    else:
        print("⚠️  Treasury rates file not found. Downloading from FRED...")
        try:
            treasury_rates = pdr.DataReader(
                TREASURY_CONFIG['symbol'], 
                TREASURY_CONFIG['source'], 
                TREASURY_CONFIG['start_date'], 
                TREASURY_CONFIG['end_date']
            )
            treasury_rates = treasury_rates.rename(columns={'DTB3': 'rate'})
            treasury_rates['rate'] = treasury_rates['rate'] / 100
            treasury_rates = treasury_rates.resample('D').ffill()
            treasury_rates.to_csv(TREASURY_RATES_FILE)
            print(f"✅ Treasury rates downloaded and saved")
        except Exception as e:
            print(f"⚠️ Error downloading from FRED: {e}")
            print(f"⚠️ Using fallback rate: {TREASURY_CONFIG['fallback_rate']:.2%}")
            date_range = pd.date_range(
                start=TREASURY_CONFIG['start_date'], 
                end=TREASURY_CONFIG['end_date'], 
                freq='D'
            )
            treasury_rates = pd.DataFrame({'rate': TREASURY_CONFIG['fallback_rate']}, index=date_range)
            treasury_rates.to_csv(TREASURY_RATES_FILE)
    
    return treasury_rates


def get_treasury_rate(date, treasury_rates):
    """Get 3-month Treasury bill rate for a given date."""
    date = pd.to_datetime(date)
    try:
        return treasury_rates.loc[date, 'rate']
    except KeyError:
        prior_dates = treasury_rates[treasury_rates.index <= date]
        if len(prior_dates) > 0:
            return prior_dates.iloc[-1]['rate']
        return TREASURY_CONFIG['fallback_rate']


def calculate_historical_volatility(df_forward_clean):
    """
    Calculate 60-day rolling historical volatility.
    Following Hutchinson et al. (1994) methodology.
    """
    TARGET_DAYS = 30
    
    tmp_fwd = df_forward_clean.copy()
    tmp_fwd['ttm_days'] = (tmp_fwd['exdate'] - tmp_fwd['date']).dt.days
    tmp_fwd = tmp_fwd[(tmp_fwd['ttm_days'] > 0) & (tmp_fwd['forward_price'].notna())].copy()
    tmp_fwd['dist'] = (tmp_fwd['ttm_days'] - TARGET_DAYS).abs()
    
    # Pick the closest-to-target forward per date
    fwd_nearest = tmp_fwd.sort_values(['date', 'dist']).groupby('date').head(1)
    
    spx_daily = fwd_nearest.set_index('date')['forward_price'].sort_index().to_frame('price')
    spx_daily['log_return'] = np.log(spx_daily['price'] / spx_daily['price'].shift(1))
    spx_daily['std_60d'] = spx_daily['log_return'].rolling(window=60, min_periods=60).std()
    spx_daily['historical_vol'] = spx_daily['std_60d'] * np.sqrt(252)
    
    return spx_daily[['historical_vol']]


def black_scholes(F, K, T, r, sigma, option_type='C'):
    """
    Black-Scholes European option pricing formula (forward version).
    
    Parameters:
    -----------
    F : Forward price
    K : Strike price
    T : Time to maturity in years
    r : Risk-free rate
    sigma : Volatility
    option_type : 'C' for call, 'P' for put
    """
    T = np.maximum(T, 1e-10)
    
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    disc = np.exp(-r * T)
    call_price = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put_price = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    
    is_call = (option_type == 'C')
    price = np.where(is_call, call_price, put_price)
    
    return price


def run_preprocessing():
    """Main preprocessing function."""

    # ==========================================================================
    # STEP 1: LOAD AND MERGE DATA
    # ==========================================================================
    
    print("=" * 60)
    print("STEP 1: LOADING AND MERGING DATA")
    print("=" * 60)
    
    # Load datasets
    df_options = pd.read_csv(SPX_OPTIONS_FILE)
    df_forward = pd.read_csv(SPX_FORWARD_FILE)
    
    print(f"Options loaded: {len(df_options):,} rows")
    print(f"Forward prices loaded: {len(df_forward):,} rows")
    
    # Rename forward price columns
    df_forward = df_forward.rename(columns={
        'expiration': 'exdate',
        'ForwardPrice': 'forward_price',
        'AMSettlement': 'am_settlement'
    })
    
    # Convert dates
    df_options['date'] = pd.to_datetime(df_options['date'])
    df_options['exdate'] = pd.to_datetime(df_options['exdate'])
    df_forward['date'] = pd.to_datetime(df_forward['date'])
    df_forward['exdate'] = pd.to_datetime(df_forward['exdate'])
    
    # Clean forward prices
    df_forward_clean = df_forward[['date', 'exdate', 'am_settlement', 'forward_price']].drop_duplicates()
    
    # Drop existing forward_price if present
    if 'forward_price' in df_options.columns:
        df_options = df_options.drop(columns=['forward_price'])
    
    # Merge
    df = df_options.merge(
        df_forward_clean,
        on=['date', 'exdate', 'am_settlement'],
        how='left'
    )
    
    print(f"\nAfter merge: {len(df):,} rows")
    print(f"Options with forward price: {df['forward_price'].notna().sum():,}")
    print(f"Match rate: {100 * df['forward_price'].notna().sum() / len(df):.1f}%")
    
    # Keep only options with valid forward prices
    df = df[df['forward_price'].notna()].copy()
    print(f"After removing missing forwards: {len(df):,} rows")
    
    # Save raw merged (for reference)
    df.to_csv(os.path.join(RESULTS_DIR, 'SPX_MERGED_BEFORE_FILTERING.csv'), index=False)
    print(f"✅ Saved: SPX_MERGED_BEFORE_FILTERING.csv")

    # ==========================================================================
    # STEP 2: BASIC CALCULATIONS
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 2: BASIC CALCULATIONS")
    print("=" * 60)
    
    # Fix strike prices (WRDS stores in pennies)
    df['strike_price'] = df['strike_price'] / 1000
    
    # Calculate moneyness (K/F)
    df['moneyness'] = df['strike_price'] / df['forward_price']
    
    # Calculate time to maturity in years
    df['T'] = (df['exdate'] - df['date']).dt.days / 365.25
    
    # Calculate mid-price
    df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2
    
    print(f"✅ Calculated: moneyness, T, mid_price")

    # ==========================================================================
    # STEP 3: RISK-FREE RATES
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 3: APPLYING RISK-FREE RATES")
    print("=" * 60)
    
    treasury_rates = load_treasury_rates()
    df['r'] = df['date'].apply(lambda d: get_treasury_rate(d, treasury_rates))
    
    print(f"\nRisk-free rate statistics:")
    print(df['r'].describe())

    # ==========================================================================
    # STEP 4: HISTORICAL VOLATILITY
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 4: CALCULATING HISTORICAL VOLATILITY")
    print("=" * 60)
    
    spx_daily = calculate_historical_volatility(df_forward_clean)
    
    print(f"Historical volatility calculated for {spx_daily['historical_vol'].notna().sum()} days")
    print(f"\nHistorical volatility statistics:")
    print(spx_daily['historical_vol'].describe())
    
    # Merge historical vol to options
    df = df.merge(
        spx_daily.reset_index(),
        on='date',
        how='left'
    )
    
    print(f"\nOptions with historical volatility: {df['historical_vol'].notna().sum():,} "
          f"({100 * df['historical_vol'].notna().sum() / len(df):.1f}%)")

    # ==========================================================================
    # STEP 5: FEATURE ENGINEERING
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 5: FEATURE ENGINEERING")
    print("=" * 60)
    
    # Log transformations
    df['log_moneyness'] = np.log(df['moneyness'])
    
    # Time transformations
    df['sqrt_T'] = np.sqrt(df['T'])
    df['log_T'] = np.log(df['T'] + 1e-10)
    
    # Binary indicator for call/put
    df['is_call'] = (df['cp_flag'] == 'C').astype(int)
    
    # Normalized forward price
    df['forward_price_norm'] = df['forward_price'] / df['forward_price'].mean()
    
    # Interaction terms (Hutchinson et al. methodology)
    df['moneyness_T'] = df['moneyness'] * df['T']
    df['log_moneyness_sqrt_T'] = df['log_moneyness'] * df['sqrt_T']
    
    # Volatility interaction (only where vol exists)
    df['historical_vol_sqrt_T'] = df['historical_vol'] * df['sqrt_T']
    
    # Liquidity features
    df['log_volume'] = np.log(df['volume'] + 1)
    df['log_open_interest'] = np.log(df['open_interest'] + 1)
    
    # Market microstructure
    df['bid_ask_spread'] = df['best_offer'] - df['best_bid']
    
    # Moneyness bucket for analysis
    df['money_bucket'] = pd.cut(
        df['moneyness'], 
        bins=[0, 0.95, 1.05, 3.0], 
        labels=['Low M', 'ATM', 'High M']
    )
    
    print("✅ Created features:")
    print("   - log_moneyness, sqrt_T, log_T")
    print("   - is_call, forward_price_norm")
    print("   - moneyness_T, log_moneyness_sqrt_T")
    print("   - historical_vol_sqrt_T")
    print("   - log_volume, log_open_interest, bid_ask_spread")
    print("   - money_bucket (OTM/ATM/ITM)")

    # ==========================================================================
    # STEP 6: DATA FILTERING
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 6: DATA FILTERING")
    print("=" * 60)
    
    print(f"Before filtering: {len(df):,} rows")
    
    df = df[
        # Moneyness filter
        (df['moneyness'] >= 0.8) &
        (df['moneyness'] <= 1.2) &
        # Liquidity filters
        (df['volume'] >= 10) &
        (df['open_interest'] >= 100) &
        # Implied volatility filter
        (df['impl_volatility'] >= 0.05) &
        (df['impl_volatility'] <= 1.0) &
        # Time to maturity filter
        (df['T'] > 0.005) &
        (df['T'] <= 2.0) &
        # Bid-ask sanity
        (df['best_bid'] > 0) &
        (df['best_offer'] > df['best_bid']) &
        # European options only
        (df['exercise_style'] == 'E') &
        # Risk-free rate sanity
        (df['r'] >= -0.01) &
        (df['r'] <= 0.15) &
        # Historical volatility exists and is reasonable
        (df['historical_vol'].notna()) &
        (df['historical_vol'] >= 0.05) &
        (df['historical_vol'] <= 1.5)
    ].copy()
    
    print(f"After filtering: {len(df):,} rows")
    
    # Remove price outliers
    price_low = df['mid_price'].quantile(0.01)
    price_high = df['mid_price'].quantile(0.99)
    df = df[(df['mid_price'] >= price_low) & (df['mid_price'] <= price_high)].copy()
    print(f"After outlier removal: {len(df):,} rows")
    
    print(f"\nData ranges:")
    print(f"  Forward price: ${df['forward_price'].min():.0f} - ${df['forward_price'].max():.0f}")
    print(f"  Strike price: ${df['strike_price'].min():.0f} - ${df['strike_price'].max():.0f}")
    print(f"  Moneyness: {df['moneyness'].min():.3f} - {df['moneyness'].max():.3f}")
    print(f"  Time to maturity: {df['T'].min():.3f} - {df['T'].max():.3f} years")
    print(f"  Mid Price: ${df['mid_price'].min():.2f} - ${df['mid_price'].max():.2f}")
    print(f"  Implied Vol: {df['impl_volatility'].min()*100:.1f}% - {df['impl_volatility'].max()*100:.1f}%")
    print(f"  Historical Vol: {df['historical_vol'].min()*100:.1f}% - {df['historical_vol'].max()*100:.1f}%")
    print(f"  Risk-free Rate: {df['r'].min()*100:.2f}% - {df['r'].max()*100:.2f}%")

    # ==========================================================================
    # STEP 7: BLACK-SCHOLES PRICING
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 7: BLACK-SCHOLES PRICING")
    print("=" * 60)
    
    print("Calculating BS with historical volatility...")
    df['bs_price_hist'] = black_scholes(
        F=df['forward_price'].values,
        K=df['strike_price'].values,
        T=df['T'].values,
        r=df['r'].values,
        sigma=df['historical_vol'].values,
        option_type=df['cp_flag'].values
    )
    
    # Calculate errors
    df['bs_error_hist'] = df['bs_price_hist'] - df['mid_price']
    df['abs_error_hist'] = df['bs_error_hist'].abs()
    
    print(f"\n📊 BLACK-SCHOLES WITH HISTORICAL VOLATILITY:")
    print(f"  Options: {len(df):,}")
    print(f"  MAE:  ${df['abs_error_hist'].mean():.2f}")
    print(f"  RMSE: ${np.sqrt((df['bs_error_hist']**2).mean()):.2f}")
    print(f"  Bias: ${df['bs_error_hist'].mean():.2f}")
    
    print("\n  By Option Type:")
    for opt in ['C', 'P']:
        subset = df[df['cp_flag'] == opt]
        print(f"    {opt}: MAE=${subset['abs_error_hist'].mean():.2f}, n={len(subset):,}")

    # ==========================================================================
    # STEP 8: WALK-FORWARD VALIDATION
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 8: WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Total options: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    fold_configs = [
        {'name': 'Fold 1', 'train_end': '2020-12-31', 'test_end': '2021-12-31'},
        {'name': 'Fold 2', 'train_end': '2021-12-31', 'test_end': '2022-12-31'},
        {'name': 'Fold 3', 'train_end': '2022-12-31', 'test_end': '2023-12-31'},
        {'name': 'Fold 4', 'train_end': '2023-12-31', 'test_end': '2024-12-31'},
        {'name': 'Fold 5', 'train_end': '2024-12-31', 'test_end': '2025-08-29'},
    ]
    
    fold_results = []
    
    for fold in fold_configs:
        fold_name = fold['name']
        test_mask = (df['date'] > fold['train_end']) & (df['date'] <= fold['test_end'])
        df_test = df[test_mask].copy()
        
        if len(df_test) == 0:
            print(f"\n⚠️ {fold_name}: No test data, skipping")
            continue
        
        mae = df_test['abs_error_hist'].mean()
        rmse = np.sqrt((df_test['bs_error_hist']**2).mean())
        bias = df_test['bs_error_hist'].mean()
        
        # ATM subset
        atm_mask = (df_test['moneyness'] >= 0.95) & (df_test['moneyness'] <= 1.05)
        df_test_atm = df_test[atm_mask]
        mae_atm = df_test_atm['abs_error_hist'].mean() if len(df_test_atm) > 0 else np.nan
        
        fold_results.append({
            'fold': fold_name,
            'train_end': fold['train_end'],
            'test_end': fold['test_end'],
            'n_test': len(df_test),
            'n_atm': len(df_test_atm),
            'mae': mae,
            'mae_atm': mae_atm,
            'rmse': rmse,
            'bias': bias,
            'mae_call': df_test[df_test['cp_flag'] == 'C']['abs_error_hist'].mean() if (df_test['cp_flag'] == 'C').sum() > 0 else np.nan,
            'mae_put': df_test[df_test['cp_flag'] == 'P']['abs_error_hist'].mean() if (df_test['cp_flag'] == 'P').sum() > 0 else np.nan,
            'bias_call': df_test[df_test['cp_flag'] == 'C']['bs_error_hist'].mean() if (df_test['cp_flag'] == 'C').sum() > 0 else np.nan,
            'bias_put': df_test[df_test['cp_flag'] == 'P']['bs_error_hist'].mean() if (df_test['cp_flag'] == 'P').sum() > 0 else np.nan,

        })
        
        print(f"\n{fold_name}:")
        print(f"  Test: {pd.to_datetime(fold['train_end']) + pd.Timedelta(days=1)} to {fold['test_end']}")
        print(f"  Test size: {len(df_test):,} options")
        print(f"  BS(HV) MAE: ${mae:.2f}")
        print(f"  BS(HV) MAE (ATM): ${mae_atm:.2f}")
    
    # Summary
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_mae_atm = np.mean([r['mae_atm'] for r in fold_results if not np.isnan(r['mae_atm'])])
    
    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)
    print(f"Average BS(HV) MAE: ${avg_mae:.2f}")
    print(f"Average BS(HV) MAE (ATM): ${avg_mae_atm:.2f}")
    
    print(f"\n{'Fold':<10} {'Year':<8} {'MAE':<10} {'MAE (ATM)':<12}")
    print("-" * 40)
    for r in fold_results:
        print(f"{r['fold']:<10} {r['test_end'][:4]:<8} ${r['mae']:<9.2f} ${r['mae_atm']:<11.2f}")
    print("-" * 40)
    print(f"{'Average':<10} {'All':<8} ${avg_mae:<9.2f} ${avg_mae_atm:<11.2f}")
    
    # Save BS results
    bs_results_df = pd.DataFrame(fold_results)
    bs_results_df.to_csv(os.path.join(RESULTS_DIR, 'bs_walk_forward_results.csv'), index=False)
    print(f"\n✅ Saved: bs_walk_forward_results.csv")

    # ==========================================================================
    # STEP 9: SAVE PROCESSED DATA
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("STEP 9: SAVING PROCESSED DATA")
    print("=" * 60)
    
    df.to_csv(SPX_BS_HIST_FILE, index=False)
    print(f"✅ Saved: {SPX_BS_HIST_FILE} ({len(df):,} rows)")
    
    # Verify features
    print("\n📊 Features available for ML models:")
    for feat in FEATURES_BASIC:
        if feat in df.columns:
            print(f"   ✓ {feat}")
        else:
            print(f"   ✗ {feat} (MISSING!)")
    
    print("\n" + "=" * 60)
    print("✅ DATA PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - Total options: {len(df):,}")
    print(f"  - Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  - BS(HV) Baseline MAE: ${avg_mae:.2f}")
    print(f"\nReady for ML training!")


if __name__ == "__main__":
    run_preprocessing()