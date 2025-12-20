"""
SPX OPTIONS DATA PREPROCESSING AND BLACK-SCHOLES BASELINE
=========================================================

Centralizes ALL feature engineering for the project.

Steps:
1. Load and merge options with forward prices
2. Data cleaning and filtering
3. Feature engineering (ALL features created here)
4. Risk-free rates from FRED
5. Historical volatility calculation
6. Black-Scholes baseline pricing
7. Walk-forward validation splits
8. Save processed data
"""
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
import pandas as pd
import numpy as np
from scipy.stats import norm
from pandas_datareader import data as pdr

def run_preprocessing():
    """Main preprocessing function called by main.py"""

    # ==============================================================================
    # STEP 1: DATA LOADING AND INSPECTION
    # ==============================================================================

    print("="*60)
    print("STEP 1: LOADING AND INSPECTING RAW DATA")
    print("="*60)

    # Load sample for quick inspection (first 1000 rows)
    df_forward_sample = pd.read_csv(SPX_FORWARD_FILE, nrows=1000)
    df_options_sample = pd.read_csv(SPX_OPTIONS_FILE, nrows=1000)
    df_forward_sample.to_csv(os.path.join(RESULTS_DIR, 'SPX_Forward_Prices_sample_1000rows.csv'))
    df_options_sample.to_csv(os.path.join(RESULTS_DIR, 'SPX_Options_sample_1000rows.csv'))

    print("\nForward Prices Structure:")
    print(f"Columns: {df_forward_sample.columns.tolist()}")
    print(f"Sample:\n{df_forward_sample.head()}")

    print("\nOptions Structure:")
    print(f"Columns: {df_options_sample.columns.tolist()}")
    print(f"Sample:\n{df_options_sample.head()}")

    # ==============================================================================
    # STEP 2: MERGE OPTIONS WITH FORWARD PRICES
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 2: MERGING OPTIONS DATA WITH FORWARD PRICES")
    print("="*60)

    # Load full datasets
    df_options = pd.read_csv(SPX_OPTIONS_FILE)
    df_forward = pd.read_csv(SPX_FORWARD_FILE)

    print(f"Options loaded: {len(df_options):,} rows")
    print(f"Forward prices loaded: {len(df_forward):,} rows")

    # Rename forward price columns to match options data
    df_forward = df_forward.rename(columns={
        'expiration': 'exdate',
        'ForwardPrice': 'forward_price',
        'AMSettlement': 'am_settlement'
    })

    # Convert all dates to datetime
    df_options['date'] = pd.to_datetime(df_options['date'])
    df_options['exdate'] = pd.to_datetime(df_options['exdate'])
    df_forward['date'] = pd.to_datetime(df_forward['date'])
    df_forward['exdate'] = pd.to_datetime(df_forward['exdate'])

    # Remove duplicates from forward prices
    df_forward_clean = df_forward[['date', 'exdate', 'am_settlement', 'forward_price']].drop_duplicates()

    # Drop forward_price column from options if it exists (will be replaced with merge)
    if 'forward_price' in df_options.columns:
        df_options = df_options.drop(columns=['forward_price'])

    # Merge on date, expiration date, and AM/PM settlement flag
    df_merged = df_options.merge(
        df_forward_clean,
        on=['date', 'exdate', 'am_settlement'],
        how='left'
    )

    print(f"\nAfter merge: {len(df_merged):,} rows")
    print(f"Options with forward price: {df_merged['forward_price'].notna().sum():,}")
    print(f"Match rate: {100*df_merged['forward_price'].notna().sum()/len(df_merged):.1f}%")

    # Keep only options with valid forward prices
    df = df_merged[df_merged['forward_price'].notna()].copy()
    print(f"After removing missing forwards: {len(df):,} rows")

    # Save complete merged data (before filtering)
    df.to_csv(os.path.join(RESULTS_DIR, 'SPX_MERGED_BEFORE_FILTERING.csv'), index=False)
    print(f"✅ Saved raw merged data: SPX_MERGED_BEFORE_FILTERING.csv")

    # ==============================================================================
    # STEP 3: DATA CLEANING AND BASIC CALCULATIONS
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 3: DATA CLEANING AND BASIC CALCULATIONS")
    print("="*60)

    # Fix strike prices (WRDS stores in pennies, we need dollars)
    df['strike_price'] = df['strike_price'] / 1000

    # Calculate moneyness (K/F ratio)
    df['moneyness'] = df['strike_price'] / df['forward_price']

    # Calculate time to maturity in years
    df['T'] = (df['exdate'] - df['date']).dt.days / 365.25

    # Calculate mid-price (average of bid and ask)
    df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2

    # ==============================================================================
    # STEP 4: FEATURE ENGINEERING (CENTRALIZED - ALL FEATURES HERE)
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 4: FEATURE ENGINEERING (Centralized)")
    print("="*60)

    # Log transformations for moneyness
    df['log_moneyness'] = np.log(df['moneyness'])
    
    # Time transformations
    df['sqrt_T'] = np.sqrt(df['T'])
    df['log_T'] = np.log(df['T'] + 1e-10)
    
    # Binary indicator for call/put
    df['is_call'] = (df['cp_flag'] == 'C').astype(int)
    
    # Normalized forward price (relative to mean)
    df['forward_price_norm'] = df['forward_price'] / df['forward_price'].mean()
    
    # Interaction terms (following Hutchinson et al. methodology)
    df['moneyness_T'] = df['moneyness'] * df['T']
    df['log_moneyness_sqrt_T'] = df['log_moneyness'] * df['sqrt_T']
    
    # Liquidity features (log-transformed for stability)
    df['log_volume'] = np.log(df['volume'] + 1)
    df['log_open_interest'] = np.log(df['open_interest'] + 1)
    
    # Market microstructure feature
    df['bid_ask_spread'] = df['best_offer'] - df['best_bid']
    
    # Moneyness bucket for analysis
    df['money_bucket'] = pd.cut(
        df['moneyness'], 
        bins=[0, 0.95, 1.05, 3.0], 
        labels=['OTM', 'ATM', 'ITM']
    )

    print("✅ Created features:")
    print("   - log_moneyness, sqrt_T, log_T")
    print("   - is_call, forward_price_norm")
    print("   - moneyness_T, log_moneyness_sqrt_T")
    print("   - log_volume, log_open_interest, bid_ask_spread")
    print("   - money_bucket (OTM/ATM/ITM)")

    # ==============================================================================
    # STEP 5: DATA FILTERING
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 5: DATA FILTERING")
    print("="*60)

    print(f"Before filtering: {len(df):,} rows")

    df = df[
        (df['moneyness'] >= 0.8) &
        (df['moneyness'] <= 1.2) &
        (df['volume'] >= 10) &
        (df['open_interest'] >= 100) &
        (df['impl_volatility'] >= 0.05) &
        (df['impl_volatility'] <= 1.0) &
        (df['T'] > 0.005) &
        (df['T'] <= 2.0) &
        (df['best_bid'] > 0) &
        (df['best_offer'] > df['best_bid']) &
        (df['exercise_style'] == 'E') 
    ].copy()

    print(f"After filtering: {len(df):,} rows")

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

    # Save filtered data
    df.to_csv(SPX_MERGED_FILE, index=False)
    print(f"\n✅ Saved: SPX_Clean_Merged.csv ({len(df):,} rows)")



    # ==============================================================================
    # STEP 6: DOWNLOAD RISK-FREE RATES (3-MONTH TREASURY BILLS)
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 6: DOWNLOADING RISK-FREE RATES FROM FRED")
    print("="*60)

    # Check if treasury rates file already exists
    if os.path.exists(TREASURY_RATES_FILE):
        print(f"✅ Loading existing treasury rates from: {TREASURY_RATES_FILE}")
        treasury_rates = pd.read_csv(TREASURY_RATES_FILE, index_col=0, parse_dates=True)
    else:
        print("⚠️  Treasury rates file not found. Attempting to download from FRED...")
        try:
            # Download from FRED (DTB3 = 3-Month Treasury Bill Rate)
            treasury_rates = pdr.DataReader(
                TREASURY_CONFIG['symbol'], 
                TREASURY_CONFIG['source'], 
                TREASURY_CONFIG['start_date'], 
                TREASURY_CONFIG['end_date']
            )
            treasury_rates = treasury_rates.rename(columns={'DTB3': 'rate'})
            treasury_rates['rate'] = treasury_rates['rate'] / 100  # Convert percentage to decimal
            
            # Forward fill to create daily rates (weekends/holidays)
            treasury_rates = treasury_rates.resample('D').ffill()
            
            # Save for future use
            treasury_rates.to_csv(TREASURY_RATES_FILE)
            
            print(f"✅ Treasury rates downloaded and saved to: {TREASURY_RATES_FILE}")
            print(f"Date range: {treasury_rates.index.min().date()} to {treasury_rates.index.max().date()}")
            
        except Exception as e:
            print(f"⚠️ Error downloading from FRED: {e}")
            print(f"⚠️ Using fallback rate: {TREASURY_CONFIG['fallback_rate']:.2%}")
            
            # Create synthetic treasury rates using fallback
            date_range = pd.date_range(
                start=TREASURY_CONFIG['start_date'], 
                end=TREASURY_CONFIG['end_date'], 
                freq='D'
            )
            treasury_rates = pd.DataFrame(
                {'rate': TREASURY_CONFIG['fallback_rate']}, 
                index=date_range
            )
            treasury_rates.to_csv(TREASURY_RATES_FILE)
            print(f"⚠️ Created fallback treasury rates file: {TREASURY_RATES_FILE}")

    def get_treasury_rate(date):
        """Get 3-month Treasury bill rate for a given date"""
        date = pd.to_datetime(date)
        try:
            return treasury_rates.loc[date, 'rate']
        except KeyError:
            # If exact date not found, use most recent prior rate
            prior_dates = treasury_rates[treasury_rates.index <= date]
            if len(prior_dates) > 0:
                return prior_dates.iloc[-1]['rate']
            return TREASURY_CONFIG['fallback_rate']
    # Apply risk-free rates to each option
    df['r'] = df['date'].apply(get_treasury_rate)

    print("\n✅ Risk-free rates applied to data")
    print(f"Risk-free rate statistics:")
    print(df['r'].describe())
    
        
        
    # ==============================================================================
    # STEP 7: CALCULATE HISTORICAL VOLATILITY (HUTCHINSON METHODOLOGY)
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 7: CALCULATING HISTORICAL VOLATILITY")
    print("="*60)

    """
    Following Hutchinson et al. (1994), we calculate historical volatility as:
    - 60-day rolling standard deviation of log returns
    - Annualized by multiplying by sqrt(252), (number of trading day in a year)
    - Calculated on the forward price (underlying index)
    """

    # Get daily forward prices (one per date)
    spx_daily = df.groupby('date')['forward_price'].mean().sort_index()
    spx_daily = pd.DataFrame({'price': spx_daily})

    # Calculate log returns
    spx_daily['log_return'] = np.log(spx_daily['price'] / spx_daily['price'].shift(1))

    # Calculate 60-day rolling standard deviation
    spx_daily['std_60d'] = spx_daily['log_return'].rolling(window=60).std()

    # Annualize (multiply by sqrt(252 trading days))
    spx_daily['historical_vol'] = spx_daily['std_60d'] * np.sqrt(252)

    print(f"Historical volatility calculated for {spx_daily['historical_vol'].notna().sum()} days")
    print(f"\nHistorical volatility statistics:")
    print(spx_daily['historical_vol'].describe())

    # Merge historical volatility back to options data
    df = df.merge(
        spx_daily[['historical_vol']].reset_index(),
        on='date',
        how='left'
    )

    # Count options with valid historical volatility
    df_hist = df[df['historical_vol'].notna()].copy()
    print(f"\nOptions with historical volatility: {len(df_hist):,} ({100*len(df_hist)/len(df):.1f}%)")
    df_hist['historical_vol_sqrt_T'] = df_hist['historical_vol'] * df_hist['sqrt_T']
    print(f"✅ Added historical_vol_sqrt_T interaction feature")

    # ==============================================================================
    # STEP 8: BLACK-SCHOLES OPTION PRICING FORMULA
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 8: IMPLEMENTING BLACK-SCHOLES FORMULA")
    print("="*60)

    def black_scholes(S, K, T, r, sigma, option_type='C'):
        """
        Black-Scholes European option pricing formula.
        
        Parameters:
        -----------
        S : float or array
            Spot/Forward price (underlying asset price)
        K : float or array
            Strike price
        T : float or array
            Time to maturity in years
        r : float or array
            Risk-free rate (annualized, decimal form)
        sigma : float or array
            Volatility (annualized, decimal form)
        option_type : str or array
            'C' for call, 'P' for put
            
        Returns:
        --------
        price : float or array
            Option premium (theoretical price)
            
        Notes:
        ------
        Formula:
        - Call: C = S*N(d1) - K*exp(-rT)*N(d2)
        - Put:  P = K*exp(-rT)*N(-d2) - S*N(-d1)
        
        Where:
        - d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        - d2 = d1 - σ√T
        - N(x) = cumulative standard normal distribution
        """
        # Avoid division by zero for very short maturities
        T = np.maximum(T, 1e-10)
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate call and put prices using cumulative normal distribution
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        # Select call or put based on option_type
        is_call = (option_type == 'C')
        price = np.where(is_call, call_price, put_price)
        
        return price
    
    # Version 1: Black-Scholes with IMPLIED VOLATILITY (Sanity check)
    print("\nCalculating BS with implied volatility...")
    df_hist['bs_price'] = black_scholes(
        S=df_hist['forward_price'].values,
        K=df_hist['strike_price'].values,
        T=df_hist['T'].values,
        r=df_hist['r'].values,
        sigma=df_hist['impl_volatility'].values,
        option_type=df_hist['cp_flag'].values
    )

    # Calculate errors for BS(IV)
    df_hist['bs_error'] = df_hist['bs_price'] - df_hist['mid_price']
    df_hist['abs_error'] = df_hist['bs_error'].abs()

    # Version 2: Black-Scholes with HISTORICAL VOLATILITY (Hutchinson style) 
    # More representative as calculating with implied vol is by definition what 
    # makes BSM and market prices match

    print("Calculating BS with historical volatility...")
    df_hist['bs_price_hist'] = black_scholes(
        S=df_hist['forward_price'].values,
        K=df_hist['strike_price'].values,
        T=df_hist['T'].values,
        r=df_hist['r'].values,
        sigma=df_hist['historical_vol'].values,
        option_type=df_hist['cp_flag'].values
    )

    # Calculate errors for BS(HV)
    df_hist['bs_error_hist'] = df_hist['bs_price_hist'] - df_hist['mid_price']
    df_hist['abs_error_hist'] = df_hist['bs_error_hist'].abs()

    # ==============================================================================
    # STEP 9: PERFORMANCE EVALUATION
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 9: BLACK-SCHOLES PERFORMANCE EVALUATION")
    print("="*60)

    #print("\n📊 BLACK-SCHOLES WITH IMPLIED VOLATILITY:")
    print(f"  Options: {len(df_hist):,}")
    print(f"  MAE:  ${df_hist['abs_error'].mean():.2f}")
    print(f"  RMSE: ${np.sqrt((df_hist['bs_error']**2).mean()):.2f}")
    print(f"  Bias: ${df_hist['bs_error'].mean():.2f}")


    print("\n📊 BLACK-SCHOLES WITH HISTORICAL VOLATILITY (Baseline):")
    print(f"  Options: {len(df_hist):,}")
    print(f"  MAE:  ${df_hist['abs_error_hist'].mean():.2f}")
    print(f"  RMSE: ${np.sqrt((df_hist['bs_error_hist']**2).mean()):.2f}")
    print(f"  Bias: ${df_hist['bs_error_hist'].mean():.2f}")


    print("\n  By Option Type:")
    for opt in ['C', 'P']:
        subset = df_hist[df_hist['cp_flag'] == opt]
        print(f"    {opt}: MAE=${subset['abs_error_hist'].mean():.2f}, n={len(subset):,}")

    print("\n  By Year:")
    yearly = df_hist.groupby(df_hist['date'].dt.year)['abs_error'].agg(['mean', 'count'])
    print(yearly)


    # ==============================================================================
    # STEP 10: WALK-FORWARD VALIDATION SPLITS
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 10: WALK-FORWARD VALIDATION ANALYSIS")
    print("="*60)

    # Sort by date for temporal split
    df_hist = df_hist.sort_values('date').reset_index(drop=True)

    print(f"Total options (with historical vol): {len(df_hist):,}")
    print(f"Date range: {df_hist['date'].min().date()} to {df_hist['date'].max().date()}")

    # define 5 fold configurations, one for each year
    fold_configs = [
    {'name': 'Fold 1', 'train_end': '2020-12-31', 'test_end': '2021-12-31'},
    {'name': 'Fold 2', 'train_end': '2021-12-31', 'test_end': '2022-12-31'},
    {'name': 'Fold 3', 'train_end': '2022-12-31', 'test_end': '2023-12-31'},
    {'name': 'Fold 4', 'train_end': '2023-12-31', 'test_end': '2024-12-31'},
    {'name': 'Fold 5', 'train_end': '2024-12-31', 'test_end': '2025-08-29'},
    ]

    # Calculate BS performance for each fold
    fold_results = []

    for fold in fold_configs:
        fold_name = fold['name']
        #train_mask = df_hist['date'] <= fold['train_end']
        test_mask = (df_hist['date'] > fold['train_end']) & (df_hist['date'] <= fold['test_end'])
        df_test_fold = df_hist[test_mask].copy()
        
        if len(df_test_fold) == 0:
            print(f"\n⚠️  {fold_name}: No test data, skipping")
            continue
        
        # Calculate metrics
        mae = df_test_fold['abs_error_hist'].mean()
        rmse = np.sqrt((df_test_fold['bs_error_hist']**2).mean())
        bias = df_test_fold['bs_error_hist'].mean()
        
        # ATM subset
        atm_mask = (df_test_fold['moneyness'] >= 0.95) & (df_test_fold['moneyness'] <= 1.05)
        df_test_atm = df_test_fold[atm_mask]
        mae_atm = df_test_atm['abs_error_hist'].mean() if len(df_test_atm) > 0 else np.nan
        
        fold_results.append({
            'fold': fold_name,
            'train_end': fold['train_end'],
            'test_end': fold['test_end'],
            'n_test': len(df_test_fold),
            'n_atm': len(df_test_atm),
            'mae': mae,
            'mae_atm': mae_atm,
            'rmse': rmse,
            'bias': bias,
        })
        
        print(f"\n{fold_name}:")
        #print(f"  Train: 2018-03-29 to {fold['train_end']}")
        print(f"  Test:  {pd.to_datetime(fold['train_end']) + pd.Timedelta(days=1)} to {fold['test_end']}")
        print(f"  Test size: {len(df_test_fold):,} options")
        print(f"  BS(HV) MAE: ${mae:.2f}")
        print(f"  BS(HV) MAE (ATM): ${mae_atm:.2f}")

    # Calculate average performance
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_mae_atm = np.mean([r['mae_atm'] for r in fold_results if not np.isnan(r['mae_atm'])])

    print("\n" + "="*60)
    print("WALK-FORWARD AVERAGE PERFORMANCE")
    print("="*60)
    print(f"Number of folds: {len(fold_results)}")
    print(f"Average BS(HV) MAE: ${avg_mae:.2f}")
    print(f"Average BS(HV) MAE (ATM only): ${avg_mae_atm:.2f}")

    print("\nFold-by-Fold Summary:")
    print(f"{'Fold':<10} {'Test Year':<12} {'MAE':<10} {'MAE (ATM)':<12}")
    print("-" * 32)
    for r in fold_results:
        test_year = r['test_end'][:4]
        print(f"{r['fold']:<10} {test_year:<12} ${r['mae']:<9.2f} ${r['mae_atm']:<11.2f}")
    print("-" * 32)
    print(f"{'Average':<10} {'All':<12} ${avg_mae:<9.2f}")
    
    """
    # Save fold info for NN to use
    fold_info = {
        'fold_configs': fold_configs,
        'fold_results': fold_results,
        'avg_mae': avg_mae,
        'avg_mae_atm': avg_mae_atm,
    }

    # Keep single split for backward compatibility (use last fold)
    train_mask = df_hist['date'] <= '2024-12-31'
    val_mask = (df_hist['date'] > '2023-12-31') & (df_hist['date'] <= '2024-12-31')
    test_mask = df_hist['date'] > '2024-12-31'
    """


    # ==============================================================================
    # STEP 11: SAVE FINAL DATASETS
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 11: SAVING PROCESSED DATA")
    print("="*60)

    # Save subset with historical volatility (main file for ML models)
    df_hist.to_csv(SPX_BS_HIST_FILE, index=False)
    print(f"✅ Saved: SPX_with_BS_Historical.csv ({len(df_hist):,} rows)")

    print(f"✅ Saved: {SPX_BS_BOTH_FILE}")
    print(f"✅ Saved: {SPX_BS_HIST_FILE}")

   # Verify all features are present
    print("\n📊 Features available for ML models:")
    for feat in FEATURES_BASIC:
        if feat in df_hist.columns:
            print(f"   ✓ {feat}")
        else:
            print(f"   ✗ {feat} (MISSING!)")


    print("\n" + "="*60)
    print("✅ DATA PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Total options processed: {len(df_hist):,}")
    print(f"  - Date range: {df_hist['date'].min().date()} to {df_hist['date'].max().date()}")
    print(f"\nWalk-Forward Validation (5 folds):")
    print(f"  - Number of folds: {len(fold_results)}")
    print(f"  - Average BS(HV) MAE: ${avg_mae:.2f}")
    print(f"  - Average BS(HV) MAE (ATM): ${avg_mae_atm:.2f}")
    print(f"\nBaseline to beat: ${avg_mae:.2f} MAE")
    print(f"\nReady for neural network training!")


if __name__ == "__main__":
    run_preprocessing()
