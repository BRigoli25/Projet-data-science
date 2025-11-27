"""
SPX OPTIONS DATA PREPROCESSING AND BLACK-SCHOLES BASELINE
Adapted for modular project structure
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

    # Save samples for reference
    df_forward_sample.to_csv('SPX_Forward_Prices_sample_1000rows.csv')
    df_options_sample.to_csv('SPX_Options_sample_1000rows.csv')

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

    df.to_csv('SPX_MERGED_TO_USE.csv') #We save here the complete data available

    # ==============================================================================
    # STEP 3: DATA CLEANING AND FEATURE ENGINEERING
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 3: DATA CLEANING AND FEATURE ENGINEERING")
    print("="*60)

    # Fix strike prices (WRDS stores in pennies, we need dollars)
    df['strike_price'] = df['strike_price'] / 1000

    # Calculate moneyness (S/K ratio) used in many papers for efficiency
    df['moneyness'] = df['strike_price'] / df['forward_price']

    # Calculate time to maturity in years
    df['T'] = (df['exdate'] - df['date']).dt.days / 365.25

    # Calculate mid-price (average of bid and ask)
    df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2


    # Filter for quality (flexible to have more or less)
    print("\nFiltering for quality data...")
    df = df[
        (df['moneyness'] >= 0.80) &
        (df['moneyness'] <= 1.20) &
        (df['volume'] >= 10) &
        (df['open_interest'] >= 100) &
        (df['impl_volatility'] >= 0.05) &
        (df['impl_volatility'] <= 1.0) &
        (df['T'] > 0.001) &
        (df['best_bid'] > 0) &
        (df['best_offer'] > df['best_bid']) &
        (df['exercise_style'] == 'E')   #should already be the case
    ].copy()

    print(f"After filtering: {len(df):,} rows")

    print(f"\nData ranges:")
    print(f"  Forward price: ${df['forward_price'].min():.0f} - ${df['forward_price'].max():.0f}")
    print(f"  Strike price: ${df['strike_price'].min():.0f} - ${df['strike_price'].max():.0f}")
    print(f"  Moneyness: {df['moneyness'].min():.3f} - {df['moneyness'].max():.3f}")
    print(f"  Time to maturity: {df['T'].min():.3f} - {df['T'].max():.3f} years")

    # Save merged data
    df.to_csv(SPX_MERGED_FILE, index=False)
    print(f"\n✅ Saved: SPX_Clean_Merged.csv ({len(df):,} rows)")

    # ==============================================================================
    # STEP 4: DOWNLOAD RISK-FREE RATES (3-MONTH TREASURY BILLS)
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 4: DOWNLOADING RISK-FREE RATES FROM FRED")
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
    # STEP 5: CALCULATE HISTORICAL VOLATILITY (HUTCHINSON METHODOLOGY)
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 5: CALCULATING HISTORICAL VOLATILITY")
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

    # ==============================================================================
    # STEP 6: BLACK-SCHOLES OPTION PRICING FORMULA
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 6: IMPLEMENTING BLACK-SCHOLES FORMULA")
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

    # ==============================================================================
    # STEP 7: CALCULATE BLACK-SCHOLES PRICES (TWO VERSIONS)
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 7: CALCULATING BLACK-SCHOLES BASELINE PRICES")
    print("="*60)

    # Version 1: Black-Scholes with IMPLIED VOLATILITY (from market)
    print("\nCalculating BS with implied volatility...")
    df['bs_price'] = black_scholes(
        S=df['forward_price'].values,
        K=df['strike_price'].values,
        T=df['T'].values,
        r=df['r'].values,
        sigma=df['impl_volatility'].values,
        option_type=df['cp_flag'].values
    )

    # Calculate errors for BS(IV)
    df['bs_error'] = df['bs_price'] - df['mid_price']
    df['abs_error'] = df['bs_error'].abs()
    df['pct_error'] = (df['bs_error'] / df['mid_price']) * 100

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
    df_hist['pct_error_hist'] = (df_hist['bs_error_hist'] / df_hist['mid_price']) * 100

    # ==============================================================================
    # STEP 8: PERFORMANCE EVALUATION
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 8: BLACK-SCHOLES PERFORMANCE EVALUATION")
    print("="*60)

    print("\n📊 BLACK-SCHOLES WITH IMPLIED VOLATILITY:")
    print(f"  Options: {len(df):,}")
    print(f"  MAE:  ${df['abs_error'].mean():.2f}")
    print(f"  RMSE: ${np.sqrt((df['bs_error']**2).mean()):.2f}")
    #print(f"  MAPE: {df['pct_error'].abs().mean():.2f}%")
    print(f"  Bias: ${df['bs_error'].mean():.2f}")

    print("\n  By Option Type:")
    for opt in ['C', 'P']:
        subset = df[df['cp_flag'] == opt]
        print(f"    {opt}: MAE=${subset['abs_error'].mean():.2f}, n={len(subset):,}")

    print("\n  By Year:")
    yearly = df.groupby(df['date'].dt.year)['abs_error'].agg(['mean', 'count'])
    print(yearly)

    print("\n  By Moneyness:")
    df['money_bucket'] = pd.cut(df['moneyness'], bins=[0, 0.9, 1.1, 3.0], labels=['OTM', 'ATM', 'ITM'])
    print(df.groupby('money_bucket', observed=True)['abs_error'].agg(['mean', 'count']))

    print("\n📊 BLACK-SCHOLES WITH HISTORICAL VOLATILITY (Hutchinson 1994):")
    print(f"  Options: {len(df_hist):,}")
    print(f"  MAE:  ${df_hist['abs_error_hist'].mean():.2f}")
    print(f"  RMSE: ${np.sqrt((df_hist['bs_error_hist']**2).mean()):.2f}")
    #print(f"  MAPE: {df_hist['pct_error_hist'].abs().mean():.2f}%")
    print(f"  Bias: ${df_hist['bs_error_hist'].mean():.2f}")

    print("\n  By Option Type:")
    for opt in ['C', 'P']:
        subset = df_hist[df_hist['cp_flag'] == opt]
        print(f"    {opt}: MAE=${subset['abs_error_hist'].mean():.2f}, n={len(subset):,}")

    # ==============================================================================
    # STEP 9: SAVE FINAL DATASETS
    # ==============================================================================

    print("\n" + "="*60)
    print("STEP 9: SAVING PROCESSED DATA")
    print("="*60)

    # Save full dataset with both BS versions
    df.to_csv(SPX_BS_BOTH_FILE, index=False)
    print(f"✅ Saved: SPX_with_BS_Both_Vols.csv ({len(df):,} rows)")

    # Save subset with historical volatility
    df_hist.to_csv(SPX_BS_HIST_FILE, index=False)
    print(f"✅ Saved: SPX_with_BS_Historical.csv ({len(df_hist):,} rows)")

    print(f"✅ Saved: {SPX_BS_BOTH_FILE}")
    print(f"✅ Saved: {SPX_BS_HIST_FILE}")

    print("\n" + "="*60)
    print("✅ DATA PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Total options processed: {len(df):,}")
    print(f"  - Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  - BS(IV) MAE: ${df['abs_error'].mean():.2f}")
    print(f"  - BS(HV) MAE: ${df_hist['abs_error_hist'].mean():.2f}")
    print(f"\nReady for neural network training!")
    print(f"Target: Beat BS(HV) = ${df_hist['abs_error_hist'].mean():.2f} MAE")

if __name__ == "__main__":
    run_preprocessing()