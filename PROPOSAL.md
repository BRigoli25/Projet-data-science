# Project Proposal: Machine Learning for Option Pricing: A Comparative Study of Neural Networks, Ensemble Methods, and Black-Scholes

---

## Problem Statement & Motivation

Options are financial contracts that give the holder the right (but not the obligation) to buy or sell an underlying asset at a fixed strike price on (European) or before (American) a specified maturity date. Their price depends on several parameters including underlying price, strike, time to maturity, interest rates, and volatility.

Black, Scholes and Merton derived in 1973 a closed-form (theoretical) price for European options under idealized assumptions that are not always met in real markets. BSM assumes for example frictionless markets and no arbitrage, plus the underlying follows geometric Brownian motion with constant volatility.

Since then, extensive research has shown that machine learning can potentially learn complex pricing patterns directly from historical market data without imposing restrictive parametric assumptions. However, it remains an open question whether ML can systematically outperform Black-Scholes when both models use the same volatility information, and which ML architecture (neural networks vs. tree-based methods) performs best on option pricing data.

**This project investigates three key questions:**
1. Can machine learning models outperform Black-Scholes when using historical volatility?
2. Which ML architecture achieves best accuracy when all methods use equivalent training data?
3. What features drive pricing accuracy — traditional inputs (moneyness, maturity, volatility) or market microstructure (bid-ask spreads, liquidity)?

---

## Planned Approach & Technology

This project investigates whether neural networks and tree-based ensemble methods can achieve competitive pricing accuracy compared to Black-Scholes, while potentially capturing market effects beyond constant volatility assumptions.

To this end we compute 1) Black-Scholes with historical volatility, 2) train a neural network with two-pass training, 3) train Random Forest, and 4) train XGBoost, then compare predicted prices to market mid quotes using walk-forward validation.

**Data requirements:** This project focuses on high-liquidity European options on the S&P 500 index (SPX). The data needed:
- S: Forward price (from WRDS OptionMetrics)
- K: Strike price
- T: Time to expiration
- r: Risk-free rate (3-month Treasury from FRED)
- σ: Historical volatility (60-day rolling window)
- Option type: call or put
- Market bid/ask quotes (mid-price as target)
- Liquidity metrics: volume, open interest
- Market microstructure: bid-ask spread

**Data source:** WRDS OptionMetrics provides institutional-grade data for SPX options from March 2018 to August 2025. After applying quality filters (moneyness 0.8-1.2, minimum liquidity thresholds, valid prices), we obtain approximately 4 million high-quality European options from an initial 35 million raw contracts.

**Feature engineering (14 features):** Following Hutchinson, Lo, and Poggio (1994), we construct core features (moneyness, log-moneyness, time to maturity, option type), interaction terms (moneyness × T, log-moneyness × √T, historical vol × √T), and market microstructure features (bid-ask spread, log-volume, log-open-interest). **Critical design choice:** We exclude implied volatility to avoid circularity.

**Validation strategy:** Walk-forward validation with 5 temporal folds (test years 2021-2025), training on all data up to year N and testing on year N+1. This mirrors realistic deployment.

---

### Model 1: Black-Scholes (Baseline)

**Type**: Analytical closed-form solution

**Inputs**: (F, K, T, r, σ, option_type)

**Call option formula** (forward price version):
$$C_{BS} = e^{-rT} \left[ F \cdot \Phi(d_1) - K \cdot \Phi(d_2) \right]$$

where:
$$d_1 = \frac{\ln(F/K) + \frac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

**Put option formula**:
$$P_{BS} = e^{-rT} \left[ K \cdot \Phi(-d_2) - F \cdot \Phi(-d_1) \right]$$

where F is the forward price, σ is 60-day historical volatility, and r is the 3-month Treasury rate.

**Key assumption**: Constant volatility

**Purpose**: 
- Fast baseline with instantaneous pricing (<0.01ms per option)
- Well-understood theoretical properties
- Known to fail for OTM options due to ignoring volatility smile

**Expected performance**: Moderate accuracy (MAE ~$18-22), with systematic underpricing of OTM options

---

### Model 2: Neural Network with Two-Pass Training

**Type**: Supervised learning - feedforward neural network trained on historical market data

**Architecture**:
```
Input layer:    14 features
                ↓
Hidden layer 1: 128 neurons, LeakyReLU, BatchNorm, Dropout(0.3)
                ↓
Hidden layer 2: 64 neurons, LeakyReLU, BatchNorm, Dropout(0.3)
                ↓
Hidden layer 3: 32 neurons, LeakyReLU, BatchNorm, Dropout(0.3)
                ↓
Output layer:   1 neuron (predicted option price)
```

**Key Methodological Innovation: Two-Pass Training**

Traditional neural network training uses 85% of data for training and 15% for validation (early stopping), while tree-based methods use 100% of data. This creates an unfair comparison.

**Solution:**
- **Pass A (Epoch Selection)**: Train on 85% with validation on 15% to find optimal epoch count N*. Discard this model.
- **Pass B (Final Training)**: Reinitialize network and train on 100% of data for exactly N* epochs. Use this model for predictions.

This ensures all models (NN, RF, XGBoost) use 100% of training data, enabling fair comparison.

**What the neural network learns:**
Unlike BS which assumes specific market dynamics, the NN learns directly from market data. It can potentially capture:
- **Volatility risk premium**: Market charges extra for volatility exposure
- **Liquidity effects**: Bid-ask spreads impact pricing
- **Jump risk**: Fat-tailed returns not in BS assumptions
- **Supply/demand imbalances**: Hedging flows, dealer inventory effects

**Training configuration:** AdamW optimizer (lr=0.001), MSE loss, batch size 2048, gradient clipping, L2 regularization

**Purpose**: 
- Test whether deep learning can compete with simpler methods on tabular data
- Achieve faster inference than complex models (target: <1ms per option)
- Fair comparison via two-pass training

**Expected performance**: Best accuracy if two-pass training works (target: MAE ~$12-15, 30-40% improvement over BS)

---

### Model 3: Random Forest

**Type**: Ensemble learning - bootstrap aggregation (bagging)

**Configuration**: 300 trees, max depth 15, min samples split 50, min samples leaf 20, built-in feature importance via Gini impurity

**Key properties:**
- No feature scaling required
- Robust to outliers  
- 100% data utilization (no validation set needed)
- Implicit regularization via bootstrap sampling

**Why Random Forest for options?** Recent research (Grinsztajn et al., 2022) shows tree-based methods often outperform neural networks on tabular data due to their ability to capture irregular decision boundaries and feature interactions naturally.

**Purpose**: 
- Test whether simpler ensemble methods can match deep learning
- Provide interpretable feature importance rankings
- Establish strong baseline for tabular financial data

**Expected performance**: Competitive accuracy (target: MAE ~$12-16) with significantly shorter training time than neural networks

---

### Model 4: XGBoost (Extreme Gradient Boosting)

**Type**: Ensemble learning - gradient boosting decision trees with advanced regularization

**Configuration**: 300 estimators, max depth 6, learning rate 0.1, L1 and L2 regularization, subsample 0.8, column sampling 0.8

**Key differences from Random Forest:**
- Sequential tree building (boosting) vs. parallel (bagging)
- Each tree corrects errors of previous ensemble
- Explicit L1/L2 penalties prevent overfitting
- Typically faster training than RF due to histogram-based optimization

**Purpose**: 
- Compare boosting vs. bagging ensemble methods
- Test state-of-the-art gradient boosting on options
- Evaluate speed-accuracy trade-offs

**Expected performance**: Similar or slightly better accuracy than Random Forest (target: MAE ~$12-16)

---

## Expected Challenges & Mitigation

- **Dataset quality:** Real option data contains noise, stale quotes, and data errors. Rigorous preprocessing following Hutchinson et al. (1994) with filters on moneyness (0.8-1.2), minimum liquidity (volume ≥10, OI ≥100), and outlier removal will ensure high-quality training data.
  
- **Model overfitting:** Neural networks with thousands of parameters can memorize training data. Mitigation: Dropout (0.3), batch normalization, L2 regularization, early stopping via two-pass training, and walk-forward validation to test generalization across time periods.

- **Fair model comparison:** Different models have different data requirements (NN needs validation, trees don't). The two-pass training procedure ensures all models use 100% of available training data, enabling fair comparison.

- **Computational constraints:** Training 5 folds × 3 ML models on millions of options (~135 minutes total). Mitigation: Efficient vectorized implementation, modular design allowing independent model training, use of university computing resources or cloud platforms if needed.

---

## Success Criteria

The project will be considered successful if the following criteria are met:

1. **Predictive accuracy relative to market quotes:**
   - Black-Scholes serves as baseline, with MAE and RMSE in dollars reported on walk-forward test sets.
   - Target performance based on preliminary results:
     - Black-Scholes MAE: $18-22 (baseline)
     - Neural Network MAE: $12-15 (30-40% improvement, target best performance)
     - Random Forest MAE: $12-16 (30-35% improvement)
     - XGBoost MAE: $12-16 (30-35% improvement)

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| P^{\text{model}}_i - P^{\text{market}}_i \right|$$

$$\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} \left( P^{\text{model}}_i - P^{\text{market}}_i \right)^2 }$$

2. **Feature importance analysis:** Extract and interpret feature importance from Random Forest and XGBoost to understand what drives pricing accuracy. Research question: Do traditional factors (volatility, moneyness) or market microstructure (bid-ask spread) matter more?

3. **Computational performance:**
   - Total runtime and average runtime per option reported for each method
   - Clear discussion of accuracy vs. speed trade-offs
   - Target: Neural network inference <1ms per option (100x faster than complex stochastic models)

4. **Reproducibility:** Complete codebase with modular architecture, comprehensive documentation, environment files (requirements.txt, environment.yml), and ability to reproduce all results via `python main.py`.

---

