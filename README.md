# Project Proposal: Machine Learning vs. Stochastic Models for SPX Option Pricing: Comparing Neural Networks, Heston, and Black-Scholes

---

## Problem Statement & Motivation

Options are financial contracts that give the holder the right (but not the obligation) to buy or sell an underlying asset at a fixed strike price on (European) or before (American) a specified maturity date. Their price depends on several parameters including underlying price, strike, time to maturity, interest rates, and volatility.
Black, Scholes and Merton derived in 1973 a closed-form (theoretical) price for European options under idealized assumptions that are not always met in real markets. BSM assumes for example frictionless markets and no arbitrage, plus the underlying follows geometric Brownian motion with constant volatility. 

Since then, extensive research has extended or relaxed these assumptions. Notable examples include:
- **Stochastic volatility models** (Heston, 1993): Allow volatility to vary randomly over time
- **Jump-diffusion models** (Merton, 1976): Incorporate sudden price jumps

These advanced models often provide better fits to observed market prices but require numerical methods (e.g., Monte Carlo simulation) for pricing, which are computationally expensive.

Meanwhile, machine learning has emerged as a data-driven alternative that can potentially learn complex pricing patterns directly from historical market data without imposing restrictive parametric assumptions. However, it remains an open question whether ML can match or exceed the accuracy of sophisticated stochastic models while maintaining computational efficiency.

**This project investigates three key questions:**
1. Where does Black-Scholes fail on real SPX options?
2. Does Heston's stochastic volatility improve accuracy?
3. Can neural networks match Heston's accuracy while being 100× faster?


---


## Planned Approach & Technology

This project investigates whether neural networks can achieve competitive pricing accuracy compared to both classical (Black-Scholes) and advanced (Heston stochastic volatility) models, while offering computational advantages for real-time applications. 
In this end we compute 1) Black-Scholes, 2) Monte Carlo simulation, 3) train a neural network, and to compare predicted prices to market mid quotes. 

Note on data requirements: NNs are data-intensive and require substantial historical quotes for reliable training => they are less suitable for thinly traded or newly listed derivatives.

This is why this project will focus on high-liquidity European options on the S&P 500 index (SPX). The data we will need will be: 
    - S: Underlying price
    - K: Strike price
    - T: Time to expiration
    - r: Risk-free rate
    - σ: volatility of the underlying
    - Option type: call or put

S, K, T, option type and market bid/ask quotes are available from Yahoo Finance via the "yfinance" Python open-source library or/and any other available data source, with mid quotes as empirical option prices. The remaining parameters, r and σ, will be estimated: r from external risk-free rate data (e.g. Treasury yields) and σ (constant for BSM) following Hutchinson, Lo, and Poggio (1994) or refer to other litterature.


### Model 1: Black-Scholes (Baseline)

**Type**: Analytical closed-form solution

**Inputs**: (S, K, T, r, σ, option_type)

**Call option formula**:
$$C_{BS} = S \cdot \Phi(d_1) - K \cdot e^{-rT} \cdot \Phi(d_2)$$

where:
$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

**Put option formula** (via put-call parity or direct calculation):
$$P_{BS} = K \cdot e^{-rT} \cdot \Phi(-d_2) - S \cdot \Phi(-d_1)$$

**Key assumption**: Constant volatility

**Purpose**: 
- Fast baseline with instantaneous pricing (<0.01ms per option)
- Well-understood theoretical properties
- Known to fail for OTM options due to ignoring volatility smile

**Expected performance**: Moderate accuracy, with systematic underpricing of OTM puts


### Model 2: Heston Stochastic Volatility (Monte Carlo)

**Type**: Numerical simulation under stochastic volatility dynamics

**Model dynamics**:

Stock price process:
$$dS_t = \mu S_t \, dt + \sqrt{v_t} \, S_t \, dW_t^S$$

Variance process:
$$dv_t = \kappa(\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW_t^v$$

where:
- $v_t$ = instantaneous variance (volatility squared)
- $\kappa$ = mean reversion speed
- $\theta$ = long-term variance mean
- $\xi$ = volatility of volatility (vol-of-vol)
- $\rho = \text{Corr}(W_t^S, W_t^v)$ = correlation between stock and volatility shocks

**Key feature**: For equity indices like SPX, ρ < 0 (typically -0.6 to -0.8), creating the **leverage effect**:
- When stock price drops, volatility increases
- This generates the volatility skew: OTM puts have higher implied volatility than OTM calls

**Parameter calibration**:
- **Frequency**: Monthly (to adapt to changing market regimes)
- **Objective**: Minimize sum of squared pricing errors on a calibration set:
  $$\min_{\Theta} \sum_{i=1}^{N} w_i \left( P_i^{\text{market}} - P_i^{\text{Heston}}(\Theta) \right)^2$$
  
  where $\Theta = (\kappa, \theta, \xi, \rho, v_0)$ and $w_i$ are weights (e.g., inverse bid-ask spread)

- **Calibration set**: ~50-100 liquid options spanning multiple maturities and strikes
- **Optimization method**: Differential Evolution or Levenberg-Marquardt

**Monte Carlo pricing**:
- **Number of paths**: 100,000 per option
- **Time discretization**: 252 steps (daily simulation)
- **Numerical scheme**: Euler-Maruyama with **full truncation** (Andersen, 2008) to ensure $v_t \geq 0$:
  $$v_{t+\Delta t} = \max(0, v_t + \kappa(\theta - v_t)\Delta t + \xi\sqrt{\max(v_t, 0)} \, \sqrt{\Delta t} \, Z_v)$$
  
  where $Z_v$ is a standard normal random variable correlated with the stock process.

- **Variance reduction**: Antithetic variates

**Implementation**: Use QuantLib (well-tested library) to avoid numerical errors

**Purpose**: 
- State-of-the-art benchmark representing sophisticated quantitative finance
- Captures volatility smile/skew observed in real markets
- Computationally expensive (~100-500ms per option)

**Expected performance**: Best theoretical model accuracy, especially for OTM options

---

### Model 3: Neural Network (Machine Learning)

**Type**: Supervised learning - feedforward neural network trained on historical market data

**Architecture (Arbitrary)**:
```
Input layer:    6 features
                ↓
Hidden layer 1: 64 neurons, ReLU activation, Dropout(0.2)
                ↓
Hidden layer 2: 32 neurons, ReLU activation, Dropout(0.2)
                ↓
Hidden layer 3: 16 neurons, ReLU activation
                ↓
Output layer:   1 neuron (predicted option price)
```

**Input features**:
1. Underlying price (S)
2. Strike price (K)
3. Time to maturity (T, in years)
4. Risk-free rate (r)
5. Historical volatility (σ_hist)
6. Option type (encoded as 0=put, 1=call)

**Feature engineering** (to improve training):
- Log-moneyness: ln(S/K)
- Normalized time: √T (volatility scales with square root of time)
- Standardization: Scale all features to mean=0, std=1


**What the neural network learns**:
Unlike BS and Heston, which are based on theoretical models, the NN learns directly from market data. It can potentially capture:
- **Volatility risk premium**: Market charges extra for volatility exposure
- **Liquidity effects**: Less liquid options trade at discounts/premiums
- **Supply/demand imbalances**: Heavy hedging flows, market maker inventory effects
- **Behavioral biases**: Crash fear premium embedded in OTM put prices

**Purpose**: 
- Evaluate whether data-driven methods can compete with sophisticated theoretical models
- Achieve faster inference than Heston MC (target: <1ms per option)
- Potentially capture market effects beyond stochastic volatility

**Expected performance**: Accuracy between BS and Heston, but with significant speed advantage over Heston
    
---

## Expected Challenges & Mitigation

- **Dataset quality (preprocessing):** Real option data can contain noise or missing values. Rigorous preprocessing will need to be applied. ML needs a lot of data for training and historiacal option data is expensive and free sources may be limited.
  
- **Model overfitting:** Overfitting occurs when a model performs well on training data but poorly on unseen data which happens often when training a model. To face this, techniques such as cross-validation, regularization or early stopping can help to improve generalization.

- **Heston Calibration** Heston has 5 parameters and calibration can be unstable or converge to local minima, leading to poor out-of-sample pricing. As it is complex implementation some useful resources can be found on QuantLib library. For calibration probably using same as academic paper values.
---

## Success Criteria

The project will be considered successful if the following criteria are met:


1. **Predictive accuracy relative to market quotes:**
   - Black–Scholes serves as a baseline, with MAE and RMSE in dollars (for call and put prices) reported on an out-of-sample test set.
   - Baseline expectations based on academic literature (Bakshi et al., 1997):
Black-Scholes MAE: 8-15 dollars per option (baseline; higher for OTM options),
Heston MAE: 30-40% lower than BS (target: $5-10 per option), 
Neural Network MAE: Between BS and Heston (target: within 10% of Heston)

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| P^{\text{model}}_i - P^{\text{mkt}}_i \right|
$$

$$
\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} \left( P^{\text{model}}_i - P^{\text{mkt}}_i \right)^2 }
$$



2. **Computational performance:**
   - Total runtime and average runtime per option are reported for each method.
   - There is a clear discussion of the trade-off between accuracy and computational cost (Black–Scholes vs. Monte Carlo vs. neural network inference).


---

## Stretch Idea: Learning the Greeks & Hedging Performance 
As a stretch objective (only if time remains/no problem before), the project will estimate and analyze the Greeks (sensitivities of the option price with respect to market variables such as Delta, Gamma, Vega, Theta, Rho) and assess whether the trained models can reproduce these sensitivities accurately enough to support hedging strategies.

## Repository Structure
```
option-pricing-thesis/
├── README.md                    # This file
├── main.py                      # Main entry point
├── environment.yml              # Conda environment
├── src/
│   ├── config.py               # Configuration & paths
│   ├── data_preprocessing.py   # Black-Scholes baseline
│   └── neural_network.py       # NN models
├── data/
│   └── raw/                    # Raw CSV files (user-provided)
├── results/                    # Output CSV files & plots
├── models/                     # Saved trained models
└── notebooks/                  # Optional exploration
```

## Prerequisites

- **Python 3.9+**
- **Conda** (Anaconda or Miniconda)
- **Data Access**: WRDS OptionMetrics subscription required