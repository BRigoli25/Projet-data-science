# Project Proposal: Comparing Numerical and Analytical Methods for Option Pricing: Monte Carlo, Neural Networks, and Black–Scholes

---

## Problem Statement & Motivation

Options are financial contracts that give you the right (not the obligation) to buy or sell an underlying asset at a fixed strike price on (European) or up to (American) a specified maturity. Their price depends on several parameters such as underlying price, strike, time to maturity, interest rates, and volatility. Black, Scholes and Merton derived in 1973 a closed-form (theoretical) price for European options under idealized assumptions that are not always met in real markets. BSM assumes for example frictionless markets and no arbitrage, plus the underlying follows geometric Brownian motion with constant volatility and applies only to European options.

Since then, many papers have extended or relaxed these assumptions introducing, for example, stochastic volatility (Steven Heston, 1993) or jump-diffusion models (Robert C. Merton 1976) and numerical methods have also been developed to price options when no closed-form solution exists or when we do not wish to rely on all of the BSM assumptions, often leading to a better fit to observed market prices.

---

## Planned Approach & Technology


This project aims to compute options prices via 1) Black-Scholes, 2) Monte Carlo simulation, 3) train a neural network, and to compare predicted prices to market mid quotes. Then evaluate pricing accuracy using MAE/RMSE in dollars, as well as computational cost (total runtime).

Note on data requirements: NNs are data-intensive and require substantial historical quotes for reliable training => they are less suitable for thinly traded or newly listed derivatives.

This is why this project will focus on high-liquidity European options on the S&P 500 index (SPX). The data we will need will be:
    - S: Underlying price
    - K: Strike price
    - T: Time to expiration
    - r: Risk-free rate
    - σ: volatility of the underlying
    - Option type: call or put

S, K, T, option type and market bid/ask quotes are available from Yahoo Finance via the "yfinance" Python open-source library with mid quotes as empirical option prices. The remaining parameters, r and σ, will be estimated: r from external risk-free rate data (e.g. Treasury yields) and σ following Hutchinson, Lo, and Poggio (1994).
   
   
---

## Expected Challenges & Mitigation

- **Dataset quality (preprocessing ):** Real option data can contain noise, missing values, and occasional arbitrage violations. Rigorous preprocessing will need to be applied.
 
- **Model overfitting:** Overfitting occurs when a model performs well on training data but poorly on unseen data which happens often when training a model. To face this, techniques such as cross-validation, regularization or early stopping can help to improve generalization.

---

## Success Criteria

The project will be considered successful if the following criteria are met:


1. **Predictive accuracy relative to market quotes:**
   - Black–Scholes serves as a baseline, with MAE and RMSE in dollars (for call and put prices) reported on an out-of-sample test set.
   - Monte Carlo and the neural network achieve comparable or lower MAE/RMSE than Black–Scholes on market mid quotes.


2. **Computational performance:**
   - Total runtime and average runtime per option are reported for each method.
   - There is a clear discussion of the trade-off between accuracy and computational cost (Black–Scholes vs. Monte Carlo vs. neural network inference).


---

## Stretch Idea: Learning the Greeks & Hedging Performance
As a stretch objective (only if time remains/no problem before), the project will estimate and analyze the Greeks (sensitivities of the option price with respect to market variables such as Delta, Gamma, Vega, Theta, Rho) and assess whether the trained models can reproduce these sensitivities accurately enough to support hedging strategies.
