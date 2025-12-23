# Machine Learning for Option Pricing: A Comparative Study of Neural Networks, Ensemble Methods, and Black-Scholes

**Advanced Programming 2025 — Data Science Project**

A comprehensive empirical comparison of machine learning approaches against Black-Scholes for pricing S&P 500 index (SPX) European options.

---

## 📊 Key Results

| Model | Avg MAE | vs Black-Scholes |
|-------|---------|------------------|
| **Black-Scholes (Historical Vol)** | $19.56 | Baseline |
| **Neural Network (Two-Pass)** | $12.09 | **+38.2%** improvement |
| **Random Forest** | $12.87 | +34.2% improvement |
| **XGBoost** | $13.05 | +33.3% improvement |

*Results from 5-fold walk-forward validation on 3.96 million SPX options (2018-2025)*

---

## 🎯 Research Questions

This study investigates 4 questions:

1. **Can machine learning outperform Black-Scholes with historical volatility?**
2. **Which model achieves best accuracy when all methods access equivalent training data?**
3. **What features drive pricing accuracy?**
4. **What are the training-inference trade-offs across model architectures, and which is most suitable for real-time applications?**



---

## 📈 Key Findings

- **Neural networks with two-pass training achieve best performance** at $12.09 MAE (38% improvement over Black-Scholes)
- **Machine learning improvements are not uniform**: Gains are largest for out-of-the-money options and during volatile regimes (2025), while at-the-money options show smaller improvements (23% vs 38% overall)
- **Bid-ask spreads dominate feature importance** at 34-36%, exceeding traditional inputs like volatility by an order of magnitude — market microstructure matters more than volatility modeling
- **Neural networks offer favorable training-inference trade-offs**: despite requiring 113 minutes to train, they achieve 5.3 million options/second inference (4-8x faster than tree-based methods)

---

## 🔬 Methodology

### Data
- **Source**: WRDS OptionMetrics
- **Asset**: SPX (S&P 500 Index) European options
- **Period**: March 2018 – August 2025
- **Size**: 3,963,262 options after filtering (from 35.3M raw)

### Data Filtering
- Moneyness: 0.80 ≤ K/F ≤ 1.20
- Liquidity: Volume ≥ 10, Open Interest ≥ 100
- Implied Volatility: 5% ≤ σ_IV ≤ 100%
- Time to Maturity: 2 days to 2 years
- European exercise style only
- Price outliers removed (1st-99th percentile)

### Walk-Forward Validation

| Fold | Training Period | Test Year | Train Size | Test Size |
|------|-----------------|-----------|------------|-----------|
| 1 | 2018-03 to 2020-12 | 2021 | 1,111,775 | 475,979 |
| 2 | 2018-03 to 2021-12 | 2022 | 1,587,754 | 587,048 |
| 3 | 2018-03 to 2022-12 | 2023 | 2,174,802 | 678,757 |
| 4 | 2018-03 to 2023-12 | 2024 | 2,853,559 | 672,342 |
| 5 | 2018-03 to 2024-12 | 2025 | 3,525,901 | 437,361 |

### Models

| Model | Type | Description |
|-------|------|-------------|
| **Black-Scholes** | Analytical | Baseline using 60-day historical volatility |
| **Neural Network** | Deep Learning | 3-layer MLP (128-64-32) with two-pass training |
| **Random Forest** | Ensemble | 300 trees, max depth 15 |
| **XGBoost** | Gradient Boosting | 300 estimators, L2 regularization |

### Two-Pass Training (Neural Network)

A key methodological contribution addressing data utilization asymmetry:

1. **Pass A (Epoch Selection)**: Train on 85% with validation-based early stopping to find optimal epoch count N*
2. **Pass B (Final Training)**: Reinitialize and train on 100% of data for exactly N* epochs

This enables fair comparison: all models now use 100% of training data.

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Conda (Anaconda or Miniconda)
- WRDS OptionMetrics access (for raw data)

### Setup

```bash
# Clone repository
git clone https://github.com/BRigoli25/Projet-data-science.git
cd Projet-data-science

# Create conda environment
conda env create -f environment.yml
conda activate pricing-option-env

# Or use pip
pip install -r requirements.txt
```

### Data Setup

Place WRDS OptionMetrics files in `data/raw/`:
1. **SPX_Options_raw_2018-2025.csv** — Option quotes
2. **SPX_Forward_Prices_Complete_2018-2025.csv** — Forward prices

---

## 🚀 Usage

### Run Complete Pipeline

```bash
python main.py
```

Pipeline executes (~135 min total):
1. Data preprocessing & Black-Scholes baseline (9 min)
2. Neural Network with two-pass training (113 min)
3. Random Forest training (10 min)
4. XGBoost training (2 min)
5. Visualization generation (1 min)

### Run Individual Steps

```bash
python main.py --preprocess   # Data preprocessing only
python main.py --models       # Train all ML models
python main.py --step nn      # Neural Network only
python main.py --step rf      # Random Forest only
python main.py --step xgb     # XGBoost only
python main.py --viz          # Generate visualizations
python main.py --runtime      # Runtime comparison
```

---

## 📁 Project Structure

```
Projet-data-science/
├── main.py                      # Pipeline orchestrator
├── environment.yml              # Conda environment
├── requirements.txt             # pip dependencies
├── README.md                    # This file
│
├── src/
│   ├── config.py               # Centralized configuration
│   ├── data_preprocessing.py   # Data processing & BS baseline
│   ├── neural_network.py       # Neural network with two-pass training
│   ├── random_forest.py        # Random Forest model
│   ├── xg_boost.py             # XGBoost model
│   └── visualizations.py       # Publication-ready plots
│
├── data/
│   └── raw/                    # Raw WRDS data (not in git)
│
├── results/                    # Output CSV files & plots
│   └── plots/                  # Generated figures
│
└── models/                     # Saved trained models
```

---

## 📊 Features (14 total)

| Feature | Description |
|---------|-------------|
| `moneyness` | K/F (strike / forward price) |
| `log_moneyness` | ln(K/F) |
| `T` | Time to maturity (years) |
| `log_T`, `sqrt_T` | Time transformations |
| `is_call` | 1 for call, 0 for put |
| `forward_price_norm` | Normalized forward price |
| `moneyness_T` | Moneyness × T interaction |
| `log_moneyness_sqrt_T` | ln(K/F) × √T interaction |
| `log_volume` | ln(volume + 1) |
| `log_open_interest` | ln(open interest + 1) |
| `bid_ask_spread` | Ask - Bid |
| `historical_vol` | 60-day rolling volatility |
| `historical_vol_sqrt_T` | Historical vol × √T |

**Note**: Implied volatility is intentionally excluded to avoid circularity.

---

## 📈 Feature Importance

| Feature | Random Forest | XGBoost |
|---------|---------------|---------|
| bid_ask_spread | 34.14% | 36.09% |
| is_call | 9.37% | 15.37% |
| moneyness_T | 7.84% | 8.53% |
| log_moneyness | 7.48% | 5.40% |
| historical_vol | 1.73% | 0.79% |

Bid-ask spread dominates, suggesting market microstructure effects are more critical for pricing accuracy than volatility estimation.

---

## ⚡ Computational Performance

| Model | Training Time | Inference (s) | Options/sec |
|-------|---------------|---------------|-------------|
| Black-Scholes | — | 0.027 | 16,142,431 |
| Neural Network | 113 min | 0.083 | 5,279,921 |
| XGBoost | 2 min | 0.339 | 1,290,513 |
| Random Forest | 10 min | 0.712 | 614,730 |

---

## 📚 References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Hutchinson, J. M., Lo, A. W., & Poggio, T. (1994). A nonparametric approach to pricing and hedging derivative securities via learning networks. *Journal of Finance*, 49(3), 851-889.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. 22nd ACM SIGKDD*, 785-794.
- Grinsztajn, L., et al. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS*.

---

## 📝 License

This project is for academic purposes (Advanced Programming 2025).

---

## 🤖 AI Tools Disclosure

This project utilized Claude (Anthropic) for code debugging, methodology discussion, and writing assistance. All code implementations, experimental design decisions, and interpretations are the author's own work.
