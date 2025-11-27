# Machine Learning for SPX Option Pricing

## ⚠️ Important: Data Files Required

Raw data files are **NOT included** in this repository due to size (6.9 GB total).

**Required files** (must be placed in `data/raw/` before running):
- `SPX_Forward_Prices_Complete_2018-2025.csv`
- `SPX_Options_raw_2018-2025.csv`


**Source**: WRDS OptionMetrics (institutional access required)

**For reviewers/graders**: Contact bastian.rigoli@unil.ch for data access.
**Option pricing Project** - Comparing Black-Scholes and Machine Learning (Neural Network, Random Forest, another model) for S&P 500 index option pricing using WRDS OptionMetrics data (2018-2025).

## Author

## Project Overview

This thesis investigates whether neural networks can outperform traditional option pricing models (Black-Scholes, Heston) for S&P 500 index options.

### Key Findings
- **Black-Scholes (Historical Vol)**: $22.00 MAE baseline
- **Neural Network (Basic Features)**: $17.45 MAE (21% improvement)
- **Neural Network (Full Features)**: $2.81 MAE (87% improvement)

Neural networks with market signals (implied volatility + Greeks) achieve near-perfect pricing accuracy.

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

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/option-pricing-thesis.git
cd option-pricing-thesis
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate thesis-env
```

### 3. Obtain Data
Download the following files from WRDS OptionMetrics and place in `data/raw/`:
- `SPX_Forward_Prices_Complete_2018-2025.csv`
- `SPX_Options_raw_2018-2025.csv`

**Data source**: Wharton Research Data Services (WRDS)  
**Access**: Requires institutional subscription

### 4. Run Pipeline

**Full pipeline** (recommended first run):
```bash
python main.py --run-all
```

**Or run individual steps:**
```bash
# Step 1: Preprocess data and calculate Black-Scholes baseline
python main.py --preprocess

# Step 2: Train neural networks
python main.py --train
```

## Output Files

All results are saved to `results/`:
- `SPX_MERGED_TO_USE.csv` - Merged options + forward prices
- `SPX_with_BS_Both_Vols.csv` - Black-Scholes predictions
- `test_predictions_comparison.csv` - Neural network predictions

Models saved to `models/`:
- `best_NN_Basic.pth` - Basic features model
- `best_NN_Full.pth` - Full features model (with IV/Greeks)

## Methodology

### Data
- **Source**: WRDS OptionMetrics
- **Period**: 2018-2025
- **Filters**: European options, 0.7-1.3 moneyness, volume ≥10, OI ≥100
- **Final dataset**: ~4.3M high-quality option contracts

### Models
1. **Black-Scholes**: Baseline using 60-day historical volatility (Hutchinson et al. 1994)
2. **Neural Network (Basic)**: Contract characteristics + liquidity only
3. **Neural Network (Full)**: Basic + implied volatility + Greeks

### Features
- **Basic**: Moneyness, time to maturity, volume, open interest
- **Advanced**: Implied volatility, delta, gamma, vega, theta
- **Engineered**: Log transforms, interactions (moneyness × time)

### Architecture
- 3 hidden layers: [128, 64, 32] neurons
- ReLU activations, Batch Normalization, Dropout (0.2)
- AdamW optimizer, learning rate scheduling
- Early stopping (patience=30)

### Evaluation
- **Metric**: Mean Absolute Error (MAE) in dollars
- **Split**: 70% train, 15% validation, 15% test (temporal)

## Key Results

| Model | Test MAE | vs BS | Description |
|-------|----------|-------|-------------|
| Black-Scholes (Historical Vol) | $22.00 | Baseline | 60-day rolling volatility |
| NN (Basic Features) | $17.45 | -21% | Pure ML learning |
| NN (Full Features) | $2.81 | -87% | ML + market signals |

The dramatic improvement from Basic → Full demonstrates that implied volatility and Greeks encode substantial pricing information beyond what neural networks can learn from contract characteristics alone.

## Reproducing Results
```bash
# Complete reproduction
python main.py --run-all

# Results will match thesis if using same data
# Training uses fixed random seed (42) for reproducibility
```

## Dependencies

- pandas 1.5+
- numpy 1.23+
- scipy 1.9+
- scikit-learn 1.1+
- pytorch 2.0+
- matplotlib 3.6+
- pandas-datareader 0.10+

See `environment.yml` for complete list.

## Citation

If you use this code or methodology, please cite:
```bibtex
@mastersthesis{rigoli2025option,
  author = {Rigoli, Bastian},
  title = {Neural Networks for SPX Option Pricing: A Comparison with Traditional Models},
  school = {[Your University]},
  year = {2025},
  type = {Master's Thesis}
}
```

## References

Key papers implemented:
- Hutchinson, Lo & Poggio (1994) - "A Nonparametric Approach to Pricing and Hedging Derivative Securities"
- Garcia & Gençay (2000) - "Pricing and Hedging with Neural Networks"
- Black & Scholes (1973) - "The Pricing of Options and Corporate Liabilities"

See `project_report.pdf` for complete bibliography.

## License

MIT License - See LICENSE file

## Contact

Bastian Rigoli  
Email: [your.email@university.edu]  
GitHub: [github.com/YOUR_USERNAME]

## Acknowledgments

- Data provided by Wharton Research Data Services (WRDS)
- Supervisor: [Supervisor Name]
- [Your University] - [Department Name]

---

**Note**: This repository contains code only. Data files are not included due to licensing restrictions. Users must obtain data access through WRDS independently.