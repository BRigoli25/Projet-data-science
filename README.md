# Machine Learning for SPX Option Pricing

**Author:** Bastian Rigoli  
**Course:** Advanced Programming 2025  
**Institution:** University of Lausanne

## Project Overview

This project compares machine learning approaches (Neural Networks, Random Forest, XGBoost) against Black-Scholes for pricing S&P 500 index options. 

**Key Innovation:** Two-pass training methodology ensures fair comparison by having all models use 100% of training data.

---

## 📁 Project Structure
```
option-pricing-project/
├── README.md                    # This file
├── PROPOSAL.md                  # Project proposal
├── requirements.txt             # Pip dependencies
├── environment.yml              # Conda environment
├── main.py                      # Entry point - RUN THIS
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data preprocessing & BS baseline
│   ├── models.py               # Neural Network, RF, XGBoost
│   └── evaluation.py           # Visualizations
├── data/
│   └── raw/                    # Place data files here (see below)
├── results/                    # Auto-generated outputs
└── models/                     # Auto-generated saved models
```

---

# Machine Learning for SPX Option Pricing

## 🚀 Quick Start - TWO OPTIONS

### **Option 1: Demo Mode (< 30 minutes) ⚡ RECOMMENDED FOR TAs**

Quick verification with pre-computed results:
```bash
# 1. Setup
git clone https://github.com/BRigoli25/Projet-data-science.git
cd Projet-data-science
bash setup.sh
conda activate fin_project

# 2. Get pre-computed files (~600 MB)
# Download from: [LINK] (contact bastian.rigoli@unil.ch)
# Extract to project root

# 3. Run demo
python demo.py
```

**What you get:**
- ✅ All results verified in < 2 minutes
- ✅ Load preprocessed data (5 sec instead of 9 min)
- ✅ Load pre-trained models (instant instead of 2+ hours)
- ✅ Generate visualizations (30 sec)


### **Option 2: Full Pipeline (148 minutes)**

Train everything from scratch:
```bash
# 1. Setup (same as above)
git clone https://github.com/BRigoli25/Projet-data-science.git
cd Projet-data-science
bash setup.sh
conda activate fin_project

# 2. Get raw data (~3.5 GB)
# Contact: bastian.rigoli@unil.ch
# Place in: data/raw/

# 3. Run full pipeline
python main.py
```

**What happens:**
- Preprocessing: 9 minutes
- Neural Network training: 128 minutes
- Random Forest: 10 minutes
- XGBoost: 25 seconds
- Visualizations: 1 minute

**Perfect for:** Reproducing results, verifying training, research

---

## 📊 Results (Either Option)

| Model | MAE | vs Black-Scholes |
|-------|-----|------------------|
| Black-Scholes | $19.56 | Baseline |
| Neural Network | $12.19 | **+37.6%** ✅ |
| Random Forest | $12.87 | **+34.2%** |
| XGBoost | $12.42 | **+36.5%** |


## 📊 Data Setup

**IMPORTANT:** Due to GitHub size limits and WRDS licensing, data files are not included in this repository.

### **Required Data Files**

Place these files in `data/raw/`:

1. **SPX_Options_raw_2018-2025.csv** (~3.5 GB)
   - Source: WRDS OptionMetrics
   - Contains: SPX option quotes (bid/ask/mid, volume, OI, implied vol)

2. **SPX_Forward_Prices_Complete_2018-2025.csv** (~100 MB)
   - Source: WRDS OptionMetrics
   - Contains: Forward prices for dividend adjustment

3. **treasury_3month_rates.csv** (optional - auto-downloads if missing)
   - Source: FRED API
   - Contains: 3-month Treasury rates (risk-free rate proxy)

### **For Course TAs/Graders**

**If you do not have access to WRDS data:**

Contact me at: bastian.rigoli@unil.ch

I can provide:
- Sample dataset (1000 options) for testing code functionality
- Full dataset via secure file transfer
- Pre-computed results for verification

**Note:** The code will auto-download Treasury rates from FRED if missing.

---

## ⚙️ Usage

### **Run Full Pipeline**
```bash
python main.py
```

**Executes:**
1. Data preprocessing & Black-Scholes baseline (~9 min)
2. Neural Network training with two-pass method (~113 min)
3. Random Forest training (~10 min)
4. XGBoost training (~2 min)
5. Visualization generation (~1 min)

**Total Runtime:** ~135 minutes on MacBook Pro (M1)

### **Run Individual Components**
```bash
# Only preprocessing
python main.py --preprocess

# Only model training (requires preprocessed data)
python main.py --models

# Only visualizations (requires results CSVs)
python main.py --viz
```

---

## 📈 Expected Outputs

### **Results Directory (`results/`)**

After running, you'll find:

**CSV Files:**
- `SPX_with_BS_Historical.csv` - Preprocessed data with BS prices
- `bs_walk_forward_results.csv` - BS baseline metrics
- `nn_walk_forward_results.csv` - Neural Network results
- `rf_walk_forward_results.csv` - Random Forest results
- `xgb_walk_forward_results.csv` - XGBoost results
- `rf_feature_importance.csv` - Feature importance rankings
- `xgb_feature_importance.csv`

**Plots Directory (`results/plots/`):**
- `mae_comparison.png` - Model performance comparison
- `feature_importance.png` - Top predictive features
- `performance_by_year.png` - Temporal analysis
- Additional diagnostic plots

### **Models Directory (`models/`)**

Saved trained models:
- `best_NN_Basic_Fold[1-5]_FINAL.pth` - Neural networks (5 folds)
- `RF_Basic_Fold[1-5].joblib` - Random Forests (5 folds)
- `XGB_Basic_Fold[1-5].joblib` - XGBoost models (5 folds)

---

## 🔬 Key Results

| Model | Avg Test MAE | vs Black-Scholes |
|-------|--------------|------------------|
| **Black-Scholes** | $19.56 | Baseline |
| **Neural Network** | $12.09 | **-38%** ✅ |
| **Random Forest** | $12.87 | **-34%** |
| **XGBoost** | $13.05 | **-33%** |

**Key Finding:** Bid-ask spread (market microstructure) dominates feature importance at 34-36%, while historical volatility contributes only 1-2%. This suggests ML models excel at capturing liquidity effects that theoretical models miss.

---

## 🧪 Testing the Code (For TAs)

### **Quick Functionality Test (Without Full Data)**

If you don't have the full dataset, you can test the code structure:
```bash
# Check imports work
python -c "from src.data_loader import FEATURES_BASIC; print('✅ Imports OK')"
python -c "from src.models import train_neural_network; print('✅ Models OK')"
python -c "from src.evaluation import generate_all_plots; print('✅ Evaluation OK')"

# Check CLI
python main.py --help
```

### **Full Test (With Data)**
```bash
# Run full pipeline
python main.py

# Expected runtime: ~135 minutes
# Expected outputs: 15+ CSV files, 10+ plots, 15 model files
```

---

## 📦 Dependencies

**Core Libraries:**
- Python 3.10+
- NumPy, Pandas, SciPy
- Scikit-learn (Random Forest, preprocessing)
- XGBoost (gradient boosting)
- PyTorch (neural networks)
- Matplotlib, Seaborn (visualizations)

**Data Sources:**
- WRDS OptionMetrics (option data)
- FRED API (risk-free rates)

---

## 🐛 Troubleshooting

### **Issue: "ModuleNotFoundError: No module named 'xgboost'"**
```bash
pip install xgboost
```

### **Issue: "FileNotFoundError: data/raw/SPX_Options_raw_2018-2025.csv"**

Data files missing. See **Data Setup** section above.

### **Issue: PyTorch CUDA errors**

The code automatically uses CPU if GPU unavailable. To disable CUDA warnings:
```python
# In src/models.py (already handled)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### **Issue: "Treasury rates download failed"**

The code auto-downloads from FRED. If that fails, it uses a fallback rate of 4%. To manually fix:
```bash
# Check internet connection or download manually from:
# https://fred.stlouisfed.org/series/DTB3
```

---

## 📧 Contact

**Bastian Rigoli**  
Email: bastian.rigoli@unil.ch  
GitHub: [BRigoli25](https://github.com/BRigoli25)

---

## 📄 License

This project is submitted as coursework for Advanced Programming 2025 at the University of Lausanne. All rights reserved.

---

## 🙏 Acknowledgments

- **WRDS OptionMetrics** for comprehensive options data
- **Hutchinson et al. (1994)** for neural network option pricing methodology
- **Grinsztajn et al. (2022)** for insights on tree-based models vs neural networks on tabular data
- **Course instructors** for guidance and feedback