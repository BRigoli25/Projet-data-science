#!/bin/bash
#
# Automated TA Demo Package Builder
# Creates a complete, ready-to-use demo package
#

set -e  # Exit on error

echo "🚀 Building TA Demo Package"
echo "============================"
echo ""

# Check we're in the right directory
if [ ! -f "demo.py" ]; then
    echo "❌ ERROR: Run this script from the project root directory"
    echo "   (where demo.py is located)"
    exit 1
fi

# Clean old package
if [ -d "ta_demo_package" ]; then
    echo "🧹 Cleaning old package..."
    rm -rf ta_demo_package
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p ta_demo_package/src
mkdir -p ta_demo_package/results/plots
mkdir -p ta_demo_package/models

# Copy Python files
echo "📄 Copying Python files..."
cp demo.py ta_demo_package/
cp src/*.py ta_demo_package/src/
cp requirements.txt ta_demo_package/

# Copy main README (for project info)
if [ -f "README.md" ]; then
    cp README.md ta_demo_package/README_PROJECT.md
fi

# Copy preprocessed data
echo "💾 Copying preprocessed data..."
if [ -f "results/preprocessed_data_lite.pkl" ]; then
    cp results/preprocessed_data_lite.pkl ta_demo_package/results/
    echo "   ✅ preprocessed_data_lite.pkl"
else
    echo "   ⚠️  WARNING: preprocessed_data_lite.pkl not found!"
    echo "   Run: python save_preprocessed_data.py"
fi

# Copy result CSVs
echo "📊 Copying result files..."
CSV_COUNT=0
for csv in results/*_results.csv results/*_feature_importance.csv results/SPX_with_BS_Historical.csv; do
    if [ -f "$csv" ]; then
        cp "$csv" ta_demo_package/results/
        CSV_COUNT=$((CSV_COUNT + 1))
        echo "   ✅ $(basename $csv)"
    fi
done

if [ $CSV_COUNT -lt 4 ]; then
    echo "   ⚠️  WARNING: Only found $CSV_COUNT result files (expected 6+)"
    echo "   Some results may be missing. Run: python demo.py or python main.py"
fi

# Copy trained models
echo "🤖 Copying trained models..."
MODEL_COUNT=0
for model in models/*.pth models/*.joblib; do
    if [ -f "$model" ]; then
        cp "$model" ta_demo_package/models/
        MODEL_COUNT=$((MODEL_COUNT + 1))
    fi
done

if [ $MODEL_COUNT -ge 15 ]; then
    echo "   ✅ Copied $MODEL_COUNT model files"
else
    echo "   ⚠️  Only found $MODEL_COUNT model files (expected 15 for all folds)"
    echo "   Models may be incomplete. Run: python main.py"
fi

# Create environment.yml
echo "🔧 Creating environment.yml..."
conda env export --from-history > ta_demo_package/environment.yml
echo "   ✅ environment.yml created"

# Copy TA-specific README
echo "📚 Adding TA instructions..."
if [ -f "TA_DEMO_README.md" ]; then
    cp TA_DEMO_README.md ta_demo_package/README.md
else
    # Create a basic README if not found
    cat > ta_demo_package/README.md << 'EOF'
# TA Demo Package - Quick Start

## Setup & Run (< 2 minutes)

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate fin_project

# 2. Run demo
python demo.py
```

That's it! Demo loads pre-computed results and generates visualizations.

## What You'll See
- Model performance comparison
- Feature importance analysis  
- Detailed evaluation metrics
- Generated plots in results/plots/

## Files Included
- demo.py: Main demo script
- results/: Pre-computed results and data
- models/: Trained models (NN, RF, XGB)
- src/: Source code modules

Total time: ~30-45 seconds after setup.
EOF
fi
echo "   ✅ README.md added"

# Calculate package size
echo ""
echo "📦 Package Summary"
echo "=================="
TOTAL_SIZE=$(du -sh ta_demo_package | cut -f1)
echo "Total size: $TOTAL_SIZE"

# File counts
echo ""
echo "Contents:"
echo "  Python files: $(find ta_demo_package -name "*.py" | wc -l)"
echo "  Result CSVs: $(find ta_demo_package/results -name "*.csv" 2>/dev/null | wc -l)"
echo "  Trained models: $(find ta_demo_package/models -name "*.*" 2>/dev/null | wc -l)"
echo "  PKL files: $(find ta_demo_package -name "*.pkl" 2>/dev/null | wc -l)"

# Create ZIP
echo ""
echo "📦 Creating ZIP file..."
if [ -f "ta_demo_package.zip" ]; then
    rm ta_demo_package.zip
fi

zip -r ta_demo_package.zip ta_demo_package/ -x "*.DS_Store" "**/__pycache__/*"
ZIP_SIZE=$(du -sh ta_demo_package.zip | cut -f1)

echo "   ✅ ta_demo_package.zip created ($ZIP_SIZE)"

# Final summary
echo ""
echo "================================"
echo "✅ Package built successfully!"
echo "================================"
echo ""
echo "📦 Package: ta_demo_package.zip ($ZIP_SIZE)"
echo "📁 Folder: ta_demo_package/ ($TOTAL_SIZE)"
echo ""
echo "Next steps:"
echo "  1. Upload ta_demo_package.zip to file sharing service"
echo "  2. Share link with TA"
echo ""
echo "TA workflow:"
echo "  unzip ta_demo_package.zip"
echo "  cd ta_demo_package"
echo "  conda env create -f environment.yml"
echo "  conda activate fin_project"
echo "  python demo.py"
echo ""
echo "Demo completes in 30-45 seconds! ⚡"
echo "================================"
