"""
Main Entry Point for SPX Option Pricing Thesis
Orchestrates preprocessing, training, and evaluation
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import run_preprocessing
from src.neural_network import run_training
from src.config import *

def check_data_files():
    """Check if required data files exist"""
    required_files = [SPX_FORWARD_FILE, SPX_OPTIONS_FILE]
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("\n⚠️  ERROR: Missing required data files!")
        print("\nPlease place the following files in data/raw/:")
        for f in missing:
            print(f"  - {os.path.basename(f)}")
        print("\nData source: WRDS OptionMetrics")
        print("See README.md for instructions")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description='SPX Option Pricing - Neural Networks vs Black-Scholes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --run-all              # Run complete pipeline
  python main.py --preprocess           # Only preprocess data
  python main.py --train                # Only train models
        """
    )
    
    parser.add_argument('--run-all', action='store_true',
                       help='Run complete pipeline (preprocess + train)')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run data preprocessing and Black-Scholes baseline')
    parser.add_argument('--train', action='store_true',
                       help='Train neural network models')
    parser.add_argument('--skip-check', action='store_true',
                       help='Skip data file existence check')
    
    args = parser.parse_args()
    
    # Default: run all if no flags specified
    if not any([args.run_all, args.preprocess, args.train]):
        args.run_all = True
    
    # Print header
    print("="*70)
    print(" SPX OPTION PRICING ")
    print(" Neural Networks vs Traditional Models")
    print("="*70)
    print(f"\nProject directory: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    
    # Check data files exist (unless skipped)
    if not args.skip_check:
        if not check_data_files():
            sys.exit(1)
    
    # Run requested steps
    try:
        if args.run_all or args.preprocess:
            print("\n" + "="*70)
            print(" STEP 1: DATA PREPROCESSING & BLACK-SCHOLES BASELINE")
            print("="*70)
            run_preprocessing()
        
        if args.run_all or args.train:
            print("\n" + "="*70)
            print(" STEP 2: NEURAL NETWORK TRAINING")
            print("="*70)
            
            # Check if merged data exists
            if not os.path.exists(SPX_MERGED_FILE):
                print(f"\n⚠️  ERROR: {SPX_MERGED_FILE} not found!")
                print("Run preprocessing first: python main.py --preprocess")
                sys.exit(1)
            
            run_training()
        
        # Success summary
        print("\n" + "="*70)
        print(" ✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nResults saved to: {RESULTS_DIR}")
        print(f"Models saved to: {MODELS_DIR}")
        print("\nGenerated files:")
        
        output_files = [
            SPX_MERGED_FILE,
            SPX_CLEAN_FILE,
            SPX_BS_BOTH_FILE,
            SPX_BS_HIST_FILE,
            TEST_PREDICTIONS_FILE,
        ]
        
        for f in output_files:
            if os.path.exists(f):
                size_mb = os.path.getsize(f) / (1024**2)
                print(f"  ✓ {os.path.basename(f)} ({size_mb:.1f} MB)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()