"""
Neural Network for SPX Option Pricing
=====================================

Walk-forward validation (5 folds: 2021-2025)
Uses FEATURES_BASIC from config.py (created in preprocessing)

NOTE: This file does NOT create features - they come from data_preprocessing.py
"""
import sys
import os
import csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import time


def run_training():
    """Main training function called by main.py"""
    
    WALK_FORWARD = True  # Set to False for single train/test split

    # ==============================================================================
    # WALK-FORWARD VALIDATION SETUP (5 FOLDS)
    # ==============================================================================
    
    fold_configs = [
    {'name': 'Fold 1', 'train_end': '2020-12-31', 'test_end': '2021-12-31'},
    {'name': 'Fold 2', 'train_end': '2021-12-31', 'test_end': '2022-12-31'},
    {'name': 'Fold 3', 'train_end': '2022-12-31', 'test_end': '2023-12-31'},
    {'name': 'Fold 4', 'train_end': '2023-12-31', 'test_end': '2024-12-31'},
    {'name': 'Fold 5', 'train_end': '2024-12-31', 'test_end': '2025-08-29'},
]
    
    
    # ==============================================================================
    # LOAD DATA
    # ==============================================================================
    
    print("="*60)
    print("NEURAL NETWORK FOR SPX OPTIONS")
    print("Using WRDS OptionMetrics Data")
    print("="*60)

    df = pd.read_csv(SPX_BS_HIST_FILE, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    print(f"\nLoaded: {len(df):,} options")
    print(f"Columns available: {len(df.columns)}")

    # ==============================================================================
    # VERIFY FEATURES EXIST (Created in preprocessing)
    # ==============================================================================
    
    print("\n" + "="*60)
    print("VERIFYING FEATURES")
    print("="*60)
    
    features_basic = FEATURES_BASIC
    missing_features = [f for f in features_basic if f not in df.columns]
    
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        print("   Run data_preprocessing.py first!")
        return None
    
    print(f"✅ All {len(features_basic)} features available")
    for feat in features_basic:
        print(f"   ✓ {feat}")

    # ==============================================================================
    # PREPARE DATA
    # ==============================================================================
    
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)

    # Sort by date for temporal split
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove any rows with missing features
    df_clean = df.dropna(subset=features_basic + ['mid_price']).copy()
    print(f"After removing NaNs: {len(df_clean):,} options")
    print(f"Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")

    # ============================================================
    # MODEL DEFINITION
    # ============================================================

    class OptionPricingMLP(nn.Module):
        def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
            super(OptionPricingMLP, self).__init__()
            
            layers = []
            prev_size = input_size
            
            for i, hidden_size in enumerate(hidden_sizes):
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, 1))
            self.network = nn.Sequential(*layers)
            
            # He initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            return self.network(x)

    # ============================================================
    # TRAINING FUNCTION
    # ============================================================

    def train_model(feature_columns, model_name, df_train, df_val, df_test, num_epochs=300, batch_size=4096, 
                    learning_rate=0.001, patience=60):
        """
        Train a neural network model for option pricing
        
        Parameters:
        -----------
        feature_columns : list
            List of feature column names to use
        model_name : str
            Name for the model (used for saving)
        num_epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Initial learning rate
        patience : int
            Early stopping patience
        
        Returns:
        --------
        dict : Dictionary containing trained model, scalers, losses, and metadata
        """
        
        print("\n" + "="*60)
        print(f"TRAINING: {model_name}")
        print("="*60)
        print(f"Features: {len(feature_columns)}")
        
        # Prepare data
        X_train = df_train[feature_columns].values
        y_train = df_train['mid_price'].values

        X_val = df_val[feature_columns].values
        y_val = df_val['mid_price'].values

        X_test = df_test[feature_columns].values
        y_test = df_test['mid_price'].values
            
        # Check for nan values
        def check_and_clean(X):
            if np.isnan(X).sum() > 0 or np.isinf(X).sum() > 0:
                print(f"  ⚠️  Cleaning {np.isnan(X).sum()} NaNs and {np.isinf(X).sum()} Infs")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return X
        
        X_train = check_and_clean(X_train)
        X_val = check_and_clean(X_val)
        X_test = check_and_clean(X_test)
        
        # Scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        
        print("✅ Data prepared and scaled")
        
        # Model setup
        np.random.seed(42)
        torch.manual_seed(42)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        input_size = len(feature_columns)
        mlp_model = OptionPricingMLP(
            input_size=input_size,
            hidden_sizes=[128, 64, 32],
            dropout_rate=0.3
        ).to(device)
        
        total_params = sum(p.numel() for p in mlp_model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(mlp_model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).reshape(-1, 1).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).reshape(-1, 1).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        print(f"\nTraining for max {num_epochs} epochs...")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Patience: {patience}")
        

        start_time = time.perf_counter() # starting point to measure training time

        for epoch in range(num_epochs):
            # Training phase
            mlp_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = mlp_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation phase
            mlp_model.eval()
            with torch.no_grad():
                val_outputs = mlp_model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                model_path = os.path.join(MODELS_DIR, f'best_{model_name}.pth')
                torch.save(mlp_model.state_dict(), model_path)
                print(f"✅ Model saved: {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                    break
        # End of training time 
        total_time = time.perf_counter() - start_time
        print(f"\n⏱️ Training time for {model_name}: {total_time:.1f} seconds "
        f"({total_time/60:.1f} minutes)")
        # Save time to csv
        runtime_path = os.path.join(RESULTS_DIR, "training_times.csv")
        header = ["model_name", "n_train", "n_val", "n_test", "seconds"]
        row = [model_name, len(df_train), len(df_val), len(df_test), total_time]

        file_exists = os.path.exists(runtime_path)
        with open(runtime_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        
        # Load best model from saved file
        model_path = os.path.join(MODELS_DIR, f'best_{model_name}.pth')
        mlp_model.load_state_dict(torch.load(model_path))
                
        print(f"\n✅ Training completed")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Total epochs: {len(train_losses)}")
        
        # Return everything needed for evaluation
        return {
            'model': mlp_model,
            'model_name': model_name,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'feature_columns': feature_columns,
            'device': device,
            # Tensors for evaluation
            'X_train_tensor': X_train_tensor,
            'X_val_tensor': X_val_tensor,
            'X_test_tensor': X_test_tensor,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
        }

    # ============================================================
    # EVALUATION FUNCTION
    # ============================================================

    def evaluate_model(trained_model_dict):
        """
        Evaluate a trained model and return comprehensive metrics
        
        Parameters:
        -----------
        trained_model_dict : dict
            Dictionary returned by train_model function
        
        Returns:
        --------
        dict : Dictionary containing predictions and metrics
        """
        
        print("\n" + "="*60)
        print(f"EVALUATING: {trained_model_dict['model_name']}")
        print("="*60)
        
        model = trained_model_dict['model']
        scaler_y = trained_model_dict['scaler_y']
        model_name = trained_model_dict['model_name']
        
        # Get tensors
        X_train_tensor = trained_model_dict['X_train_tensor']
        X_val_tensor = trained_model_dict['X_val_tensor']
        X_test_tensor = trained_model_dict['X_test_tensor']
        
        y_train = trained_model_dict['y_train']
        y_val = trained_model_dict['y_val']
        y_test = trained_model_dict['y_test']
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            y_train_scaled = model(X_train_tensor).cpu().numpy().flatten()
            y_val_scaled = model(X_val_tensor).cpu().numpy().flatten()
            y_test_scaled = model(X_test_tensor).cpu().numpy().flatten()
        
        # Inverse transform to dollar prices
        y_train_pred = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()
        y_val_pred = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
        y_test_pred = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        def calculate_metrics(y_true, y_pred):
            errors = y_pred - y_true
            abs_errors = np.abs(errors)
            
            return {
                'mae': np.mean(abs_errors),
                'rmse': np.sqrt(np.mean(errors ** 2)),
                'median_ae': np.median(abs_errors),
                'max_error': np.max(abs_errors),
                'std_error': np.std(errors),
                'mean_error': np.mean(errors),
            }
        
        train_metrics = calculate_metrics(y_train, y_train_pred)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        
        # Print results
        print(f"\n{'Set':<12} {'MAE':<12} {'RMSE':<12} {'Median AE':<12} {'Max Error':<12}")
        print("-" * 60)
        print(f"{'Train':<12} ${train_metrics['mae']:<11.2f} ${train_metrics['rmse']:<11.2f} "
            f"${train_metrics['median_ae']:<11.2f} ${train_metrics['max_error']:<11.2f}")
        print(f"{'Validation':<12} ${val_metrics['mae']:<11.2f} ${val_metrics['rmse']:<11.2f} "
            f"${val_metrics['median_ae']:<11.2f} ${val_metrics['max_error']:<11.2f}")
        print(f"{'Test':<12} ${test_metrics['mae']:<11.2f} ${test_metrics['rmse']:<11.2f} "
            f"${test_metrics['median_ae']:<11.2f} ${test_metrics['max_error']:<11.2f}")
        
        print(f"\n✅ Evaluation completed for {model_name}")
        
        return {
            'model_name': model_name,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred,
                'test': y_test_pred,
            },
            'metrics': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics,
            },
            'train_losses': trained_model_dict['train_losses'],
            'val_losses': trained_model_dict['val_losses'],
            'feature_columns': trained_model_dict['feature_columns'],
        }

    # ============================================================
    # WALK-FORWARD TRAINING
    # ============================================================

    print("\n" + "="*80)
    print("WALK-FORWARD TRAINING (5 FOLDS)")
    print("="*80)

    if WALK_FORWARD:
        print("\n" + "="*80)
        print("WALK-FORWARD TRAINING (5 FOLDS)")
        print("="*80)
        
        all_fold_results_basic = []
        #all_fold_results_full = []
        
        for fold_idx, fold_config in enumerate(fold_configs):
            print(f"\n{'='*80}")
            print(f"FOLD {fold_idx+1}/5: {fold_config['name']}")
            print(f"{'='*80}")
            
            # Use ALL data up to train_end (like Random Forest)
            train_mask = df_clean['date'] <= fold_config['train_end']
            test_mask = (df_clean['date'] > fold_config['train_end']) & (df_clean['date'] <= fold_config['test_end'])
            
            df_train_fold = df_clean[train_mask].copy()  # ALL training data
            df_test_fold = df_clean[test_mask].copy()
            
            # Use last 15% of training data for validation (early stopping only)
            n_total = len(df_train_fold)
            n_val_start = int(0.85 * n_total)
            df_val_fold = df_train_fold.iloc[n_val_start:].copy()
            
            print(f"\nTrain: {len(df_train_fold):,} options (up to {fold_config['train_end']})")
            print(f"Val:   {len(df_val_fold):,} options (last 15%, for early stopping)")
            print(f"Test:  {len(df_test_fold):,} options ({fold_config['test_end'][:4]})")
                    
            if len(df_test_fold) < 100:
                print(f"⚠️  Insufficient test data, skipping fold")
                continue
            
            # Train Basic model
            print(f"\n--- Training Basic Model (Fold {fold_idx+1}) ---")
            trained_basic = train_model(
                feature_columns=features_basic,
                model_name=f"NN_Basic_Fold{fold_idx+1}",
                df_train=df_train_fold,
                df_val=df_val_fold,
                df_test=df_test_fold,
                num_epochs=300,
                batch_size=2048,
                learning_rate=0.001,
                patience=60
            )
            results_basic = evaluate_model(trained_basic)
            
            all_fold_results_basic.append({
                'fold': fold_config['name'],
                'test_year': fold_config['test_end'][:4],
                'mae': results_basic['metrics']['test']['mae'],
                'rmse': results_basic['metrics']['test']['rmse'],
            })
    

        # ==============================================================================
        # SUMMARY
        # ==============================================================================
        
        print("\n" + "="*80)
        print("📊 WALK-FORWARD AVERAGE RESULTS (5 FOLDS)")
        print("="*80)
            
        avg_mae_basic = np.mean([r['mae'] for r in all_fold_results_basic])
        #avg_mae_full = np.mean([r['mae'] for r in all_fold_results_full])
        
        bs_baseline = 24.14  # From preprocessing
        
        print(f"\n{'Model':<30} {'Avg Test MAE':<15} {'vs BS':<15}")
        print("-" * 60)
        print(f"{'BS (Historical Vol)':<30} ${bs_baseline:<14.2f} Baseline")
        print(f"{'NN (Basic + Hist Vol)':<30} ${avg_mae_basic:<14.2f} {(bs_baseline-avg_mae_basic)/bs_baseline*100:+.1f}%")
        #print(f"{'NN (Full + IV + Greeks)':<30} ${avg_mae_full:<14.2f} {(bs_baseline-avg_mae_full)/bs_baseline*100:+.1f}%")
        
        print(f"\n📊 Fold-by-Fold Comparison:")
        print(f"\n{'Fold':<12} {'Test Year':<12} {'BS MAE':<12} {'Basic MAE':<12} {'Full MAE':<12}")
        print("-" * 60)
        
        for r in all_fold_results_basic:
            print(f"{r['fold']:<12} {r['test_year']:<12} ${r['mae']:<11.2f} ${r['rmse']:<11.2f}")
        
        print("-" * 48)
        print(f"{'Average':<12} {'All':<12} ${avg_mae_basic:<11.2f}")
        
        # Save results
        results_df = pd.DataFrame(all_fold_results_basic)
        results_df.to_csv(os.path.join(RESULTS_DIR, 'nn_walk_forward_results.csv'), index=False)
        print(f"\n✅ Results saved to: nn_walk_forward_results.csv")
        
        print("\n✅ Walk-forward training completed!")
        
        return results_df
    
    else:
         # Single split mode (for quick testing)
        print("\n" + "="*60)
        print("SINGLE SPLIT TRAINING")
        print("="*60)
        
        n = len(df_clean)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        
        df_train = df_clean.iloc[:train_end].copy()
        df_val = df_clean.iloc[train_end:val_end].copy()
        df_test = df_clean.iloc[val_end:].copy()
        
        trained_basic = train_model(features_basic, "NN_Basic", df_train, df_val, df_test)
        results_basic = evaluate_model(trained_basic)
        
        return results_basic
    
# ============================================================
# NOTE: Full model (IV + Greeks) excluded
# ============================================================
# IV (implied volatility) is derived from market prices via BS inversion,
# so using it to predict prices creates circular reasoning.
# The Full model achieved ~$1.73 MAE but is methodologically problematic.
# See thesis discussion section.
# ============================================================


if __name__ == "__main__":
    run_training()


