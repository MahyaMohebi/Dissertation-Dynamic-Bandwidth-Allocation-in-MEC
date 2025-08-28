#!/usr/bin/env python3
"""Evaluate a trained LSTM on a test set and plot a predicted-vs-actual series for sanity.
Usage:
  python ml/evaluate_lstm.py --model ml/models/checkpoints/model.h5 --test ml/data/processed/test_windows.npz --out results/figures/pred_vs_actual.png
"""
import argparse, numpy as np, tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', default='results/figures/pred_vs_actual.png')
    ap.add_argument('--normalize', action='store_true', default=True, help='Use normalization (requires scalers)')
    args = ap.parse_args()

    # Load test data
    data = np.load(args.test)
    X, y = data['X'], data['y']
    
    print(f"Test data: X={X.shape}, y={y.shape}")
    print(f"Original y range: [{y.min():.2f}, {y.max():.2f}]")

    # Load model
    model = tf.keras.models.load_model(args.model, compile=False)
    
    # Handle normalization if used during training
    X_test, y_test = X.copy(), y.copy()
    y_scaler = None
    
    if args.normalize:
        scaler_dir = os.path.dirname(args.model)
        X_scaler_path = os.path.join(scaler_dir, 'X_scaler.pkl')
        y_scaler_path = os.path.join(scaler_dir, 'y_scaler.pkl')
        
        if os.path.exists(X_scaler_path) and os.path.exists(y_scaler_path):
            X_scaler = joblib.load(X_scaler_path)
            y_scaler = joblib.load(y_scaler_path)
            
            # Normalize test data
            X_reshaped = X_test.reshape(-1, 1)
            X_test = X_scaler.transform(X_reshaped).reshape(X_test.shape)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
            
            print("Using saved scalers for normalization")
        else:
            print("Warning: Normalization requested but scalers not found. Using raw data.")
            y_scaler = None

    # Make predictions
    yhat = model.predict(X_test, verbose=0).squeeze()
    
    # Denormalize predictions if needed
    if y_scaler is not None:
        yhat = y_scaler.inverse_transform(yhat.reshape(-1, 1)).flatten()
        y_test = y  # Use original unnormalized targets for evaluation
    else:
        y_test = y

    # Calculate metrics
    mae = float(np.mean(np.abs(yhat - y_test)))
    rmse = float(np.sqrt(np.mean((yhat - y_test)**2)))
    mape = float(np.mean(np.abs((y_test - yhat) / (y_test + 1e-8))) * 100)  # Add small epsilon to avoid division by zero
    r2 = float(1 - np.sum((y_test - yhat)**2) / np.sum((y_test - np.mean(y_test))**2))
    
    print(f'\nTest Results:')
    print(f'MAE: {mae:.4f} Mbps')
    print(f'RMSE: {rmse:.4f} Mbps') 
    print(f'MAPE: {mape:.2f}%')
    print(f'R²: {r2:.4f}')

    # Create comprehensive plots
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Time series comparison (first 200 points)
    n = min(200, len(y_test))
    ax1.plot(y_test[:n], label='Actual', alpha=0.8)
    ax1.plot(yhat[:n], label='Predicted', alpha=0.8)
    ax1.set_title('Bandwidth Prediction vs Actual (First 200 samples)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Bandwidth (Mbps)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of predicted vs actual
    ax2.scatter(y_test[::10], yhat[::10], alpha=0.5, s=1)  # Subsample for visibility
    min_val, max_val = min(y_test.min(), yhat.min()), max(y_test.max(), yhat.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax2.set_xlabel('Actual Bandwidth (Mbps)')
    ax2.set_ylabel('Predicted Bandwidth (Mbps)')
    ax2.set_title(f'Predicted vs Actual (R² = {r2:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    errors = yhat - y_test
    ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_xlabel('Prediction Error (Mbps)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Error Distribution (MAE = {mae:.2f})')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Absolute error over time
    abs_errors = np.abs(errors)
    ax4.plot(abs_errors[:n], alpha=0.7)
    ax4.axhline(mae, color='red', linestyle='--', label=f'Mean AE = {mae:.2f}')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Absolute Error (Mbps)')
    ax4.set_title('Absolute Error Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches='tight', dpi=150)
    print(f'Comprehensive evaluation plots saved to {args.out}')
    
    # Save detailed results
    results_file = args.out.replace('.png', '_metrics.txt')
    with open(results_file, 'w') as f:
        f.write(f"Model Evaluation Results\n")
        f.write(f"========================\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Data: {args.test}\n")
        f.write(f"Test Samples: {len(y_test)}\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"MAE: {mae:.4f} Mbps\n")
        f.write(f"RMSE: {rmse:.4f} Mbps\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        f.write(f"R²: {r2:.4f}\n\n")
        f.write(f"Data Statistics:\n")
        f.write(f"Actual - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}\n")
        f.write(f"Predicted - Mean: {yhat.mean():.2f}, Std: {yhat.std():.2f}\n")
    
    print(f'Detailed metrics saved to {results_file}')

if __name__ == '__main__':
    main()
