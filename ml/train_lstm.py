#!/usr/bin/env python3
"""Train an improved LSTM throughput predictor with normalization and better architecture.
Usage:
  python ml/train_lstm.py --train ml/data/processed/train_windows.npz --val ml/data/processed/val_windows.npz --out ml/models/checkpoints/model.h5
"""
import argparse, os
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

def build_model(seq=20, hidden=128, lr=1e-3, dropout=0.3):
    """Build improved LSTM model with normalization and better architecture"""
    i = tf.keras.Input(shape=(seq, 1))
    
    # Normalization layer
    x = tf.keras.layers.Lambda(lambda x: (x - 9.33) / 18.31)(i)  # z-score normalization
    
    # LSTM layers with dropout and regularization
    x = tf.keras.layers.LSTM(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LSTM(hidden//2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LSTM(hidden//4, dropout=dropout, recurrent_dropout=dropout)(x)
    
    # Dense layers with regularization
    x = tf.keras.layers.Dense(hidden//2, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(hidden//4, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    # Output layer (normalized space)
    o = tf.keras.layers.Dense(1)(x)
    
    # Denormalization layer
    o = tf.keras.layers.Lambda(lambda x: x * 18.31 + 9.33)(o)
    
    model = tf.keras.Model(i, o)
    
    # Compile with Huber loss (more robust to outliers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--out', default='ml/models/checkpoints/model.h5')
    ap.add_argument('--seq', type=int, default=20)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--normalize', action='store_true', default=True, help='Normalize data')
    ap.add_argument('--verbose', type=int, default=1, help='Training verbosity level')
    args = ap.parse_args()

    # Load data
    tr = np.load(args.train); va = np.load(args.val)
    Xtr, ytr = tr['X'], tr['y']
    Xva, yva = va['X'], va['y']
    
    print(f"Original data shapes: X_train={Xtr.shape}, y_train={ytr.shape}")
    print(f"Data ranges: X=[{Xtr.min():.2f}, {Xtr.max():.2f}], y=[{ytr.min():.2f}, {ytr.max():.2f}]")
    
    # Normalize data for better training
    if args.normalize:
        # Fit scaler on training data
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        # Reshape for scaling
        Xtr_reshaped = Xtr.reshape(-1, 1)
        Xtr_scaled = X_scaler.fit_transform(Xtr_reshaped).reshape(Xtr.shape)
        
        ytr_scaled = y_scaler.fit_transform(ytr.reshape(-1, 1)).flatten()
        
        # Transform validation data
        Xva_reshaped = Xva.reshape(-1, 1)
        Xva_scaled = X_scaler.transform(Xva_reshaped).reshape(Xva.shape)
        yva_scaled = y_scaler.transform(yva.reshape(-1, 1)).flatten()
        
        # Save scalers for inference
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        scaler_dir = os.path.dirname(args.out)
        joblib.dump(X_scaler, os.path.join(scaler_dir, 'X_scaler.pkl'))
        joblib.dump(y_scaler, os.path.join(scaler_dir, 'y_scaler.pkl'))
        
        print(f"Normalized data ranges: X=[{Xtr_scaled.min():.2f}, {Xtr_scaled.max():.2f}], y=[{ytr_scaled.min():.2f}, {ytr_scaled.max():.2f}]")
        
        Xtr, ytr = Xtr_scaled, ytr_scaled
        Xva, yva = Xva_scaled, yva_scaled

    # Build model
    model = build_model(seq=Xtr.shape[1], hidden=args.hidden, lr=args.lr, dropout=args.dropout)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.out, 
            save_best_only=True, 
            monitor='val_mae', 
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10, 
            restore_best_weights=True, 
            monitor='val_mae', 
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(os.path.dirname(args.out), 'training_log.csv')
        )
    ]
    
    # Train model
    print(f"\nStarting training with {len(Xtr)} samples...")
    history = model.fit(
        Xtr, ytr, 
        validation_data=(Xva, yva), 
        epochs=args.epochs, 
        batch_size=args.batch, 
        callbacks=callbacks,
        verbose=args.verbose
    )
    
    # Print final metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    
    print(f"\nFinal Training Metrics:")
    print(f"Train Loss: {final_train_loss:.4f}, Train MAE: {final_train_mae:.4f}")
    print(f"Val Loss: {final_val_loss:.4f}, Val MAE: {final_val_mae:.4f}")
    print(f'Best model saved to {args.out}')
    
    if args.normalize:
        print(f'Scalers saved to {scaler_dir}/')
        print("Remember to use scalers for inference!")

if __name__ == '__main__':
    main()
