#!/usr/bin/env python3
"""Create (X, y) windows from processed CSV traces.
Each CSV must have columns: time,mbps.

Usage:
  python ml/data/window_maker.py --list ml/data/processed/train.txt --out ml/data/processed/train_windows.npz --seq 20 --horizon 1
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path

def build_windows(files, seq, horizon):
    Xs, ys = [], []
    for f in files:
        df = pd.read_csv(f)
        v = df['mbps'].values.astype('float32')
        for i in range(len(v) - seq - horizon + 1):
            Xs.append(v[i:i+seq])
            ys.append(v[i+seq+horizon-1])
    X = np.array(Xs, dtype='float32')[..., None]  # (N, seq, 1)
    y = np.array(ys, dtype='float32')
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seq', type=int, default=20)
    ap.add_argument('--horizon', type=int, default=1)
    args = ap.parse_args()

    files = [Path(p) for p in Path(args.list).read_text(encoding='utf-8').splitlines() if p.strip()]
    X, y = build_windows(files, args.seq, args.horizon)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    print(f'Saved {args.out}: X{X.shape}, y{y.shape}')

if __name__ == '__main__':
    main()
