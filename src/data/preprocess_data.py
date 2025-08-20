#!/usr/bin/env python
import numpy as np, pandas as pd, argparse, pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

FEATURES = ["throughput_mbps","rtt_ms","loss_rate","jitter_ms","users","cell_load"]
TARGET = "throughput_mbps"
WINDOW = 10

def load_traces(indir):
    frames = []
    skipped = []
    for p in sorted(Path(indir).glob("trace_*.csv")):
        df = pd.read_csv(p)
        # Normalize column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]
        required = set(FEATURES + [TARGET])
        if required.issubset(df.columns):
            frames.append(df)
        else:
            skipped.append(p.name)
    if not frames:
        raise FileNotFoundError(
            f"No valid traces found in {indir}. Expected columns {sorted(required)}. "
            f"Checked {len(skipped)} files; examples skipped: {skipped[:5]}"
        )
    if skipped:
        print(
            f"Skipping {len(skipped)} non-conforming files: "
            + ", ".join(skipped[:5])
            + (" ..." if len(skipped) > 5 else "")
        )
    return frames

def build_windows(frames):
    Xs, ys = [], []
    for df in frames:
        arr = df[FEATURES].values
        y = df[TARGET].shift(-1).values  # next-step throughput
        for i in range(len(df) - WINDOW - 1):
            Xs.append(arr[i:i+WINDOW, :])
            ys.append(y[i+WINDOW-1])
    X = np.array(Xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/network_traces")
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    frames = load_traces(args.indir)
    X, y = build_windows(frames)
    scaler = StandardScaler()
    X_2d = X.reshape(-1, X.shape[-1])
    Xs = scaler.fit_transform(X_2d).reshape(X.shape)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    np.save(out/"X.npy", Xs); np.save(out/"y.npy", y)
    with open(out/"scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Preprocessing done:", Xs.shape, y.shape, "saved to", out)
