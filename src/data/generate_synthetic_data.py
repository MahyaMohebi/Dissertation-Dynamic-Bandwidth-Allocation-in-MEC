#!/usr/bin/env python
import numpy as np, pandas as pd, argparse
from pathlib import Path

def ou_series(n, mu=10.0, theta=0.15, sigma=1.0, dt=0.1, x0=None):
    x = np.zeros(n, dtype=float)
    x[0] = x0 if x0 is not None else mu
    for i in range(1, n):
        dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        x[i] = max(0.1, x[i-1] + dx)
    return x

def make_trace(T=300.0, step=0.1, seed=None):
    rng = np.random.default_rng(seed)
    n = int(T/step)
    t = np.arange(n) * step
    throughput = ou_series(n, mu=rng.uniform(5, 25), theta=rng.uniform(0.05,0.25),
                           sigma=rng.uniform(0.5,2.0), dt=step, x0=rng.uniform(3,20))
    rtt = np.clip(ou_series(n, mu=rng.uniform(15,60), theta=0.2, sigma=5.0, dt=step), 5, 200)
    loss = np.clip(np.abs(rng.normal(0.01, 0.01, n)), 0, 0.2)
    jitter = np.clip(np.abs(rng.normal(5, 3, n)), 0, 100)
    users = np.clip((throughput.mean()/2 + rng.normal(10, 3, n)).astype(int), 1, 200)
    cell_load = np.clip(throughput/throughput.max() + rng.normal(0,0.05,n), 0, 1)
    df = pd.DataFrame({
        "time_s": t,
        "throughput_mbps": throughput,
        "rtt_ms": rtt,
        "loss_rate": loss,
        "jitter_ms": jitter,
        "users": users,
        "cell_load": cell_load,
    })
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=122, help="Total number of traces to have")
    ap.add_argument("--outdir", type=str, default="data/network_traces", help="Output directory")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("trace_*.csv"))
    to_make = max(0, args.count - len(existing))
    start_idx = len(existing)
    for i in range(to_make):
        idx = start_idx + i
        df = make_trace(seed=idx)
        df.to_csv(out / f"trace_{idx:03d}.csv", index=False)
    print(f"Generated {to_make} traces. Total now: {len(list(out.glob('trace_*.csv')))}")
