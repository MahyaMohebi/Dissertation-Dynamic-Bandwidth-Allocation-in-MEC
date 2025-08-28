#!/usr/bin/env python3
"""Compute QoE per run.
We heavily penalize rebuffering; if no rebuffer logs exist yet, penalty=0 and QoE reduces to bitrate + smoothness.
Outputs per-run QoE CSV.
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path

def qoe_from_logs(df):
    # Decision rows: bitrate_kbps, seg_index
    dec = df[df['type']=='decision'].copy()
    if dec.empty: return None
    dec.sort_values(['seg_index'], inplace=True)
    br = dec['bitrate_kbps'].values.astype(float)
    # bitrate utility (log form)
    util = np.log1p(br/100.0).mean()
    # smoothness penalty: mean absolute delta between segments
    smooth = np.mean(np.abs(np.diff(br))) / 1000.0
    # rebuffer penalty (if present)
    reb = df[df['type']=='rebuffer']['duration_s'].sum() if 'duration_s' in df.columns else 0.0
    # QoE = utility - alpha*rebuffer - beta*smoothness
    alpha = 4.0   # strong penalty on rebuffering
    beta = 0.5
    qoe = util - alpha*reb - beta*smooth
    return float(qoe)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='results/logs')
    ap.add_argument('--out', default='results/qoe/summary.csv')
    args = ap.parse_args()

    log_dir = Path(args.logs)
    rows = []
    for p in log_dir.glob('*_decisions.csv'):
        df = pd.read_csv(p)
        q = qoe_from_logs(df)
        rows.append({'run_id': p.stem.replace('_decisions',''), 'qoe': q})
    out = pd.DataFrame(rows).sort_values('qoe', ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
