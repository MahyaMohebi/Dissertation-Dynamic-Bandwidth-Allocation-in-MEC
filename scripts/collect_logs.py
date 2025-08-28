wsl bash -lc "cd /mnt/c/KF7029/video-qoe-mec && python scripts/compute_qoe.py"#!/usr/bin/env python3
"""Collect per-run CSV logs into tidy dataframes.
Currently expects *decisions* logs (<run_id>_decisions.csv).
"""
import pandas as pd
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='results/logs')
    ap.add_argument('--out', default='results/qoe/decisions_merged.csv')
    args = ap.parse_args()

    log_dir = Path(args.logs)
    rows = []
    for p in log_dir.glob('*_decisions.csv'):
        try:
            df = pd.read_csv(p)
            df['run_id'] = p.stem.replace('_decisions','')
            rows.append(df)
        except Exception as e:
            print('skip', p, e)
    if not rows:
        print('No decision logs found under', log_dir); return
    merged = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print('Wrote', args.out, merged.shape)

if __name__ == '__main__':
    main()
