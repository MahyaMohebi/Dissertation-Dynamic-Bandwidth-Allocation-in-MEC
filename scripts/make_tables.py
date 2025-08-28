#!/usr/bin/env python3
"""Create CSV + LaTeX summary tables from QoE results."""
import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default='results/qoe/summary.csv')
    ap.add_argument('--out_csv', default='results/tables/summary_by_abr_cache.csv')
    ap.add_argument('--out_tex', default='results/tables/summary_by_abr_cache.tex')
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    meta = df['run_id'].str.extract(r'_(BOLA|MPC|Pensive)_(lru|nocache)$')
    df['abr'] = meta[0]; df['cache'] = meta[1]
    tbl = df.groupby(['abr','cache'])['qoe'].agg(['mean','std','count']).reset_index()
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(args.out_csv, index=False)
    # Simple LaTeX
    with open(args.out_tex, 'w', encoding='utf-8') as f:
        f.write(tbl.to_latex(index=False, float_format='%.3f'))
    print('Wrote', args.out_csv, 'and', args.out_tex)

if __name__ == '__main__':
    main()
