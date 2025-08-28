#!/usr/bin/env python3
"""Make CDF of QoE, boxplots of bitrate/rebuffering (if available), and a simple line plot.
Note: uses matplotlib only.
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def cdf_plot(series, out_path, title):
    x = np.sort(series)
    y = np.arange(1, len(x)+1)/len(x)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('QoE')
    plt.ylabel('CDF')
    plt.title(title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=120)

def box_plot(df, col, by, out_path, title):
    plt.figure()
    df.boxplot(column=col, by=by)
    plt.suptitle('')
    plt.title(title)
    plt.savefig(out_path, bbox_inches='tight', dpi=120)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default='results/qoe/summary.csv')
    ap.add_argument('--logs_merged', default='results/qoe/decisions_merged.csv')
    args = ap.parse_args()

    summ = pd.read_csv(args.summary)
    cdf_plot(summ['qoe'].values, 'results/figures/qoe_cdf.png', 'QoE CDF')

    # If we have decision logs merged, derive avg bitrate per run for a boxplot
    if Path(args.logs_merged).exists():
        dm = pd.read_csv(args.logs_merged)
        grp = dm[dm['type']=='decision'].groupby('run_id')['bitrate_kbps'].mean().reset_index()
        # parse ABR from run_id
        meta = grp['run_id'].str.extract(r'_(BOLA|MPC|Pensive)_(lru|nocache)$')
        grp['abr'] = meta[0]; grp['cache'] = meta[1]
        box_plot(grp, 'bitrate_kbps', 'abr', 'results/figures/bitrate_box_by_abr.png', 'Avg Bitrate by ABR')
        box_plot(grp, 'bitrate_kbps', 'cache', 'results/figures/bitrate_box_by_cache.png', 'Avg Bitrate by Cache')

    print('Saved figures under results/figures')

if __name__ == '__main__':
    main()
