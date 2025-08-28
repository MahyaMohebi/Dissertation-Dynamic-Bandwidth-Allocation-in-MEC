#!/usr/bin/env python3
import argparse, pandas as pd, re
from pathlib import Path
from scipy import stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default='results/qoe/summary.csv')
    ap.add_argument('--out', default='results/tables/stats_tests.csv')
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    parts = df['run_id'].str.extract(r'^(t\d+)_(.+)_(BOLA|MPC|Pensive)_(lru|nocache)$')
    df[['tid','trace','abr','cache']] = parts
    rows = []
    pairs = [('BOLA','MPC'), ('BOLA','Pensive'), ('MPC','Pensive')]
    for cache in ['lru','nocache']:
        for a,b in pairs:
            da = df[(df['abr']==a) & (df['cache']==cache)].sort_values('tid')
            db = df[(df['abr']==b) & (df['cache']==cache)].sort_values('tid')
            merged = da[['tid','qoe']].merge(db[['tid','qoe']], on='tid', suffixes=(f'_{a}', f'_{b}'))
            if merged.empty: continue
            t = stats.ttest_rel(merged[f'qoe_{a}'], merged[f'qoe_{b}'])
            try:
                w = stats.wilcoxon(merged[f'qoe_{a}'], merged[f'qoe_{b}'])
            except ValueError:
                w = type('obj', (), {'statistic': float('nan'), 'pvalue': float('nan')})
            rows.append({'cache':cache,'pair':f'{a} vs {b}','t_stat':t.statistic,'t_p':t.pvalue,'w_stat':w.statistic,'w_p':w.pvalue,'n':len(merged)})
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
