#!/usr/bin/env python3
"""Parse & normalize raw traces under datasets/norway_lte/{local,global} into CSV files
with columns: time,mbps, and produce train/val/test splits listing file paths.

Usage:
  python ml/data/prepare_traces.py --source datasets/norway_lte --out ml/data/processed
"""
import argparse, os
from pathlib import Path
import csv

def normalize_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Try read two-column time,mbps; otherwise treat each line as mbps at 1 Hz
    records = []
    with open(src, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            try:
                if len(parts) >= 2:
                    t = float(parts[0]); v = float(parts[1])
                else:
                    v = float(parts[0]); t = len(records)
                records.append((t, v))
            except:
                # skip header/bad rows
                continue
    # Normalize negative values and obvious outliers
    cleaned = [(t, max(0.0, v)) for (t, v) in records if v == v]  # drop NaNs
    with open(dst, 'w', newline='', encoding='utf-8') as out:
        w = csv.writer(out); w.writerow(['time','mbps']); w.writerows(cleaned)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='datasets/norway_lte')
    ap.add_argument('--out', default='ml/data/processed')
    ap.add_argument('--split', default='0.8,0.1,0.1', help='train,val,test fractions')
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_files = []
    for sub in ['local', 'global']:
        d = src / sub
        if d.exists():
            for p in d.rglob('*'):
                if p.is_file():
                    all_files.append(p)

    all_files = sorted(all_files)
    proc_files = []
    for i, f in enumerate(all_files):
        rel = f.relative_to(src)
        dst = out / rel.with_suffix('.csv')
        try:
            normalize_file(f, dst)
            proc_files.append(dst)
        except Exception as e:
            print('WARNING: failed to parse', f, e)

    # write splits
    tr, va, te = [float(x) for x in args.split.split(',')]
    n = len(proc_files)
    n_tr = int(tr*n); n_va = int(va*n); n_te = n - n_tr - n_va
    train = proc_files[:n_tr]
    val = proc_files[n_tr:n_tr+n_va]
    test = proc_files[n_tr+n_va:]

    (out/'train.txt').write_text('\n'.join(str(p) for p in train), encoding='utf-8')
    (out/'val.txt').write_text('\n'.join(str(p) for p in val), encoding='utf-8')
    (out/'test.txt').write_text('\n'.join(str(p) for p in test), encoding='utf-8')

    print(f'Processed {len(proc_files)} files. Splits: train={len(train)}, val={len(val)}, test={len(test)}')
    print(f'Lists written to: {out}/train.txt, val.txt, test.txt')

if __name__ == '__main__':
    main()
