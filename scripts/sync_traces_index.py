#!/usr/bin/env python3
"""Copy datasets/traceset_norway.csv to ns3/configs/traceset_norway.csv."""
from pathlib import Path
src = Path('datasets/traceset_norway.csv')
dst = Path('ns3/configs/traceset_norway.csv')
if src.exists():
    dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
    print('Synced', src, '->', dst)
else:
    print('Source not found:', src)
