#!/usr/bin/env python3

import os
import random
from pathlib import Path

# Get all processed CSV files
files = []
for root, dirs, filenames in os.walk('ml/data/processed'):
    for filename in filenames:
        if filename.endswith('.csv'):
            filepath = os.path.join(root, filename)
            # Check if file has actual data (more than just header)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # Has header + data
                        files.append(filepath)
            except:
                pass

print(f'Found {len(files)} processed CSV files with data')

if len(files) == 0:
    print("No valid processed files found!")
    exit(1)

# Shuffle and split
random.seed(42)
random.shuffle(files)

n = len(files)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_files = files[:train_end]
val_files = files[train_end:val_end]
test_files = files[val_end:]

# Write splits
with open('ml/data/processed/train.txt', 'w') as f:
    for file in train_files:
        f.write(file + '\n')
        
with open('ml/data/processed/val.txt', 'w') as f:
    for file in val_files:
        f.write(file + '\n')
        
with open('ml/data/processed/test.txt', 'w') as f:
    for file in test_files:
        f.write(file + '\n')

print(f'Created splits: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}')
