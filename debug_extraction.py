#!/usr/bin/env python3

import pandas as pd

# Read the CSV
df = pd.read_csv('datasets/norway_lte/local/Video Streaming/YouTube/YouTube14Agg.csv')
print(f'Original shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'Sample Time values: {df["Time"].head(3).tolist()}')
print(f'Sample Length values: {df["Length"].head(3).tolist()}')

# Convert Time to second granularity
df['time'] = pd.to_datetime(df['Time'], errors='coerce').dt.floor('S')
print(f'After datetime conversion: {df["time"].head(3).tolist()}')

# Remove rows with invalid times or missing Length
df_clean = df.dropna(subset=['time', 'Length'])
print(f'After dropna: {df_clean.shape}')

if not df_clean.empty:
    # Group by second and sum Length (bytes)
    throughput = df_clean.groupby('time')['Length'].sum()
    print(f'Throughput sample: {throughput.head(3)}')
    
    # Convert bytes to megabits
    mbps = throughput * 8 / 1_000_000
    print(f'Mbps sample: {mbps.head(3)}')
