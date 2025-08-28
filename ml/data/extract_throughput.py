#!/usr/bin/env python3

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Read the CSV
    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]
    required = ['Time', 'Length']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: Input file {args.input} missing required columns: {missing}. Found columns: {df.columns.tolist()}")
        exit(1)

    # Convert Time to second granularity
    df['time'] = pd.to_datetime(df['Time'], errors='coerce').dt.floor('S')
    # Remove rows with invalid times or missing Length
    df = df.dropna(subset=['time', 'Length'])
    if df.empty:
        print(f"WARNING: All rows dropped due to missing Time or Length in {args.input}")
        pd.DataFrame(columns=['time','mbps']).to_csv(args.output, index=False)
        return
    # Group by second and sum Length (bytes)
    throughput = df.groupby('time')['Length'].sum()
    # Convert bytes to megabits
    mbps = throughput * 8 / 1_000_000
    out_df = mbps.reset_index().rename(columns={'Length': 'mbps'})
    out_df.columns = ['time', 'mbps']
    out_df.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
