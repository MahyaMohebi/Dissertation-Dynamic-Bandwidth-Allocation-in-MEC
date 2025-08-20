This folder is excluded from Git to keep the repository size small and avoid distributing raw or proprietary datasets.

Expected contents (generated or provided locally):
- network_traces/ (adapted ns-3 CSVs; contains trace_*.csv and trace_index.csv)
- processed/ (X.npy, y.npy, scaler.pkl) — produced by `src/data/preprocess_data.py`
- raw/ (original dumps, if any)

To regenerate:
1) Standardize raw CSVs to `data/network_traces/` using the adapter.
2) Run preprocessing to produce processed arrays and scaler.
See RUNBOOK.md for exact commands.
