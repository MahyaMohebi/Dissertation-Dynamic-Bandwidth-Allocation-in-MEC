# VIDEO-QOE-MEC — One‑Flow Full Guide (QoE‑First, Rebuffering-Minimization, WSL/Bash Edition)

> Modeled after your “One‑Flow Guide” format, but tailored to **video‑qoe‑mec** with **ns‑3 skeleton + LSTM predictor + edge cache**, **WSL/Bash-first workflow**, and your **local/global** dataset layout. This is copy‑paste runnable and lists terminal, ## 11) What's next (roadmap aligning to your objectives)aths, and realistic time estimates for each step. 

---

## 0) Repository Layout (must match)

```
video-qoe-mec/
├─ README.md
├─ .gitignore
├─ env/
│  ├─ requirements.txt
│  ├─ environment.yml
│  └─ setup.ps1
├─ external/
│  ├─ 5g-lena/
│  └─ onnxruntime/
├─ datasets/
│  ├─ README.md
│  ├─ ingest_dataset.py
│  └─ norway_lte/
│     ├─ local/      # ← you will paste here (your dataset)
│     └─ global/     # ← you will paste here (your dataset)
├─ ns3/
│  ├─ CMakeLists.txt
│  ├─ include/...    # headers (already created)
│  ├─ src/...        # sources (already created)
│  └─ configs/
│     ├─ default.yaml
│     ├─ sweep.yaml
│     ├─ traceset_norway.csv
│     └─ video_profiles.json
├─ ml/
│  ├─ data/
│  │  ├─ prepare_traces.py
│  │  └─ window_maker.py
│  ├─ train_lstm.py
│  ├─ evaluate_lstm.py
│  ├─ export_onnx.py
│  └─ models/
│     ├─ checkpoints/
│     └─ final/
│        ├─ bandwidth_lstm.onnx
│        └─ README.md
├─ scripts/
│  ├─ run_single.ps1
│  ├─ run_single.sh
│  ├─ run_experiments.py
│  ├─ collect_logs.py
│  ├─ compute_qoe.py
│  ├─ stats_tests.py
│  ├─ plots.py
│  └─ make_tables.py
├─ analysis/
│  └─ analysis.ipynb
├─ results/
│  ├─ logs/
│  ├─ predictions/
│  ├─ qoe/
│  ├─ figures/
│  └─ tables/
└─ docs/
   ├─ ethics/
   │  └─ Ethics form.pdf  # optional archive
   └─ dissertation/
      ├─ figures/
      └─ references.bib
```

> **Do not rename** folders above. All commands assume this layout.

---

## 1) Terminals, Paths, Roles, and Timing Assumptions

**OS**: Windows 10/11 (64‑bit) with **WSL2 (Ubuntu recommended)**  
**Primary terminal**: **WSL Bash** (Ubuntu, Debian, etc.)  
**CPU assumption for time estimates**: 4–8‑core laptop CPU, 16 GB RAM, no CUDA.

**Path variables we will use:**

- **Repo root (WSL/Bash):**
  ```bash
  export ROOT="/mnt/c/Users/<YOU>/video-qoe-mec"
  cd "$ROOT"
  ```

**Terminal roles:**

- **[ENV]** – Environment setup & Python scripts (Bash/WSL)
- **[BUILD]** – CMake configure/build for the C++ runner (Bash/WSL)
- **[ANALYZE]** – Plotting / making tables / notebook (Bash/WSL)

> You can use **one WSL window** for all roles; role tags are for clarity.

---

## 2) Prerequisites (install once) — ~10–20 min

**Software (WSL/Ubuntu):**
1. **Python 3.8 (64‑bit)** (required for TF 2.10). Install via apt or pyenv.  
2. **Git** (recommended).  
3. **CMake 3.16+** (`sudo apt install cmake`).  
4. **g++/build-essential** (`sudo apt install build-essential`).

**Install (Bash/WSL):**
```bash
sudo apt update
sudo apt install python3.8 python3.8-venv python3-pip git cmake build-essential
```

**Verify (Bash/WSL) — [ENV], 1–2 min:**
```bash
python3.8 --version
cmake --version
git --version
```
All should print versions without errors.

---

## 3) Create venv and install Python deps — [ENV], ~2–6 min

```bash
cd "$ROOT"
# Creates .venv, installs pinned deps (TF 2.10, onnxruntime, numpy, pandas, etc.)
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r env/requirements.txt

# Quick check:
python -c "import tensorflow as tf; print('python env ok', tf.__version__)"
```

> **ONNX and TensorFlow are not compatible in the same venv due to protobuf version conflicts.**
> If you need ONNX export, create a separate venv and install ONNX there:
> ```bash
> python3.8 -m venv onnx-venv
> source onnx-venv/bin/activate
> pip install --upgrade pip
> pip install onnx==1.13.1 onnxruntime
> # (export your model)
> deactivate
> source .venv/bin/activate  # return to main venv
> ```

**Re‑activate later (new terminal):**
```bash
cd "$ROOT"
source .venv/bin/activate
```

---

## 4) Process Dataset for ML Pipeline — [ENV], ~5–15 min

Before running the ML pipeline, you must process the raw dataset to extract throughput and prepare it for training. Follow these steps:

### Step 4.1: Extract Throughput from Raw Logs

The raw CSV files in `datasets/norway_lte/{local,global}` contain multi-column logs (e.g., `Time,Source,Destination,...,Length,...`). You need to convert them into per-second throughput CSVs (`time,mbps`).

1. **Run the throughput extraction script for all files**:
   ```bash
   # Run this from the repo root (WSL/Bash):
   for SRC in datasets/norway_lte/{local,global}/*/*/*.csv; do 
     OUT="ml/data/processed/$(echo "$SRC" | cut -d/ -f3- | sed 's/^Local/local/;s/^Global/global/')"
     mkdir -p "$(dirname "$OUT")"
     python ml/data/extract_throughput.py --input "$SRC" --output "$OUT"
   done
   ```

2. **Verify the output**:
   - Check that the processed files are not empty and contain valid `time,mbps` data.
   - Preview a sample file:
     ```bash
     head -n 10 ml/data/processed/local/Video\ Streaming/YouTube/YouTube14Agg.csv
     ```

### Step 4.2: Create Train/Val/Test Splits

After extracting throughput, create train/val/test splits from the processed files:

1. **Create splits from processed files**:
   ```bash
   python -c "
import os, random
files = []
for root, dirs, filenames in os.walk('ml/data/processed'):
    for filename in filenames:
        if filename.endswith('.csv'):
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r') as f:
                    if len(f.readlines()) > 1:
                        files.append(filepath)
            except: pass
random.seed(42); random.shuffle(files)
n = len(files); train_end = int(0.8 * n); val_end = int(0.9 * n)
splits = [files[:train_end], files[train_end:val_end], files[val_end:]]
for i, (name, split) in enumerate(zip(['train', 'val', 'test'], splits)):
    with open(f'ml/data/processed/{name}.txt', 'w') as f:
        f.write('\n'.join(split) + '\n')
print(f'Created splits: train={len(splits[0])}, val={len(splits[1])}, test={len(splits[2])}')
"
   ```

2. **Verify the splits**:
   - Ensure the splits are created:
     ```bash
     ls ml/data/processed/train.txt ml/data/processed/val.txt ml/data/processed/test.txt
     ```

### Step 4.3: Create Training Windows

Generate sliding windows of data for the LSTM model:

1. **Run the window maker script**:
   ```bash
   python ml/data/window_maker.py --list ml/data/processed/train.txt --out ml/data/processed/train_windows.npz --seq 20 --horizon 1
   python ml/data/window_maker.py --list ml/data/processed/val.txt   --out ml/data/processed/val_windows.npz   --seq 20 --horizon 1
   python ml/data/window_maker.py --list ml/data/processed/test.txt  --out ml/data/processed/test_windows.npz  --seq 20 --horizon 1
   ```

2. **Verify the output**:
   - Check the `.npz` files to ensure they contain valid data:
     ```bash
     python -c "import numpy as np; data = np.load('ml/data/processed/train_windows.npz'); print(data.files, data['X'].shape, data['y'].shape)"
     ```

If all steps are successful, proceed to Step 5.

## 5) Train LSTM and export ONNX — [ENV], 10–30 min CPU (dataset‑dependent)

```bash
# Train improved model with normalization and better architecture
python ml/train_lstm.py --train ml/data/processed/train_windows.npz --val ml/data/processed/val_windows.npz --out ml/models/checkpoints/model.h5 --normalize --epochs 50 --hidden 128 --lr 0.001 --verbose 1

# Additional options for fine-tuning:
--dropout 0.3 (dropout rate for regularization)
--batch 512 (batch size)  
--normalize (enable data normalization - recommended)

# Evaluate on test set + save a sanity plot in results/figures/
python ml/evaluate_lstm.py --model ml/models/checkpoints/model.h5 --test ml/data/processed/test_windows.npz --out results/figures/pred_vs_actual.png

# Export ONNX for C++ predictor (see ONNX note above)
python ml/export_onnx.py --model ml/models/checkpoints/model.h5 --out ml/models/final/bandwidth_lstm.onnx
```

> The C++ `OnnxPredictor` is currently a stub; we’ll wire ONNX Runtime later. You can already use the **StubPredictor** for ablations.

---

## 6) Build the C++ runner — [BUILD], 1–3 min

The current C++ code is **self‑contained** (no ns‑3 linkage yet). You just need MSVC and CMake.

```bash
cd "$ROOT"
source .venv/bin/activate   # optional; not required for CMake

# Configure + build (uses ns3/CMakeLists.txt)
bash scripts/run_single.sh -Abr BOLA -Cache lru -RunId sanity
```
Under the hood this:
- Configures `ns3/build` with CMake (Release),
- Builds the `run_scenario` executable,
- Runs a **sanity** case (5 segments, decision logs only).

**Artifacts:**
```
ns3/build/run_scenario[.exe]
results/logs/sanity_decisions.csv
```

---

## 7) Single‑run with your own settings — [BUILD], <1 min

```bash
# Example: MPC, no cache
bash scripts/run_single.sh -Abr MPC -Cache nocache -RunId mpc_nocache
# Output → results/logs/mpc_nocache_decisions.csv
```

Advanced CLI (if launching manually):
```bash
ns3/build/run_scenario \
  --abr BOLA --cache_policy lru \
  --results_dir results/logs --run_id demo \
  --segment_duration_s 2.0 --buffer_target_s 20.0 \
  --ladder_kbps "300,750,1200,2500,4000" \
  --trace_csv datasets/norway_lte/local/YOUR_FILE.csv
```

---

## 8) Sweep (30 traces × 3 ABR × 2 caches) — [BUILD], ~10–60 min

1) **Sync** the dataset index into `ns3/configs/`:
```bash
python scripts/sync_traces_index.py
```
2) **Run the sweep**:
```bash
python scripts/run_experiments.py
```
This will build `run_scenario` if needed and then iterate:
```
t000_..._BOLA_lru, t001_..._BOLA_lru, ..., t029_..._Pensive_nocache
```
Outputs per‑run decision logs in `results/logs/`.

> Time depends on how many traces are present and your CPU. The current runner simulates **decisions only** (fast).

---

## 9) Collate logs & compute QoE — [ANALYZE], 1–3 min

```bash
# Merge *_decisions.csv files → results/qoe/decisions_merged.csv
python scripts/collect_logs.py

# Compute QoE per run (heavy penalty on rebuffering, smoothness penalty):
python scripts/compute_qoe.py    # → results/qoe/summary.csv
```

**Note:** Until we add download/rebuffer logs, QoE reduces to bitrate utility − smoothness penalty. The penalty on **rebuffering** is already configured (α=4.0) for when those events are logged.

---

## 10) Plots, tables, notebook — [ANALYZE], 1–3 min

```bash
# Figures (CDF of QoE, plus bitrate boxplots if merged logs are available)
python scripts/plots.py

# Tables (CSV + LaTeX by ABR × cache)
python scripts/make_tables.py

# Optional: open the notebook
jupyter lab  # then open analysis/analysis.ipynb
```
Artifacts land in `results/figures/` and `results/tables/`.

---

## 12) What’s next (roadmap aligning to your objectives)

1. **Wire ONNX Runtime** in `ns3/src/predict/onnx_predictor.cc` to load `ml/models/final/bandwidth_lstm.onnx`.  
2. **Integrate ns‑3 + 5G‑LENA**: convert the runner into a true ns‑3 scenario (NR stack, traffic over the link).  
3. **Emit download and rebuffer events** in `DashClient` to feed the QoE calculator (already supports large rebuffer penalties).  
4. **Ablations**: Use config toggles `enable_prediction` and `enable_cache` to isolate benefits, and extend `sweep.yaml` accordingly.  
5. **Stat tests** (`scripts/stats_tests.py`) once QoE is populated for all baselines.

---

## 12) Troubleshooting (WSL/Bash)

- **CMake can’t find the compiler:** Ensure CMake is installed and available in your PATH.  
- **Build succeeds but no exe:** Check `ns3/build` for `run_scenario`. If missing, re‑configure:
  ```bash
  cmake -S ns3 -B ns3/build -DCMAKE_BUILD_TYPE=Release
  cmake --build ns3/build --config Release --target run_scenario
  ```
- **Python import errors:** Ensure venv is active (`source .venv/bin/activate`).  
- **No traces found in sweep:** Run `python scripts/sync_traces_index.py` after ingesting your dataset.  
- **Permissions:** If Bash blocks scripts, ensure you have execution permissions:
  ```bash
  chmod +x ./scripts/*.sh
  ```

---

## 13) Verification Checklist (copy/paste)

- [ ] `.venv` created, venv activated, `python env ok` printed.  
- [ ] `datasets/norway_lte/{local,global}` populated; `datasets/traceset_norway.csv` exists.  
- [ ] `ns3/build/run_scenario` exists (after first run).  
- [ ] `results/logs/*_decisions.csv` present.  
- [ ] `results/qoe/summary.csv` generated.  
- [ ] `results/figures/*.png` and `results/tables/*.csv|*.tex` created.  

---

## 14) Time Budget Summary (typical laptop)

- Prerequisites install: **10–20 min** (one‑time).  
- Venv setup: **2–6 min**.  
- Ingest dataset: **1–10 min** (size‑dependent).  
- Process traces + windows: **5–15 min**.  
- Train + export ONNX: **10–30 min**.  
- Build runner: **1–3 min**.  
- Single run: **<1 min**.  
- Full sweep (180 runs on stub): **10–60 min**.

> As we add real ns‑3 networking and downloads, run times will increase accordingly.

---

## 15) Canonical Run Order (TL;DR)

```bash
# 0) open WSL/Bash in repo root
export ROOT="/mnt/c/Users/<YOU>/video-qoe-mec"
cd "$ROOT"

# 1) env
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r env/requirements.txt



# 2) dataset
# If you have already pasted your dataset into datasets/norway_lte/{local,global}, generate the index file:
python ingest_dataset.py --source datasets/norway_lte
python ingest_dataset.py --verify
# Otherwise, to ingest from an external folder:
# python ingest_dataset.py --source "/mnt/d/data/norway"
python scripts/sync_traces_index.py

# 4) extract throughput + create splits
for SRC in datasets/norway_lte/{local,global}/*/*/*.csv; do 
  OUT="ml/data/processed/$(echo "$SRC" | cut -d/ -f3- | sed 's/^Local/local/;s/^Global/global/')"
  mkdir -p "$(dirname "$OUT")"
  python ml/data/extract_throughput.py --input "$SRC" --output "$OUT"
done

python -c "
import os, random
files = []
for root, dirs, filenames in os.walk('ml/data/processed'):
    for filename in filenames:
        if filename.endswith('.csv'):
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r') as f:
                    if len(f.readlines()) > 1:
                        files.append(filepath)
            except: pass
random.seed(42); random.shuffle(files)
n = len(files); train_end = int(0.8 * n); val_end = int(0.9 * n)
splits = [files[:train_end], files[train_end:val_end], files[val_end:]]
for i, (name, split) in enumerate(zip(['train', 'val', 'test'], splits)):
    with open(f'ml/data/processed/{name}.txt', 'w') as f:
        f.write('\n'.join(split) + '\n')
print(f'Created splits: train={len(splits[0])}, val={len(splits[1])}, test={len(splits[2])}')
"

# 5) create windows
python ml/data/window_maker.py --list ml/data/processed/train.txt --out ml/data/processed/train_windows.npz --seq 20 --horizon 1
python ml/data/window_maker.py --list ml/data/processed/val.txt   --out ml/data/processed/val_windows.npz   --seq 20 --horizon 1
python ml/data/window_maker.py --list ml/data/processed/test.txt  --out ml/data/processed/test_windows.npz  --seq 20 --horizon 1

# 6) train
python ml/train_lstm.py --train ml/data/processed/train_windows.npz --val ml/data/processed/val_windows.npz --out ml/models/checkpoints/model.h5
python ml/export_onnx.py --model ml/models/checkpoints/model.h5 --out ml/models/final/bandwidth_lstm.onnx

# 3) build + single run
bash scripts/run_single.sh -Abr BOLA -Cache lru -RunId sanity

# 4) sweep
python scripts/run_experiments.py

# 5) analysis
python scripts/collect_logs.py
python scripts/compute_qoe.py
python scripts/plots.py
python scripts/make_tables.py
```

---

### Appendix: Where each result goes


- `results/logs/` – raw per‑run decision logs (and later, download/rebuffer logs)
- `results/qoe/` – per‑run QoE + merged logs
- `results/figures/` – plots (CDF, boxplots, pred‑vs‑actual)
- `results/tables/` – CSV and LaTeX summary tables

**End of Guide.**
