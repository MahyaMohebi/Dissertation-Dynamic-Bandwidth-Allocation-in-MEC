# video-qoe-mec (bootstrap: env + datasets)

This pack gives you the **environment setup** and **dataset ingestion** pieces for Windows (PowerShell), aligned with ns-3 + 5G-LENA and a Python 3.8 + TensorFlow 2.10 toolchain.

> Focus: keep everything ready for **QoE-first experiments** (minimizing rebuffering) while you later plug in the ns-3 code and analysis scripts.

## Prerequisites (Windows)
- Python **3.8 (64-bit)** installed (use the `py` launcher).
- PowerShell 5+
- Git
- (Later for ns-3) Visual Studio Build Tools + CMake

## Quick start
1) Open PowerShell **in the repo root** and run:
```powershell
./env/setup.ps1
```
This creates `.venv/` and installs Python deps from `env/requirements.txt`. It also sets the env var `VIDEO_QOE_MEC_ROOT` to the repo path (user-level).

2) **Dataset ingestion** (your dataset has `local/` and `global/` folders):
- If your downloaded dataset lives at `D:\data\norway`, with `D:\data\norway\local\...` and `D:\data\norway\global\...`, run:
```powershell
# make sure the venv is active first:
. .\.venv\Scripts\Activate.ps1

python .\datasets\ingest_dataset.py --source "D:\data\norway"
```
This will copy the data into:
```
datasets/
└─ norway_lte/
   ├─ local/   # copied files
   └─ global/  # copied files
```
…and it will generate `datasets/traceset_norway.csv` listing all trace files (relative paths) for later experiments.

3) **Verify** the dataset layout anytime:
```powershell
python .\datasets\ingest_dataset.py --verify
```

## Notes
- We intentionally **ignore** large raw data in Git. The `.gitignore` keeps `datasets/` out of version control but tracks this README and the ingestion script.
- Next steps (in upcoming packs): ns-3 CMake skeleton, config templates, and analysis scaffolding.
