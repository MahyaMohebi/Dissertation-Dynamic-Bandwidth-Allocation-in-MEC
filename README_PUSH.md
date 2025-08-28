Push instructions for MahyaMohebi/Dissertation-Dynamic-Bandwidth-Allocation-in-MEC

This file contains safe, repeatable steps to create a GitHub repository and push the project content from local path `/mnt/c/Users/mahya/video-qoe-mec` to the remote repository `https://github.com/MahyaMohebi/Dissertation-Dynamic-Bandwidth-Allocation-in-MEC.git`.

Important: these steps do not include build artifacts, the virtual environment, or large external packages. They only commit scripts, docs, results, and source code (excluding ignored paths).

1) Preview what will be added (WSL/Bash -- recommended):

```bash
cd /mnt/c/Users/mahya/video-qoe-mec
# show gitignore effective list
git status --porcelain
# preview files to commit (exclude ignored)
git add --all --dry-run
```

2) Initialize a new git repo (if not already initialized) and set remote:

WSL / Bash
```bash
cd /mnt/c/Users/mahya/video-qoe-mec
git init
git remote add origin https://github.com/MahyaMohebi/Dissertation-Dynamic-Bandwidth-Allocation-in-MEC.git
git fetch origin || true
```

PowerShell (Windows)
```powershell
Set-Location -Path C:\Users\mahya\video-qoe-mec
git init
git remote add origin https://github.com/MahyaMohebi/Dissertation-Dynamic-Bandwidth-Allocation-in-MEC.git
git fetch origin
```

3) Add and commit only the tracked files (scripts, docs, results):

```bash
# add all files except those ignored by .gitignore
git add .
# optionally check staged files
git status --short
# commit
git commit -m "Initial push: scripts, docs, results (exclude venv/build/checkpoints)"
```

4) Push to GitHub (create main branch if not present):

```bash
# create main branch locally and push
git branch -M main
git push -u origin main
```

If your remote already has a history and you want to replace it (force push), run:

```bash
# WARNING: force push will overwrite remote history
git push -u origin main --force
```

5) Verify on GitHub.

Notes and safety
- The `.gitignore` added at repository root excludes `.venv`, `ns3/build`, `ml/models/checkpoints`, `external/onnxruntime` and `datasets/` to avoid committing large binaries and sensitive environment files.
- If you want to also exclude `results/` remove it from `.gitignore` (currently results are included). If results are large, consider adding `results/raw/` to `.gitignore` and only commit summary tables/figures.
- If you need help with authentication (PAT vs SSH) let me know and I can show the recommended commands.

If you want I can also create a single commit containing only the files you specified by scripting the git add of selected paths, or generate the exact git commands you should run in PowerShell. Tell me which you prefer.
