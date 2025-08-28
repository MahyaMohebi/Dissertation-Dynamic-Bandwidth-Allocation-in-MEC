#!/usr/bin/env python3

"""
Ingest a dataset that has two subfolders: `local/` and `global/`.

Usage:
  # Copy from source root (contains local/ and global/) into this repo's datasets/
  python datasets/ingest_dataset.py --source "D:\data\norway"

  # Verify current datasets/ layout without copying
  python datasets/ingest_dataset.py --verify
"""
import argparse
import shutil
from pathlib import Path
import sys
import csv

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
DEST_ROOT = DATASETS_DIR / "norway_lte"
TRACES_INDEX = DATASETS_DIR / "traceset_norway.csv"

INCLUDE_EXT = {".csv", ".txt", ".tsv", ".json", ".log"}  # permissive; adjust if needed

def scan_dir(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            if p.suffix.lower() in INCLUDE_EXT or not INCLUDE_EXT:
                files.append(p)
    return files

def copy_tree(src_root: Path, dst_root: Path):
    count = 0
    for src in scan_dir(src_root):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        count += 1
    return count

def build_index(dst_root: Path, index_file: Path):
    # index relative paths **from datasets/** (so later configs can use them easily)
    rel_rows = []
    for subset in ["local", "global"]:
        subdir = dst_root / subset
        if not subdir.exists():
            continue
        for f in sorted(subdir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(DATASETS_DIR)  # e.g., norway_lte/local/trace1.csv
                rel_rows.append([str(rel).replace("\\", "/")])
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["relative_path"])
        writer.writerows(rel_rows)
    return len(rel_rows)

def verify_layout():
    problems = []
    if not DEST_ROOT.exists():
        problems.append(f"Missing folder: {DEST_ROOT}")
    else:
        for subset in ["local", "global"]:
            subdir = DEST_ROOT / subset
            if not subdir.exists():
                problems.append(f"Missing subfolder: {subdir}")
            else:
                files = list(subdir.rglob("*"))
                file_count = len([f for f in files if f.is_file()])
                if file_count == 0:
                    problems.append(f"No files found in {subdir}")
    if not TRACES_INDEX.exists():
        problems.append(f"Missing index file: {TRACES_INDEX}")
    summary = "OK" if not problems else "PROBLEMS:\n- " + "\n- ".join(problems)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="Path to dataset root that contains 'local/' and 'global/'")
    ap.add_argument("--dest", type=Path, default=DEST_ROOT, help="Destination root (default: datasets/norway_lte)")
    ap.add_argument("--verify", action="store_true", help="Only verify current layout without copying")
    args = ap.parse_args()

    if args.verify:
        print(verify_layout())
        return 0

    if not args.source or not args.source.exists():
        print("ERROR: --source is required and must exist when not using --verify", file=sys.stderr)
        return 2

    local_src = args.source / "local"
    global_src = args.source / "global"
    if not local_src.exists() or not global_src.exists():
        print(f"ERROR: Expected 'local/' and 'global/' under: {args.source}", file=sys.stderr)
        return 2

    # Copy
    local_dst = args.dest / "local"
    global_dst = args.dest / "global"

    print(f"Copying LOCAL traces from: {local_src}")
    lc = copy_tree(local_src, local_dst)
    print(f"  Copied {lc} files -> {local_dst}")

    print(f"Copying GLOBAL traces from: {global_src}")
    gc = copy_tree(global_src, global_dst)
    print(f"  Copied {gc} files -> {global_dst}")

    # Build index CSV
    idx_count = build_index(args.dest, TRACES_INDEX)
    print(f"Wrote index: {TRACES_INDEX}  ({idx_count} rows)")

    # Final verify
    print("\nVerification:")
    print(verify_layout())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
