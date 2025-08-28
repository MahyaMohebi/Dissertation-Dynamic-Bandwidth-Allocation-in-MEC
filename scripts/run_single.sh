#!/usr/bin/env bash
set -euo pipefail

# Default values
BUILD_TYPE="Release"
ABR="BOLA"
CACHE="lru"
RUN_ID="sanity"
RESULTS_DIR="results/logs"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -Abr)
      ABR="$2"
      shift 2
      ;;
    -Cache)
      CACHE="$2"
      shift 2
      ;;
    -RunId)
      RUN_ID="$2"
      shift 2
      ;;
    -BuildType)
      BUILD_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

cmake -S ns3 -B ns3/build -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build ns3/build --target run_scenario

mkdir -p "$RESULTS_DIR"
EXE="ns3/build/run_scenario"
[ -x "$EXE" ] || EXE="ns3/build/run_scenario.exe"
"$EXE" --abr "$ABR" --cache_policy "$CACHE" --results_dir "$RESULTS_DIR" --run_id "$RUN_ID"
