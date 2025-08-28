param(
  [string]$BuildType = "Release",
  [string]$Abr = "BOLA",
  [string]$Cache = "lru",
  [string]$RunId = "sanity",
  [string]$ResultsDir = "results/logs"
)
$ErrorActionPreference = "Stop"
$root = Get-Location

# Configure & build
$buildDir = Join-Path $root "ns3\build"
if (!(Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
cmake -S ns3 -B $buildDir -DCMAKE_BUILD_TYPE=$BuildType
cmake --build $buildDir --config $BuildType --target run_scenario

# Ensure results dir
if (!(Test-Path $ResultsDir)) { New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null }

# Run
$exe = Join-Path $buildDir "run_scenario.exe"
if (!(Test-Path $exe)) { $exe = Join-Path $buildDir "run_scenario"; }
& $exe --abr $Abr --cache_policy $Cache --results_dir $ResultsDir --run_id $RunId
