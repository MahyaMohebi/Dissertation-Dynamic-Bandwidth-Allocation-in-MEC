#!/usr/bin/env python3
"""Run the 30×3×2 sweep: traces × ABR × cache policy.
Reads ns3/configs/sweep.yaml and traces list, then calls the run_scenario executable.
"""
import subprocess, os, sys, yaml
from pathlib import Path

def main():
    cfg = yaml.safe_load(Path('ns3/configs/sweep.yaml').read_text(encoding='utf-8'))
    traces_index = Path(cfg['traces_index'])
    if not traces_index.exists():
        print('ERROR: traces index not found at', traces_index, file=sys.stderr)
        print('Tip: copy datasets/traceset_norway.csv -> ns3/configs/traceset_norway.csv')
        sys.exit(2)
    lines = [l.strip() for l in traces_index.read_text(encoding='utf-8').splitlines() if l.strip()]
    header = lines[0].lower()
    rows = lines[1:]
    if 'relative_path' in header:
        traces = [r for r in rows][: cfg['num_traces']]
    else:
        traces = rows[: cfg['num_traces']]

    abrs = cfg['abrs']
    caches = cfg['cache_policies']
    results_dir = cfg['results_dir']

    exe = Path('ns3/build/run_scenario')
    if os.name == 'nt':
        exe = Path('ns3/build/run_scenario.exe')

    if not exe.exists():
        print('Building run_scenario...')
        subprocess.check_call(['cmake','-S','ns3','-B','ns3/build','-DCMAKE_BUILD_TYPE=Release'])
        subprocess.check_call(['cmake','--build','ns3/build','--config','Release','--target','run_scenario'])

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    i = 0
    for t in traces:
        for abr in abrs:
            for cache in caches:
                run_id = f"t{str(i).zfill(3)}_{Path(t).stem}_{abr}_{cache}"
                cmd = [str(exe),
                       '--abr', abr,
                       '--cache_policy', cache,
                       '--results_dir', results_dir,
                       '--run_id', run_id,
                       '--trace_csv', str(Path('datasets')/t)]
                print('>>', ' '.join(cmd))
                subprocess.call(cmd)
                i += 1

    print('Done sweep. Logs in', results_dir)

if __name__ == '__main__':
    main()
