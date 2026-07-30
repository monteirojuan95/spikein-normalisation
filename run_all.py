#!/usr/bin/env python3
"""Regenerate every table and figure in the manuscript.

    python3 run_all.py            # full run, ~4 min
    python3 run_all.py --quick    # reduced replication smoke test, ~1 min

Runs the four analysis stages in order. Each is launched as a separate process
so that it starts from its own seeded generator; this is what makes the
archived outputs reproducible stage by stage, and it means any single stage can
equally be run on its own.

    run_analysis.py       Table 2, Figures 1-3
    run_robustness.py     Figure 4
    run_validation.py     Table 3
    run_calibration.py    Table 4

`--quick` is passed to the two stages that accept it. run_validation.py has no
reduced mode; lower N_STUDIES or N_BOOT at the top of that file instead.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (script, accepts --quick)
STAGES = [
    ('run_analysis.py', True),
    ('run_robustness.py', True),
    ('run_validation.py', False),
    ('run_calibration.py', False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='reduced replication, for a smoke test')
    a = ap.parse_args()

    t0 = time.time()
    for script, takes_quick in STAGES:
        cmd = [sys.executable, os.path.join(HERE, script)]
        if a.quick and takes_quick:
            cmd.append('--quick')
        print(f'\n=== {" ".join(os.path.basename(c) for c in cmd[1:])}',
              flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode:
            sys.exit(f'{script} failed with exit code {r.returncode}')

    print(f'\nall stages complete in {time.time() - t0:.0f}s '
          f'-> {os.path.join(HERE, "results")}/')


if __name__ == '__main__':
    main()
