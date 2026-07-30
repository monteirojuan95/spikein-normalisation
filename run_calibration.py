#!/usr/bin/env python3
"""Table 4: published control-channel dispersions converted to the log10 scale.

    python run_calibration.py

Writes results/table4.csv. This script produces no figure: the empirical
comparison is tabular in the manuscript, and the parameter-plane figure that
earlier versions of this repository generated has been withdrawn.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import calibrate as cal

OUT = os.path.join(os.path.dirname(__file__), 'results')
COLS = ['source', 'assay_and_matrix', 'as_reported', 'total_sd_log10']


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = cal.table4_rows()

    with open(f'{OUT}/table4.csv', 'w') as f:
        f.write(','.join(COLS) + ',min_log10,max_log10\n')
        for r in rows:
            f.write(','.join(f'"{r[c]}"' for c in COLS) +
                    f',{r["min_log10"]:.4f},{r["max_log10"]:.4f}\n')

    w = [max(len(r[c]) for r in rows + [dict(zip(COLS, COLS))]) for c in COLS]
    print('  '.join(c.ljust(x) for c, x in zip(COLS, w)))
    last = None
    for r in rows:
        src = '' if r['source'] == last else r['source']
        last = r['source']
        print('  '.join(v.ljust(x) for v, x in
                        zip([src, r['assay_and_matrix'], r['as_reported'],
                             r['total_sd_log10']], w)))

    lo, hi = cal.span()
    print(f'\nreported control-channel dispersion spans {lo:.2f}-{hi:.2f} log10 units')
    print(f'done -> {OUT}/table4.csv')


if __name__ == '__main__':
    main()
