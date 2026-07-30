#!/usr/bin/env python3
"""Robustness experiments: does the recommendation survive assumption violations?

    python run_robustness.py [--quick]

Writes results/robustness.csv and results/figure4.png.

Every performance measure carries a Monte Carlo standard error, computed as the
between-study standard deviation divided by the square root of the number of
independent synthetic studies, as stated in the Methods. The MCSE of the two
percentage contrasts is obtained by the delta method on the paired study-level
values, because all estimators are evaluated on the same synthetic studies and
the same cross-validation folds.
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import robustness as rb
import spikein as sp

SEED = 20260728
OUT = os.path.join(os.path.dirname(__file__), 'results')

REGIMES = {'A': (0.60, 0.10), 'B': (0.35, 0.25), 'D': (0.10, 0.50)}


def beta_from_efficiency(E):
    """Cycles per log10 for amplification efficiency E (E = 1 is 100%)."""
    return 1.0 / np.log10(1.0 + E)


SCENARIOS = {
    'baseline':                 {},
    'control efficiency 90%':   dict(beta_C=beta_from_efficiency(0.90)),
    'control efficiency 80%':   dict(beta_C=beta_from_efficiency(0.80)),
    'heavy-tailed errors (t3)': dict(df=3),
    'batch effect (rho = 0.5)': dict(sd_batch=0.20, rho_batch=0.5),
    'batch effect (rho = 0)':   dict(sd_batch=0.20, rho_batch=0.0),
    'spike-in amount varies':   dict(spike_sd=0.15),
    'censoring at Ct = 38':     dict(ct_max=38.0),
}

FIELDS = ['scenario', 'regime', 'studies',
          'M1', 'M1_mcse', 'M2', 'M2_mcse', 'M3', 'M3_mcse', 'M4', 'M4_mcse',
          'M2_vs_M1_pct', 'M2_vs_M1_pct_mcse',
          'M3_vs_M1_pct', 'M3_vs_M1_pct_mcse']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ns, rp = (10, 1) if a.quick else (100, 3)
    rng = np.random.default_rng(SEED)

    rows = []
    for scen, kw in SCENARIOS.items():
        for rg, reg in REGIMES.items():
            runs, names, k = rb.compare(reg, n_studies=ns, repeats=rp, rng=rng, **kw)
            s = sp.summarise(runs, names)
            col = {n: runs[:, j] for j, n in enumerate(names)}
            d2, d2se = sp.pct_change_mcse(col['M2 dCt'], col['M1 naive'])
            d3, d3se = sp.pct_change_mcse(col['M3 partial'], col['M1 naive'])
            rows.append({
                'scenario': scen, 'regime': rg, 'studies': k,
                'M1': s['M1 naive']['mean'],   'M1_mcse': s['M1 naive']['mcse'],
                'M2': s['M2 dCt']['mean'],     'M2_mcse': s['M2 dCt']['mcse'],
                'M3': s['M3 partial']['mean'], 'M3_mcse': s['M3 partial']['mcse'],
                'M4': s['M4 interaction']['mean'],
                'M4_mcse': s['M4 interaction']['mcse'],
                'M2_vs_M1_pct': d2, 'M2_vs_M1_pct_mcse': d2se,
                'M3_vs_M1_pct': d3, 'M3_vs_M1_pct_mcse': d3se})
            print(f'{scen:<26}{rg}  M2 {d2:+7.1f}% (MCSE {d2se:4.2f})   '
                  f'M3 {d3:+7.1f}% (MCSE {d3se:4.2f})')

    with open(f'{OUT}/robustness.csv', 'w') as f:
        f.write(','.join(FIELDS) + '\n')
        for r in rows:
            f.write(','.join(f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c])
                             for c in FIELDS) + '\n')

    worst = max(max(abs(r['M2_vs_M1_pct_mcse']), abs(r['M3_vs_M1_pct_mcse']))
                for r in rows)
    print(f'\nlargest MCSE on any percentage contrast: {worst:.2f} points')

    # Figure 4: M2 and M3 relative to M1, per scenario, in the intermediate regime
    scen = list(SCENARIOS)
    b = [next(r for r in rows if r['scenario'] == s and r['regime'] == 'B') for s in scen]
    y = np.arange(len(scen))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.barh(y - 0.19, [r['M2_vs_M1_pct'] for r in b], 0.36,
            xerr=[r['M2_vs_M1_pct_mcse'] for r in b], capsize=2,
            error_kw=dict(lw=0.8, ecolor='0.25'),
            label='M2 $\\Delta$Ct', color='#c0392b')
    ax.barh(y + 0.19, [r['M3_vs_M1_pct'] for r in b], 0.36,
            xerr=[r['M3_vs_M1_pct_mcse'] for r in b], capsize=2,
            error_kw=dict(lw=0.8, ecolor='0.25'),
            label='M3 estimated coefficient', color='#2980b9')
    ax.axvline(0, color='k', lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(scen); ax.invert_yaxis()
    ax.set_xlabel('Change in RMSE relative to ignoring the control (%)')
    ax.set_title('Intermediate regime ($\\sigma_s$ = 0.35, $\\sigma_c$ = 0.25)')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f'{OUT}/figure4.png', dpi=200); plt.close(fig)
    print(f'done -> {OUT}/robustness.csv, {OUT}/figure4.png')


if __name__ == '__main__':
    main()
