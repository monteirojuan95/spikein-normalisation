#!/usr/bin/env python3
"""End-to-end evaluation of the decision procedure.

    python run_decision.py

Each simulated laboratory
  1. runs a replicate validation panel (24 processed aliquots x 3 technical
     replicates) and applies the test of D = 0 from assess.py;
  2. picks a quantification strategy from the test result;
  3. fits that strategy on n calibrators; and
  4. is scored by RMSE on 1000 independent processed test samples.

Two calibration situations are compared on the same validation panels:

  processed  calibrators carried through the full workflow, so M1, M2 and M3
             can all be fitted. Rule: supports -> M2, against -> M1,
             indeterminate -> M3.
  clean      calibrators are a clean dilution series amplified directly (no
             processing perturbation), so the coupling cannot be learned and
             the practical choice is M1 or M2. Rule: supports -> M2,
             otherwise M1 (prespecified); the variant ruleM2 sends an
             indeterminate result to M2 instead.

Each policy is compared with an oracle that, in each laboratory, uses whichever
of the available estimators has the lowest test RMSE.

Writes results/decision.csv and results/decision_labs.csv.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import spikein as sp
from assess import _unit_stats, variance_test
from run_validation import simulate_validation_design, SD_S_EXACT

OUT = os.path.join(os.path.dirname(__file__), 'results')
SEED = 20261104
N_LABS = 500
N_TEST = 1000
NS = [15, 40]
ALPHA = 0.05

SETTINGS = [
    ('Favourable',       0.60,       0.10),
    ('theta* 0.905',     0.50,       0.15),
    ('theta* 0.649',     0.35,       0.25),
    ('Near boundary',    0.30,       0.30),
    ('Exact boundary',   SD_S_EXACT, 0.30),
    ('theta* 0.331',     0.25,       0.35),
    ('Unfavourable',     0.10,       0.50),
]


def test_decision(sd_s, sd_c, rng):
    """Validation panel -> 'supports' / 'against' / 'indeterminate'."""
    ct_T, ct_C, unit, _ = simulate_validation_design(sd_s, sd_c, rng=rng)
    mT, mC, n, _, ssC = _unit_stats(ct_T, ct_C, unit)
    tau_C2 = ssC.sum() / (n - 1).sum()
    t, _, p = variance_test(mT, mC, n, tau_C2, r_routine=1)
    if p < ALPHA:
        return 'supports' if t > 0 else 'against'
    return 'indeterminate'


def clean_standards(n, sd_meas=sp.SD_MEAS, rng=None):
    """Dilution series with the spike, amplified directly: no processing loss."""
    N = rng.uniform(2.0, 6.0, n)
    ct_T = sp.A_T - sp.BETA * N + rng.normal(0, sd_meas, n)
    ct_C = sp.A_C - sp.BETA * sp.K_SPIKE + rng.normal(0, sd_meas, n)
    return N, ct_T, ct_C


def rmse(est, N, ct_T, ct_C):
    return float(np.sqrt(np.mean((est.predict(ct_T, ct_C) - N) ** 2)))


def one_lab(sd_s, sd_c, n, rng):
    dec = test_decision(sd_s, sd_c, rng)
    Nt, tT, tC = sp.simulate(N_TEST, sd_s, sd_c, rng=rng)          # test samples

    est = {e.name: e for e in sp.estimator_set(rng=rng)[:3]}         # M1, M2, M3
    Np, pT, pC = sp.simulate(n, sd_s, sd_c, rng=rng)                 # processed
    proc = {k: rmse(e.fit(pT, pC, Np), Nt, tT, tC) for k, e in est.items()}

    est = {e.name: e for e in sp.estimator_set(rng=rng)[:2]}         # M1, M2
    Nc, cT, cC = clean_standards(n, rng=rng)                         # clean
    clean = {k: rmse(e.fit(cT, cC, Nc), Nt, tT, tC) for k, e in est.items()}

    pick_p = {'supports': 'M2 dCt', 'against': 'M1 naive',
              'indeterminate': 'M3 partial'}[dec]
    pick_c = 'M2 dCt' if dec == 'supports' else 'M1 naive'
    pick_c2 = 'M1 naive' if dec == 'against' else 'M2 dCt'   # indeterminate -> M2
    return {
        'decision': dec,
        'proc_M1': proc['M1 naive'], 'proc_M2': proc['M2 dCt'],
        'proc_M3': proc['M3 partial'], 'proc_rule': proc[pick_p],
        'proc_oracle': min(proc.values()),
        'clean_M1': clean['M1 naive'], 'clean_M2': clean['M2 dCt'],
        'clean_rule': clean[pick_c], 'clean_ruleM2': clean[pick_c2],
        'clean_oracle': min(clean.values()),
    }


POLICIES = {'proc': ['M1', 'M2', 'M3', 'rule'], 'clean': ['M1', 'M2', 'rule', 'ruleM2']}


def main():
    os.makedirs(OUT, exist_ok=True)
    summary, labs = [], []
    for si, (label, ss, sc) in enumerate(SETTINGS):
        th = sp.theta_star(ss, sc)
        for n in NS:
            rows = []
            for k in range(N_LABS):
                rng = np.random.default_rng([SEED, si, n, k])
                r = one_lab(ss, sc, n, rng)
                r.update(setting=label, n=n, lab=k)
                rows.append(r)
            labs.extend(rows)
            out = {'setting': label, 'sigma_s': ss, 'sigma_c': sc,
                   'theta_star': th, 'n': n, 'labs': N_LABS}
            for d in ('supports', 'against', 'indeterminate'):
                out[f'pct_{d}'] = 100.0 * np.mean([r['decision'] == d for r in rows])
            for sit, pols in POLICIES.items():
                orc = np.array([r[f'{sit}_oracle'] for r in rows])
                out[f'{sit}_oracle'] = orc.mean()
                for p in pols:
                    v = np.array([r[f'{sit}_{p}'] for r in rows])
                    exc = 100.0 * (v / orc - 1.0)                     # per lab
                    out[f'{sit}_{p}'] = v.mean()
                    out[f'{sit}_{p}_excess'] = 100.0 * (v.mean() / orc.mean() - 1.0)
                    out[f'{sit}_{p}_excess_p95'] = float(np.percentile(exc, 95))
                    out[f'{sit}_{p}_excess_max'] = float(exc.max())
            summary.append(out)
            print(f"{label:15s} n={n:3d} theta*={th:.3f}  test: sup {out['pct_supports']:5.1f} "
                  f"ag {out['pct_against']:5.1f} ind {out['pct_indeterminate']:5.1f} | "
                  f"processed excess %: " + ' '.join(
                      f"{p} {out[f'proc_{p}_excess']:6.1f}" for p in POLICIES['proc']) +
                  " | clean excess %: " + ' '.join(
                      f"{p} {out[f'clean_{p}_excess']:6.1f}" for p in POLICIES['clean']))

    cols = list(summary[0].keys())
    with open(f'{OUT}/decision.csv', 'w') as f:
        f.write(','.join(cols) + '\n')
        for s in summary:
            f.write(','.join(s[c] if isinstance(s[c], str) else f'{s[c]:.6g}'
                             for c in cols) + '\n')
    keys = ['setting', 'n', 'lab', 'decision'] + [f'{s}_{p}' for s, ps in POLICIES.items()
                                                   for p in ps + ['oracle']]
    with open(f'{OUT}/decision_labs.csv', 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in labs:
            f.write(','.join(r[k] if isinstance(r[k], str) else f'{r[k]:.6g}'
                             for k in keys) + '\n')
    print(f'done -> {OUT}/decision.csv, {OUT}/decision_labs.csv')


if __name__ == '__main__':
    main()
