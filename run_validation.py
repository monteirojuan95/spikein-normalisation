#!/usr/bin/env python3
"""Table 3: performance of the operational decision procedure.

    python run_validation.py

Simulates replicate validation designs -- aliquots of ONE homogenised material
carried through the complete workflow, with technical replicates measured
within each aliquot -- and runs src/assess.py on each. Reports how well the
variance components, theta* and the decision contrast D are recovered, and how
often the bootstrap interval for D yields the correct recommendation.

The point of the exercise: the individual components are poorly determined at
this number of aliquots, but D is a difference of components whose estimation
errors are negatively correlated, so its SIGN -- which is what the
recommendation depends on -- survives better than its parts.

Writes results/table3.csv and results/validation_studies.csv.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import spikein as sp
from assess import assess_normalisation

OUT = os.path.join(os.path.dirname(__file__), 'results')

# ---------------------------------------------------------------- design ----
N_MATERIAL = 4.0      # true log10 concentration of the homogenised material
N_UNITS = 24          # independently processed aliquots per validation study
N_REP = 3             # technical measurements of each channel within aliquot
SD_TARGET = 0.15      # target-specific perturbation, baseline value

N_STUDIES = 500       # independent validation studies per setting
N_BOOT = 2000         # bootstrap resamples of aliquots within a study
SEED = 20260729

# (label, sigma_s, sigma_c) -- favourable is regime A, unfavourable regime D,
# boundary chosen so that theta* sits just below one half.
SETTINGS = [
    ('Favourable',   0.60, 0.10),
    ('Boundary',     0.30, 0.30),
    ('Unfavourable', 0.10, 0.50),
]


def simulate_validation_design(sd_shared, sd_ctrl, sd_target=SD_TARGET,
                               sd_meas=sp.SD_MEAS, n_units=N_UNITS,
                               n_rep=N_REP, n_material=N_MATERIAL, rng=None):
    """One simulated validation study.

    The material is homogenised, so the true concentration is a single fixed
    value and every aliquot receives the same known spike-in amount. The
    processing perturbations s, t, c are drawn ONCE PER ALIQUOT; measurement
    error is drawn per replicate. This is the structure assess.py assumes and
    is deliberately not the structure of spikein.simulate(), which varies the
    concentration across samples and measures each once.
    """
    rng = rng or np.random.default_rng()

    s = rng.normal(0, sd_shared, n_units)      # shared processing
    t = rng.normal(0, sd_target, n_units)      # target-specific
    c = rng.normal(0, sd_ctrl, n_units)        # control-specific

    unit = np.repeat(np.arange(n_units), n_rep)
    eT = rng.normal(0, sd_meas, n_units * n_rep)
    eC = rng.normal(0, sd_meas, n_units * n_rep)

    ct_T = sp.A_T - sp.BETA * (n_material + s[unit] + t[unit]) + eT
    ct_C = sp.A_C - sp.BETA * (sp.K_SPIKE + s[unit] + c[unit]) + eC
    rep = np.tile(np.arange(n_rep), n_units)
    return ct_T, ct_C, unit, rep


def truth(sd_shared, sd_ctrl, sd_meas=sp.SD_MEAS, beta=sp.BETA):
    """Population theta*, D and components at a setting."""
    m_c = sd_meas ** 2 / beta ** 2
    return {'sigma_s2': sd_shared ** 2,
            'sigma_c2': sd_ctrl ** 2,
            'm_c': m_c,
            'theta_star': sp.theta_star(sd_shared, sd_ctrl, sd_meas, beta),
            'contrast': sd_shared ** 2 - sd_ctrl ** 2 - m_c}


def classify(ci):
    """Decision implied by a bootstrap interval for D, matching assess.py."""
    lo, hi = ci
    if lo > 0:
        return 'supports'
    if hi < 0:
        return 'against'
    return 'indeterminate'


def run_setting(label, sd_shared, sd_ctrl, n_studies=N_STUDIES, base_seed=SEED):
    tru = truth(sd_shared, sd_ctrl)
    rows, tally = [], {'supports': 0, 'against': 0, 'indeterminate': 0}

    for k in range(n_studies):
        rng = np.random.default_rng([base_seed, k])
        ct_T, ct_C, unit, rep = simulate_validation_design(
            sd_shared, sd_ctrl, rng=rng)
        out = assess_normalisation(ct_T, ct_C, unit, rep,
                                   n_boot=N_BOOT, seed=base_seed + k)
        d = classify(out.contrast_ci)
        tally[d] += 1
        rows.append({'setting': label, 'study': k,
                     'sigma_s2': out.sigma_s2, 'sigma_c2': out.sigma_c2,
                     'm_c': out.tau_C2 / out.beta ** 2,
                     'theta_star': out.theta_star, 'contrast': out.contrast,
                     'ci_lo': out.contrast_ci[0], 'ci_hi': out.contrast_ci[1],
                     'decision': d})

    a = {k: np.array([r[k] for r in rows], float)
         for k in ('sigma_s2', 'sigma_c2', 'm_c', 'theta_star',
                   'contrast', 'ci_lo', 'ci_hi')}
    # sign of D recovered correctly, judged against the true sign
    want = 'supports' if tru['contrast'] > 0 else 'against'
    summary = {
        'setting': label, 'sigma_s': sd_shared, 'sigma_c': sd_ctrl,
        'true_sigma_s2': tru['sigma_s2'], 'true_sigma_c2': tru['sigma_c2'],
        'true_m_c': tru['m_c'],
        'true_theta_star': tru['theta_star'], 'true_contrast': tru['contrast'],
        'n_studies': n_studies,
        'pct_supports': 100.0 * tally['supports'] / n_studies,
        'pct_against': 100.0 * tally['against'] / n_studies,
        'pct_indeterminate': 100.0 * tally['indeterminate'] / n_studies,
        'pct_correct_sign': 100.0 * tally[want] / n_studies,
        'pct_wrong_sign': 100.0 * tally['supports' if want == 'against'
                                        else 'against'] / n_studies,
        'frac_sigma_s2_at_floor': float(np.mean(a['sigma_s2'] <= 0)),
        'frac_sigma_c2_at_floor': float(np.mean(a['sigma_c2'] <= 0)),
    }
    for k, v in a.items():
        summary[f'mean_{k}'] = float(v.mean())
        summary[f'sd_{k}'] = float(v.std(ddof=1))
    return summary, rows


COLS = ['setting', 'sigma_s', 'sigma_c', 'true_contrast', 'true_theta_star',
        'mean_sigma_s2', 'sd_sigma_s2', 'mean_sigma_c2', 'sd_sigma_c2',
        'mean_m_c', 'mean_theta_star', 'sd_theta_star',
        'mean_contrast', 'sd_contrast', 'mean_ci_lo', 'mean_ci_hi',
        'pct_supports', 'pct_against', 'pct_indeterminate',
        'pct_correct_sign', 'pct_wrong_sign',
        'frac_sigma_s2_at_floor', 'frac_sigma_c2_at_floor', 'n_studies']


def main():
    os.makedirs(OUT, exist_ok=True)
    summaries, all_rows = [], []
    for label, ss, sc in SETTINGS:
        print(f'{label:14s} sigma_s={ss:.2f} sigma_c={sc:.2f} ... ',
              end='', flush=True)
        s, rows = run_setting(label, ss, sc)
        summaries.append(s)
        all_rows.extend(rows)
        print(f"D_true={s['true_contrast']:+.4f}  "
              f"theta*_true={s['true_theta_star']:.3f}  "
              f"correct={s['pct_correct_sign']:.1f}%  "
              f"wrong={s['pct_wrong_sign']:.1f}%  "
              f"indet={s['pct_indeterminate']:.1f}%")

    with open(f'{OUT}/table3.csv', 'w') as f:
        f.write(','.join(COLS) + '\n')
        for s in summaries:
            f.write(','.join(
                s[c] if isinstance(s[c], str) else f'{s[c]:.6g}'
                for c in COLS) + '\n')

    keys = ['setting', 'study', 'sigma_s2', 'sigma_c2', 'm_c', 'theta_star',
            'contrast', 'ci_lo', 'ci_hi', 'decision']
    with open(f'{OUT}/validation_studies.csv', 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in all_rows:
            f.write(','.join(
                r[k] if isinstance(r[k], str) else f'{r[k]:.6g}'
                for k in keys) + '\n')

    print('\n--- Table 3 rows (paste these into the manuscript) ---')
    for s in summaries:
        print(f"\n{s['setting']}  (sigma_s={s['sigma_s']:.2f}, "
              f"sigma_c={s['sigma_c']:.2f})")
        print(f"  true D            {s['true_contrast']:+.4f}")
        print(f"  true theta*       {s['true_theta_star']:.3f}")
        print(f"  sigma_s2 hat      {s['mean_sigma_s2']:.4f} "
              f"(SD {s['sd_sigma_s2']:.4f})   true {s['true_sigma_s2']:.4f}")
        print(f"  sigma_c2 hat      {s['mean_sigma_c2']:.4f} "
              f"(SD {s['sd_sigma_c2']:.4f})   true {s['true_sigma_c2']:.4f}")
        print(f"  tauC2/beta2 hat   {s['mean_m_c']:.4f}"
              f"                 true {s['true_m_c']:.4f}")
        print(f"  theta* hat        {s['mean_theta_star']:.3f} "
              f"(SD {s['sd_theta_star']:.3f})")
        print(f"  D hat             {s['mean_contrast']:+.4f} "
              f"(SD {s['sd_contrast']:.4f}), mean CI "
              f"[{s['mean_ci_lo']:+.4f}, {s['mean_ci_hi']:+.4f}]")
        print(f"  decisions         supports {s['pct_supports']:.1f}%  "
              f"against {s['pct_against']:.1f}%  "
              f"indeterminate {s['pct_indeterminate']:.1f}%")
        print(f"  sign correct      {s['pct_correct_sign']:.1f}%   "
              f"wrong {s['pct_wrong_sign']:.1f}%")
        print(f"  at zero floor     sigma_s2 "
              f"{100 * s['frac_sigma_s2_at_floor']:.1f}%  sigma_c2 "
              f"{100 * s['frac_sigma_c2_at_floor']:.1f}%")

    print(f'\ndone -> {OUT}/table3.csv, {OUT}/validation_studies.csv')
    print(f'seed {SEED}, {N_UNITS} aliquots x {N_REP} replicates, '
          f'{N_STUDIES} studies per setting, {N_BOOT} bootstrap resamples')


if __name__ == '__main__':
    main()
