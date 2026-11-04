#!/usr/bin/env python3
"""Calibration paradigm: inverse regression vs inverted standard curve, with EIV.

    python run_paradigm.py

On the same processed calibrators and the same independent test samples,
compares
  inverse    regress N on the threshold cycles and apply the fitted equation
             (M1, M2, M3 as in the manuscript);
  classical  regress the composite Z(theta) = CtT - theta*CtC on N and invert
             the fitted line (theta = 0 for M1, 1 for M2); for the estimated-
             coefficient form, regress (CtT, CtC) jointly on N and invert by
             generalised least squares with the estimated residual covariance
             (bivariate classical calibration, 'C-GLS');
  Deming     classical M1 and M2 fitted by Deming regression, for the case in
             which the assigned calibrator concentrations are uncertain.

Assigned calibrator concentrations are N_i + delta_i with delta_i ~ N(0, sd_delta^2),
sd_delta in {0, 0.05, 0.10} log10. Test RMSE is always against the true N.
The Deming error-variance ratio is set to its true value, which in practice
would come from a replicate validation design; Deming is not run at sd_delta = 0,
where it coincides with the classical fit.

Writes results/paradigm.csv.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import spikein as sp

OUT = os.path.join(os.path.dirname(__file__), 'results')
SEED = 20261105
N_STUDIES = 500
N_TEST = 1000
NS = [15, 40]
SD_DELTA = [0.0, 0.05, 0.10]
REGIMES = {'A': (0.60, 0.10), 'B': (0.35, 0.25), 'D': (0.10, 0.50)}
SD_T = 0.15


def error_budget(theta, ss, sc, st=SD_T, sm=sp.SD_MEAS, beta=sp.BETA):
    """E(theta) in log10^2 units, eq. (12)."""
    m = sm ** 2 / beta ** 2
    return (1 - theta) ** 2 * ss ** 2 + st ** 2 + theta ** 2 * sc ** 2 + m + theta ** 2 * m


def ols_line(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    b = (xc * yc).sum() / (xc ** 2).sum()
    return y.mean() - b * x.mean(), b


def deming_line(x, y, lam):
    """Deming fit of y on x; lam = Var(error in y) / Var(error in x)."""
    xc, yc = x - x.mean(), y - y.mean()
    sxx, syy, sxy = (xc ** 2).mean(), (yc ** 2).mean(), (xc * yc).mean()
    b = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    return y.mean() - b * x.mean(), b


def classical_invert(a, b, z):
    return (z - a) / b


def cgls(Nc, cT, cC, tT, tC):
    """Bivariate classical calibration inverted by GLS."""
    X = np.column_stack([np.ones_like(Nc), Nc])
    Y = np.column_stack([cT, cC])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)           # 2 x 2: intercepts, slopes
    R = Y - X @ B
    S = np.cov(R.T, ddof=2)
    a, b = B[0], B[1]
    w = np.linalg.solve(S, b)
    Yt = np.column_stack([tT, tC]) - a
    return (Yt @ w) / (b @ w)


def rmse(pred, N):
    return float(np.sqrt(np.mean((pred - N) ** 2)))


def one_study(ss, sc, n, sd_delta, rng):
    Nt, tT, tC = sp.simulate(N_TEST, ss, sc, rng=rng)
    Nc, cT, cC = sp.simulate(n, ss, sc, rng=rng)
    Na = Nc + (rng.normal(0, sd_delta, n) if sd_delta > 0 else 0.0)   # assigned
    out = {}
    for e in sp.estimator_set(rng=rng)[:3]:
        e.fit(cT, cC, Na)
        out['inv_' + e.name.split()[0]] = rmse(e.predict(tT, tC), Nt)
    for name, th in (('M1', 0.0), ('M2', 1.0)):
        zc, zt = cT - th * cC, tT - th * tC
        a, b = ols_line(Na, zc)
        out['cls_' + name] = rmse(classical_invert(a, b, zt), Nt)
        if sd_delta > 0:
            lam = sp.BETA ** 2 * error_budget(th, ss, sc) / sd_delta ** 2
            a, b = deming_line(Na, zc, lam)
            out['dem_' + name] = rmse(classical_invert(a, b, zt), Nt)
    out['cls_M3'] = rmse(cgls(Na, cT, cC, tT, tC), Nt)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for ri, (rg, (ss, sc)) in enumerate(REGIMES.items()):
        for n in NS:
            for di, sd in enumerate(SD_DELTA):
                runs = [one_study(ss, sc, n, sd, np.random.default_rng([SEED, ri, n, di, k]))
                        for k in range(N_STUDIES)]
                keys = runs[0].keys()
                row = {'regime': rg, 'sigma_s': ss, 'sigma_c': sc,
                       'theta_star': sp.theta_star(ss, sc), 'n': n, 'sd_delta': sd}
                for k in keys:
                    v = np.array([r[k] for r in runs])
                    row[k] = v.mean()
                    row[k + '_mcse'] = v.std(ddof=1) / np.sqrt(len(v))
                rows.append(row)
                print(f"{rg} n={n:3d} sd_delta={sd:.2f}  " + '  '.join(
                    f"{k} {row[k]:.3f}" for k in keys))
    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    with open(f'{OUT}/paradigm.csv', 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join('' if c not in r else
                             (r[c] if isinstance(r[c], str) else f'{r[c]:.6g}')
                             for c in cols) + '\n')
    print(f'done -> {OUT}/paradigm.csv')


if __name__ == '__main__':
    main()
