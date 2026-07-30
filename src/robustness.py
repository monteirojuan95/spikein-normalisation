"""Robustness experiments: what happens when the model's assumptions fail.

The closed-form criterion assumes equal amplification efficiency in both
channels, Gaussian homoscedastic errors, a single shared perturbation, an
exactly known spike-in amount and no detection limit. Each function below
breaks one of those and re-runs the estimator comparison.
"""
import numpy as np
import spikein as sp

BETA = sp.BETA


def simulate_general(n, sd_shared, sd_ctrl, sd_target=0.15, sd_meas=sp.SD_MEAS,
                     beta_T=BETA, beta_C=BETA, lo=2.0, hi=6.0, rng=None,
                     df=None, rho_batch=1.0, sd_batch=0.0, batch_size=8,
                     ct_max=None, spike_sd=0.0):
    """Generative model with optional assumption violations.

    beta_T, beta_C   different amplification efficiencies per channel
    df               Student-t errors with df degrees of freedom (heavy tails)
    sd_batch, rho_batch   batch-level perturbation shared only in part (rho<1)
    ct_max           censor threshold cycles above this value (non-detects)
    spike_sd         the spike-in amount itself varies between samples
    """
    rng = rng or np.random.default_rng()

    def noise(scale, size):
        if df is None:
            return rng.normal(0, scale, size)
        t = rng.standard_t(df, size)
        return t * scale / np.sqrt(df / (df - 2))     # rescale to sd = scale

    N = rng.uniform(lo, hi, n)
    s = noise(sd_shared, n)
    t_ = noise(sd_target, n)
    c = noise(sd_ctrl, n)
    eT = noise(sd_meas, n)
    eC = noise(sd_meas, n)

    # batch perturbation shared only partially between the two channels
    if sd_batch > 0:
        nb = int(np.ceil(n / batch_size))
        bT = np.repeat(rng.normal(0, sd_batch, nb), batch_size)[:n]
        indep = np.repeat(rng.normal(0, sd_batch, nb), batch_size)[:n]
        bC = rho_batch * bT + np.sqrt(max(0.0, 1 - rho_batch ** 2)) * indep
    else:
        bT = bC = np.zeros(n)

    K = sp.K_SPIKE + (rng.normal(0, spike_sd, n) if spike_sd > 0 else 0.0)

    Ct_T = sp.A_T - beta_T * (N + s + t_ + bT) + eT
    Ct_C = sp.A_C - beta_C * (K + s + c + bC) + eC

    keep = np.ones(n, dtype=bool)
    if ct_max is not None:
        keep = (Ct_T <= ct_max) & (Ct_C <= ct_max)
    return N[keep], Ct_T[keep], Ct_C[keep]


def compare(regime, n_studies=120, n=40, repeats=3, rng=None, **kw):
    """CV RMSE per estimator under the given perturbation.

    Returns (runs, names, n_kept) where `runs` is an (n_kept, n_estimators)
    array holding one value per independent synthetic study. The per-study
    values are returned rather than their means so that the caller can attach
    Monte Carlo standard errors, and so that contrasts between estimators are
    formed on the paired studies that produced them.
    """
    ss, sc = regime
    rng = rng or np.random.default_rng(0)
    est = [e for e in sp.estimator_set(ss, sc, rng=rng) if e.name != 'M5 ridge']
    runs, names = [], [e.name for e in est]
    for _ in range(n_studies):
        N, tT, tC = simulate_general(n, ss, sc, rng=rng, **kw)
        if len(N) < 15:
            continue
        res, names = sp.cv_rmse(N, tT, tC, est, V=5, repeats=repeats, rng=rng)
        runs.append([res[k] for k in names])
    return np.asarray(runs, float), names, len(runs)
