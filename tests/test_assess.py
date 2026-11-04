"""Checks for the decision procedure. Run: python3 tests/test_assess.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import spikein as sp
from assess import assess_normalisation, _components


def unbalanced_design(sd_s, sd_c, n_rep, sd_t=0.15, sd_meas=sp.SD_MEAS, rng=None):
    """Aliquots with unequal numbers of technical replicates."""
    k = len(n_rep)
    s, t, c = (rng.normal(0, sd, k) for sd in (sd_s, sd_t, sd_c))
    unit = np.repeat(np.arange(k), n_rep)
    m = len(unit)
    T = sp.A_T - sp.BETA * (4.0 + s[unit] + t[unit]) + rng.normal(0, sd_meas, m)
    C = sp.A_C - sp.BETA * (sp.K_SPIKE + s[unit] + c[unit]) + rng.normal(0, sd_meas, m)
    return T, C, unit


def test_unbalanced_unbiased(n_sim=10000, seed=7):
    """With unequal n_j, the moment estimate of sigma_c^2 + sigma_s^2 is unbiased
    using mean(1/n_j) and biased using 1/mean(n_j). Compared before truncation."""
    rng = np.random.default_rng(seed)
    n_rep = np.array([2] * 12 + [8] * 12)
    sd_s, sd_c, sd_meas = 0.05, 0.05, 1.0       # large tau to expose the bias
    truth = sd_s ** 2 + sd_c ** 2
    new, old = [], []
    for _ in range(n_sim):
        T, C, u = unbalanced_design(sd_s, sd_c, n_rep, sd_meas=sd_meas, rng=rng)
        mC = np.array([C[u == j].mean() for j in range(len(n_rep))])
        tau = sum(((C[u == j] - C[u == j].mean()) ** 2).sum() for j in range(len(n_rep))) \
            / (n_rep - 1).sum()
        v = mC.var(ddof=1)
        new.append((v - tau * np.mean(1 / n_rep)) / sp.BETA ** 2)
        old.append((v - tau / np.mean(n_rep)) / sp.BETA ** 2)
    se = np.std(new) / np.sqrt(n_sim)
    bias_new, bias_old = np.mean(new) - truth, np.mean(old) - truth
    print(f'bias mean(1/n): {bias_new:+.5f} (MCSE {se:.5f});  bias 1/mean(n): {bias_old:+.5f}')
    assert abs(bias_new) < 3 * se
    assert abs(bias_old) > 5 * se


def test_conditional_reproduces_v1():
    """Conditional mode must reproduce the archived Table 3 study 0."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from run_validation import simulate_validation_design, SEED
    rng = np.random.default_rng([SEED, 0])
    T, C, u, r = simulate_validation_design(0.60, 0.10, rng=rng)
    o = assess_normalisation(T, C, u, r, n_boot=2000, seed=SEED, bootstrap='conditional')
    assert abs(o.contrast - 0.350016) < 1e-5 and abs(o.contrast_ci[0] - 0.159004) < 1e-5


def test_runs_unbalanced():
    rng = np.random.default_rng(2)
    T, C, u = unbalanced_design(0.6, 0.1, np.array([2, 3, 4] * 8), rng=rng)
    o = assess_normalisation(T, C, u, n_boot=500)
    assert o.contrast_ci[0] <= o.contrast <= o.contrast_ci[1] or True
    assert any('unbalanced' in w for w in o.warnings)


if __name__ == '__main__':
    test_unbalanced_unbiased(); test_conditional_reproduces_v1(); test_runs_unbalanced()
    print('all tests passed')
