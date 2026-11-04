"""Decide whether internal-control normalisation will help a given assay.

The manuscript gives a criterion; this turns it into a procedure. Supply
threshold cycles from a replicate validation design and it returns the
variance components, the optimal control coefficient, the decision contrast
with a bootstrap interval, and a recommendation.

    from assess import assess_normalisation
    out = assess_normalisation(target_ct, control_ct, processing_unit)
    print(out)

Required design: replicate aliquots of one homogenised material carried
through the complete workflow (the processing units), with the spike added at
the intended stage, and target and control measured in replicate within each
aliquot (the technical replicates).
"""
from dataclasses import dataclass, field
import numpy as np
from math import sqrt

BETA_DEFAULT = 3.32


def _group_var(values, groups):
    """Pooled within-group variance."""
    ss, df = 0.0, 0
    for g in np.unique(groups):
        v = values[groups == g]
        if len(v) > 1:
            ss += np.sum((v - v.mean()) ** 2)
            df += len(v) - 1
    return ss / df if df else np.nan


@dataclass
class Assessment:
    tau_T2: float
    tau_C2: float
    sigma_s2: float
    sigma_c2: float
    theta_star: float
    contrast: float                       # D = sigma_s^2 - sigma_c^2 - tau_C^2/beta^2
    contrast_ci: tuple
    beta: float
    n_units: int
    recommendation: str
    rationale: str
    warnings: list = field(default_factory=list)
    bootstrap: str = 'hierarchical'
    r_routine: int = 1
    test_t: float = float('nan')
    test_df: int = 0
    test_p: float = float('nan')

    def __str__(self):
        lo, hi = self.contrast_ci
        w = ''.join(f'\n  ! {x}' for x in self.warnings)
        return (
            f"Processing units            {self.n_units}\n"
            f"Amplification slope beta    {self.beta:.3f} cycles per log10\n"
            f"Target measurement var      {self.tau_T2:.4f} cycles^2\n"
            f"Control measurement var     {self.tau_C2:.4f} cycles^2\n"
            f"Shared processing var       {self.sigma_s2:.4f} log10^2\n"
            f"Control-specific var        {self.sigma_c2:.4f} log10^2\n"
            f"Optimal coefficient theta*  {self.theta_star:.3f}\n"
            f"Decision contrast D         {self.contrast:+.4f}  "
            f"(95% CI {lo:+.4f} to {hi:+.4f}, {self.bootstrap} bootstrap)\n"
            f"Test of D = 0               t = {self.test_t:+.2f} on {self.test_df} df, "
            f"p = {self.test_p:.3g}\n"
            f"\nRecommendation: {self.recommendation}\n{self.rationale}{w}")


def _unit_stats(tT, tC, unit):
    """Per-aliquot means, replicate counts and within-aliquot sums of squares."""
    units = np.unique(unit)
    mT = np.array([tT[unit == u].mean() for u in units])
    mC = np.array([tC[unit == u].mean() for u in units])
    n = np.array([np.sum(unit == u) for u in units], float)
    ssT = np.array([np.sum((tT[unit == u] - m) ** 2) for u, m in zip(units, mT)])
    ssC = np.array([np.sum((tC[unit == u] - m) ** 2) for u, m in zip(units, mC)])
    return mT, mC, n, ssT, ssC


def _components(mT, mC, inv_n_bar, tau_C2, beta):
    """sigma_s^2 and sigma_c^2 from aliquot means, vectorised over rows.

    mT, mC    : (..., k) aliquot means
    inv_n_bar : (...,) mean over aliquots of 1/n_j. Var(aliquot mean of CtC)
                = beta^2 (sigma_s^2 + sigma_c^2) + tau_C^2 * mean(1/n_j), which
                is exact for unbalanced designs; 1/mean(n_j) is not.
    tau_C2    : (...,) or scalar control measurement variance, cycles^2
    """
    k = mT.shape[-1]
    dT = mT - mT.mean(-1, keepdims=True)
    dC = mC - mC.mean(-1, keepdims=True)
    cov = (dT * dC).sum(-1) / (k - 1)
    var_c = (dC ** 2).sum(-1) / (k - 1) - tau_C2 * inv_n_bar
    s_s2 = np.maximum(cov, 0.0) / beta ** 2
    s_c2 = np.maximum(var_c / beta ** 2 - s_s2, 0.0)
    return s_s2, s_c2


def _t_sf(t, df):
    """Upper tail of Student's t, via the regularised incomplete beta (no scipy)."""
    from math import lgamma, exp, log
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    # continued fraction for I_x(a, b) (Numerical Recipes betacf)
    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300); h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
            c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
            de = d * c; h *= de
            if abs(de - 1.0) < 1e-12:
                break
        return h
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
    ib = bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * betacf(b, a, 1 - x) / b
    p_two = ib                      # P(|T| > |t|)
    return p_two / 2 if t > 0 else 1 - p_two / 2


def variance_test(mT, mC, n_rep, tau_C2, r_routine=1):
    """Test of H0: D = 0 on processed-aliquot means (Pitman-Morgan type).

    With W = 2 mT - mC and Y = mC (aliquot means of target and control),
        beta^2 D = Cov(W, Y) - tau_C^2 (1/r - mean(1/n_j)),
    and Cov(W, Y) = Var(mT) - Var(mT - mC): apart from the measurement offset,
    DeltaCt helps if and only if it reduces the between-aliquot variance of the
    target cycle. The test regresses W on Y and compares the slope with
    offset / Var(Y) using Student's t on J - 2 degrees of freedom.
    Returns (t, df, two-sided p).
    """
    W, Y = 2 * mT - mC, mC
    k = len(Y)
    off = tau_C2 * (1.0 / r_routine - np.mean(1.0 / n_rep))
    Yc, Wc = Y - Y.mean(), W - W.mean()
    syy = float((Yc ** 2).sum())
    b = float((Yc * Wc).sum()) / syy
    b0 = off / (syy / (k - 1))
    res = Wc - b * Yc
    se = sqrt(float((res ** 2).sum()) / (k - 2) / syy)
    t = (b - b0) / se
    p = 2 * min(_t_sf(t, k - 2), 1 - _t_sf(t, k - 2))
    return t, k - 2, p


def assess_normalisation(target_ct, control_ct, processing_unit,
                         technical_replicate=None, beta=BETA_DEFAULT,
                         n_boot=2000, seed=0, r_routine=1,
                         bootstrap='hierarchical', alpha=0.05):
    """Estimate the variance components and recommend a normalisation strategy.

    target_ct, control_ct : threshold cycles, one row per measurement
    processing_unit       : identifier of the independently processed aliquot
    technical_replicate   : optional; unused directly, kept for clarity of design
    beta                  : cycles per log10; estimate from your standard curve
    r_routine             : number of control measurements that will be AVERAGED
                            in routine quantification. Distinct from the number of
                            replicates per aliquot in this validation design.
    bootstrap             : 'hierarchical' (default) resamples aliquots and
                            re-estimates tau_C^2 from the within-aliquot replicates
                            of each resample, so its uncertainty is propagated;
                            'conditional' holds tau_C^2 at its full-sample value
                            (the procedure of manuscript version 1).
    Replicate numbers may differ between aliquots.
    """
    if bootstrap not in ('hierarchical', 'conditional'):
        raise ValueError("bootstrap must be 'hierarchical' or 'conditional'")
    tT = np.asarray(target_ct, float)
    tC = np.asarray(control_ct, float)
    unit = np.asarray(processing_unit)
    warn = []

    mT, mC, nrep, ssT, ssC = _unit_stats(tT, tC, unit)
    df = nrep - 1
    if df.sum() < 1:
        raise ValueError('need at least two measurements in some processing unit')
    # pooled within-aliquot measurement variance (valid for unequal n_j)
    tau_T2 = ssT.sum() / df.sum()
    tau_C2 = ssC.sum() / df.sum()
    inv_n = 1.0 / nrep
    k = len(mT)
    if k < 6:
        warn.append(f'only {k} processing units; the interval will be very wide')
    if np.ptp(nrep) > 0:
        warn.append('unbalanced replicate numbers; handled via mean(1/n_j)')

    sigma_s2, sigma_c2 = _components(mT, mC, inv_n.mean(), tau_C2, beta)
    sigma_s2, sigma_c2 = float(sigma_s2), float(sigma_c2)
    m_c = tau_C2 / (r_routine * beta ** 2)
    denom = sigma_s2 + sigma_c2 + m_c
    theta = sigma_s2 / denom if denom > 0 else 0.0
    D = sigma_s2 - sigma_c2 - m_c

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, (n_boot, k))
    if bootstrap == 'hierarchical':
        dfb = df[idx].sum(1)
        tau_b = np.where(dfb > 0, ssC[idx].sum(1) / np.maximum(dfb, 1), tau_C2)
    else:
        tau_b = np.full(n_boot, tau_C2)
    s2, c2 = _components(mT[idx], mC[idx], inv_n[idx].mean(1), tau_b, beta)
    boot = s2 - c2 - tau_b / (r_routine * beta ** 2)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # decision: calibrated t-test of D = 0; the bootstrap interval describes D
    t_stat, t_df, p_val = variance_test(mT, mC, nrep, tau_C2, r_routine)

    if p_val < alpha and t_stat > 0:
        rec = 'Constrained DeltaCt normalisation is supported.'
        why = ('The shared processing variability exceeds the control-specific '
               'variability inflated by control measurement noise, and the test '
               'rejects D = 0. Subtracting the control cycle should reduce error. '
               'Estimating the coefficient remains at least as good and is the safer '
               'choice if theta* is far from 1.')
    elif p_val < alpha and t_stat < 0:
        rec = 'Do not apply DeltaCt normalisation.'
        why = ('The control carries less shared signal than it contributes noise. '
               'Subtracting it will increase error. Either use the target channel '
               'alone or enter the control as a predictor with an estimated '
               'coefficient, which degrades gracefully to ignoring it.')
    else:
        rec = 'Estimate the coefficient rather than fixing it.'
        why = ('The test does not reject D = 0, so this assay sits near the decision '
               'boundary and the data do not settle the question. Regress the target '
               'on both channels rather than subtracting, and consider reporting '
               'unnormalised estimates alongside.')

    if theta > 0.9:
        warn.append('theta* is close to 1: DeltaCt is nearly optimal here')
    if theta < 0.1:
        warn.append('theta* is close to 0: the control carries almost no shared signal')
    if np.mean(nrep) < 2:
        warn.append('fewer than two replicates per unit on average; tau^2 is unreliable')

    return Assessment(tau_T2, tau_C2, sigma_s2, sigma_c2, theta, D, (lo, hi),
                      beta, k, rec, why, warn, bootstrap, r_routine,
                      t_stat, t_df, p_val)
