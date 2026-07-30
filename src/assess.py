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
            f"(95% CI {lo:+.4f} to {hi:+.4f})\n"
            f"\nRecommendation: {self.recommendation}\n{self.rationale}{w}")


def assess_normalisation(target_ct, control_ct, processing_unit,
                         technical_replicate=None, beta=BETA_DEFAULT,
                         n_boot=2000, seed=0):
    """Estimate the variance components and recommend a normalisation strategy.

    target_ct, control_ct : threshold cycles, one row per measurement
    processing_unit       : identifier of the independently processed aliquot
    technical_replicate   : optional; unused directly, kept for clarity of design
    beta                  : cycles per log10; estimate from your standard curve
    """
    tT = np.asarray(target_ct, float)
    tC = np.asarray(control_ct, float)
    unit = np.asarray(processing_unit)
    warn = []

    # measurement variance from replicates within a processing unit
    tau_T2 = _group_var(tT, unit)
    tau_C2 = _group_var(tC, unit)
    if np.isnan(tau_T2) or np.isnan(tau_C2):
        raise ValueError('need at least two measurements in some processing unit')

    # unit means carry processing variation plus measurement noise / n_rep
    units = np.unique(unit)
    mT = np.array([tT[unit == u].mean() for u in units])
    mC = np.array([tC[unit == u].mean() for u in units])
    nrep = np.array([np.sum(unit == u) for u in units], float)
    k = len(units)
    if k < 6:
        warn.append(f'only {k} processing units; the interval will be very wide')

    def components(mt, mc, nr):
        cov = np.cov(mt, mc, ddof=1)[0, 1]
        var_c = np.var(mc, ddof=1) - tau_C2 / np.mean(nr)
        s_s2 = max(cov, 0.0) / beta ** 2
        s_c2 = max(var_c / beta ** 2 - s_s2, 0.0)
        return s_s2, s_c2

    sigma_s2, sigma_c2 = components(mT, mC, nrep)
    m_c = tau_C2 / beta ** 2
    denom = sigma_s2 + sigma_c2 + m_c
    theta = sigma_s2 / denom if denom > 0 else 0.0
    D = sigma_s2 - sigma_c2 - m_c

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, k)
        s2, c2 = components(mT[idx], mC[idx], nrep[idx])
        boot.append(s2 - c2 - m_c)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    if lo > 0:
        rec = 'Constrained DeltaCt normalisation is supported.'
        why = ('The shared processing variability exceeds the control-specific '
               'variability inflated by control measurement noise, and the interval '
               'excludes zero. Subtracting the control cycle should reduce error. '
               'Estimating the coefficient remains at least as good and is the safer '
               'choice if theta* is far from 1.')
    elif hi < 0:
        rec = 'Do not apply DeltaCt normalisation.'
        why = ('The control carries less shared signal than it contributes noise. '
               'Subtracting it will increase error. Either use the target channel '
               'alone or enter the control as a predictor with an estimated '
               'coefficient, which degrades gracefully to ignoring it.')
    else:
        rec = 'Estimate the coefficient rather than fixing it.'
        why = ('The interval spans zero, so this assay sits near the decision '
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
                      beta, k, rec, why, warn)
