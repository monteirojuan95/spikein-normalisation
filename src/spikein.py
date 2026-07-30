"""
Simulation framework for evaluating internal-control (spike-in) normalisation
in metagenomic / qPCR quantification.

Revision of simulate.py. The generative model is unchanged. What changed:

  1. M5 (ridge) now regularises. The original penalised BOTH Ct coefficients
     on the RAW cycle scale with a fixed lambda = 1. Since diag(X'X) is of
     order 1e4 there, the relative penalty was ~4e-5 and M5 was numerically
     identical to M3. It now standardises the predictors on training-set
     statistics, penalises the CONTROL coefficient only, and selects lambda
     by inner cross-validation -- i.e. what the manuscript describes.
  2. M4 (interaction) now centres the two channels before forming the
     product. On the raw scale ctT*ctC is ~660 and nearly collinear with the
     main effects, which is why M4 was uniformly slightly worse.
  3. Added M6 (method-of-moments theta) and M7 (oracle theta*). Both are
     available via estimator_set(include_aux=True) but are NOT instantiated
     by any driver script: no result in the manuscript uses them, and the
     Discussion states that the moment shortcut was not evaluated.

Note on theta*: the optimal coupling is Cov(CtT, CtC) / Var(CtC), i.e. the
ordinary regression slope of the target cycle on the control cycle. It does
NOT require knowing tau_C.

Generative model
----------------
For sample i:
    N_i     ~ U(lo, hi)                 true log10 target concentration
    s_i     ~ N(0, sd_shared^2)         shared processing perturbation
    t_i     ~ N(0, sd_target^2)         target-specific perturbation
    c_i     ~ N(0, sd_ctrl^2)           control-specific perturbation
    e_T,e_C ~ N(0, sd_meas^2)           qPCR measurement noise

    Ct_T = a_T - beta*(N_i + s_i + t_i) + e_T
    Ct_C = a_C - beta*(K   + s_i + c_i) + e_C     K = known spike-in level
"""
import numpy as np

BETA = 3.32          # Ct per log10 at 100% PCR efficiency
A_T = 40.0
A_C = 38.0
K_SPIKE = 4.0        # log10 copies of spike-in added
SD_MEAS = 0.20       # default measurement sd, cycles (both channels)

RIDGE_GRID = np.logspace(-4, 3, 25)   # lambda grid on standardised predictors


def simulate(n, sd_shared, sd_ctrl, sd_target=0.15, sd_meas=SD_MEAS,
             lo=2.0, hi=6.0, rng=None):
    rng = rng or np.random.default_rng()
    N = rng.uniform(lo, hi, n)
    s = rng.normal(0, sd_shared, n)
    t = rng.normal(0, sd_target, n)
    c = rng.normal(0, sd_ctrl, n)
    eT = rng.normal(0, sd_meas, n)
    eC = rng.normal(0, sd_meas, n)
    Ct_T = A_T - BETA * (N + s + t) + eT
    Ct_C = A_C - BETA * (K_SPIKE + s + c) + eC
    return N, Ct_T, Ct_C


def theta_star(sd_shared, sd_ctrl, sd_meas=SD_MEAS, beta=BETA):
    """Population-optimal control coefficient."""
    m_c = sd_meas ** 2 / beta ** 2
    return sd_shared ** 2 / (sd_shared ** 2 + sd_ctrl ** 2 + m_c)


def _ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


class Estimator:
    """Fits on a training fold, predicts on a held-out fold."""
    name = "base"

    def fit(self, ctT, ctC, N):
        raise NotImplementedError

    def predict(self, ctT, ctC):
        raise NotImplementedError


class Plain(Estimator):
    """OLS on a design built from (ctT, ctC), optionally centred first."""

    def __init__(self, name, cols, centre=False):
        self.name, self.cols, self.centre = name, cols, centre

    def _X(self, ctT, ctC):
        if self.centre:
            ctT, ctC = ctT - self.mu_T, ctC - self.mu_C
        return np.column_stack([np.ones_like(ctT)] + [f(ctT, ctC) for f in self.cols])

    def fit(self, ctT, ctC, N):
        self.mu_T, self.mu_C = ctT.mean(), ctC.mean()
        self.b = _ols(self._X(ctT, ctC), N)
        return self

    def predict(self, ctT, ctC):
        return self._X(ctT, ctC) @ self.b


class Ridge(Estimator):
    """Unconstrained form, L2 penalty on the CONTROL coefficient only.

    Predictors are standardised on training statistics so that lambda is on a
    meaningful scale; lambda is chosen by inner V-fold CV within the training
    set. lambda -> 0 recovers M3, lambda -> inf recovers M1.
    """
    name = "M5 ridge"

    def __init__(self, grid=RIDGE_GRID, inner_V=5, rng=None):
        self.grid, self.inner_V = grid, inner_V
        self.rng = rng or np.random.default_rng(0)

    def _std(self, ctT, ctC):
        return np.column_stack([np.ones_like(ctT),
                                (ctT - self.mu[0]) / self.sd[0],
                                (ctC - self.mu[1]) / self.sd[1]])

    @staticmethod
    def _solve(X, y, lam):
        P = np.zeros((X.shape[1], X.shape[1]))
        P[2, 2] = 1.0                      # penalise the control column only
        return np.linalg.solve(X.T @ X + lam * P, X.T @ y)

    def fit(self, ctT, ctC, N):
        self.mu = np.array([ctT.mean(), ctC.mean()])
        self.sd = np.array([ctT.std(ddof=1) or 1.0, ctC.std(ddof=1) or 1.0])
        X = self._std(ctT, ctC)
        n = len(N)
        V = min(self.inner_V, n)
        idx = self.rng.permutation(n)
        folds = np.array_split(idx, V)
        err = np.zeros(len(self.grid))
        for j, lam in enumerate(self.grid):
            se = 0.0
            for f in folds:
                tr = np.setdiff1d(idx, f)
                if len(tr) < 4:
                    continue
                b = self._solve(X[tr], N[tr], lam)
                se += np.sum((X[f] @ b - N[f]) ** 2)
            err[j] = se
        self.lam = self.grid[int(np.argmin(err))]
        self.b = self._solve(X, N, self.lam)
        return self

    def predict(self, ctT, ctC):
        return self._std(ctT, ctC) @ self.b


class FixedTheta(Estimator):
    """Regress N on the composite (ctT - theta*ctC) for a supplied theta.

    theta_fn(ctT, ctC) is evaluated on the TRAINING fold. Passing a constant
    gives the oracle; passing the moment estimator gives the data-driven
    shortcut. theta = 1 reproduces M2.
    """

    def __init__(self, name, theta_fn):
        self.name, self.theta_fn = name, theta_fn

    def fit(self, ctT, ctC, N):
        self.theta = float(self.theta_fn(ctT, ctC))
        z = ctT - self.theta * ctC
        self.b = _ols(np.column_stack([np.ones_like(z), z]), N)
        return self

    def predict(self, ctT, ctC):
        z = ctT - self.theta * ctC
        return np.column_stack([np.ones_like(z), z]) @ self.b


def theta_moments(ctT, ctC):
    """Method-of-moments theta* = Cov(CtT, CtC) / Var(CtC)."""
    v = np.var(ctC, ddof=1)
    if v <= 0:
        return 0.0
    return float(np.cov(ctT, ctC, ddof=1)[0, 1] / v)


def estimator_set(sd_shared=None, sd_ctrl=None, sd_meas=SD_MEAS, rng=None,
                  include_aux=False):
    """Build the estimator family. Oracle/moment forms are opt-in."""
    est = [
        Plain("M1 naive", [lambda a, b: a]),
        Plain("M2 dCt", [lambda a, b: a - b]),
        Plain("M3 partial", [lambda a, b: a, lambda a, b: b]),
        Plain("M4 interaction", [lambda a, b: a, lambda a, b: b,
                                 lambda a, b: a * b], centre=True),
        Ridge(rng=rng),
    ]
    if include_aux:
        est.append(FixedTheta("M6 moment theta", theta_moments))
        if sd_shared is not None:
            th = theta_star(sd_shared, sd_ctrl, sd_meas)
            est.append(FixedTheta("M7 oracle theta*", lambda a, b, th=th: th))
    return est


def cv_rmse(N, ctT, ctC, estimators, V=5, repeats=20, rng=None):
    """Repeated V-fold CV. Returns dict name -> mean RMSE over repeats."""
    rng = rng or np.random.default_rng()
    n = len(N)
    names = [e.name for e in estimators]
    acc = {k: [] for k in names}
    for _ in range(repeats):
        idx = rng.permutation(n)
        folds = np.array_split(idx, V)
        preds = {k: np.empty(n) for k in names}
        for f in folds:
            tr = np.setdiff1d(idx, f)
            for e in estimators:
                e.fit(ctT[tr], ctC[tr], N[tr])
                preds[e.name][f] = e.predict(ctT[f], ctC[f])
        for k in names:
            acc[k].append(np.sqrt(np.mean((preds[k] - N) ** 2)))
    return {k: float(np.mean(v)) for k, v in acc.items()}, names


def mcse(per_study):
    """Monte Carlo standard error of a mean over independent studies.

    The between-study standard deviation divided by the square root of the
    number of studies, as reported in the manuscript. `per_study` is a 1-D
    array of one value per independent synthetic study.
    """
    x = np.asarray(per_study, float)
    if x.size < 2:
        return float('nan')
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def summarise(runs, names):
    """Mean, between-study SD and MCSE per estimator.

    runs : (n_studies, n_estimators) array of per-study performance values.
    """
    r = np.asarray(runs, float)
    return {k: {'mean': float(r[:, j].mean()),
                'sd': float(r[:, j].std(ddof=1)) if r.shape[0] > 1 else float('nan'),
                'mcse': mcse(r[:, j])}
            for j, k in enumerate(names)}


def ratio_mcse(num, den):
    """MCSE of a ratio of two means computed on the SAME studies.

    The estimators share synthetic studies and cross-validation folds, so their
    per-study errors are strongly positively correlated and treating the two
    means as independent overstates the Monte Carlo error of their ratio. The
    delta method applied to R = mean(num) / mean(den) gives

        Var(R) ~ [Var(a) - 2 R Cov(a, b) + R^2 Var(b)] / (B mean(b)^2)

    with a = num, b = den and B the number of studies. Returns (R, MCSE(R)).
    """
    a = np.asarray(num, float)
    b = np.asarray(den, float)
    B = a.size
    ma, mb = a.mean(), b.mean()
    R = ma / mb
    if B < 2 or mb == 0:
        return float(R), float('nan')
    C = np.cov(a, b, ddof=1)
    var = C[0, 0] - 2 * R * C[0, 1] + R ** 2 * C[1, 1]
    return float(R), float(np.sqrt(max(var, 0.0) / B) / abs(mb))


def pct_change_mcse(num, den):
    """Percentage change of `num` relative to `den`, with its MCSE."""
    R, se = ratio_mcse(num, den)
    return 100.0 * (R - 1.0), 100.0 * se


def analytic_rmse(sd_shared, sd_ctrl, sd_target=0.15, sd_meas=SD_MEAS,
                  lo=2.0, hi=6.0, beta=BETA, n=None, V=5):
    """Closed-form population RMSE for M1, M2, M3, optionally inflated to the
    finite-sample expectation for a model trained on n(V-1)/V samples."""
    m = sd_meas ** 2 / beta ** 2
    sN2 = (hi - lo) ** 2 / 12.0
    E1 = sd_shared ** 2 + sd_target ** 2 + m
    E2 = sd_target ** 2 + sd_ctrl ** 2 + m + m
    E3 = E1 - sd_shared ** 4 / (sd_shared ** 2 + sd_ctrl ** 2 + m)
    out = {}
    for name, E, p in [("M1 naive", E1, 2), ("M2 dCt", E2, 2), ("M3 partial", E3, 3)]:
        r = np.sqrt(sN2 * E / (sN2 + E))
        if n is not None:
            r *= np.sqrt(1 + p / (n * (V - 1) / V))
        out[name] = float(r)
    return out
