"""Empirical calibration: published control-channel dispersions on the log10 scale.

A study that reports control-channel dispersion at a known input constrains
sigma_s^2 + sigma_c^2 + tau_C^2/beta^2, the denominator of theta*. It does not
fix the split between the three components, so a published value bounds the
magnitude of control-channel dispersion without locating an assay relative to
the benefit boundary.

This module reproduces Table 4 of the manuscript. Every entry below records the
value AS REPORTED in the source, together with the conversion that puts it on
the log10 scale, so that the table is derived rather than transcribed.

Three conversions cover what the literature reports:

    threshold-cycle SD      sd_cycles / beta
    repeatability limit r   (r / 2.8) -- already log10, so no beta division
    recovery mean +- SD     sqrt(ln(1 + CV^2)) / ln(10),  CV = sd / mean

The small-CV approximation is not used: reported coefficients of variation
frequently exceed one.
"""
import numpy as np

BETA = 3.32
TAU_C = 0.20
M_C = TAU_C ** 2 / BETA ** 2

R_TO_SD = 2.8      # repeatability limit r = 2.8 * s_r (ISO 5725)


def from_ct_sd(sd_cycles, beta=BETA):
    """Control Ct standard deviation (cycles) at fixed input -> log10 units."""
    return np.asarray(sd_cycles, float) / beta


def from_repeatability_limit(r):
    """Repeatability limit r (log10 units) -> within-condition SD (log10)."""
    return np.asarray(r, float) / R_TO_SD


def from_log10_sd(sd_log10):
    """Already on the log10 scale; passes through unchanged."""
    return np.asarray(sd_log10, float)


def from_recovery(mean_pct, sd_pct):
    """Recovery mean and SD (per cent) -> SD on the log10 scale, lognormal."""
    cv = np.asarray(sd_pct, float) / np.asarray(mean_pct, float)
    return np.sqrt(np.log(1 + cv ** 2)) / np.log(10)


CONVERSIONS = {
    'ct_sd':               lambda v: from_ct_sd(v),
    'repeatability_limit': lambda v: from_repeatability_limit(v),
    'log10_sd':            lambda v: from_log10_sd(v),
    'recovery':            lambda v: from_recovery([p[0] for p in v],
                                                   [p[1] for p in v]),
}


# Table 4, in manuscript row order.
#   (source, assay and matrix, as reported, conversion, values)
# For 'recovery' the values are (mean_pct, sd_pct) pairs; otherwise scalars.
PUBLISHED = [
    ('Hennechart-Collette [22]', 'MNV-1, bottled water',
     '0.45-0.89 cycles', 'ct_sd', [0.45, 0.89]),
    ('Hennechart-Collette [22]', 'MNV-1, semi-dried tomato',
     '0.98-1.30 cycles', 'ct_sd', [0.98, 1.30]),
    ('Hennechart-Collette [22]', 'MNV-1, lettuce',
     '2.15-2.67 cycles', 'ct_sd', [2.15, 2.67]),
    ('Hennechart-Collette [22]', 'Mengovirus, bottled water',
     '0.66-0.67 cycles', 'ct_sd', [0.66, 0.67]),
    ('Hennechart-Collette [22]', 'Mengovirus, semi-dried tomato',
     '0.85-1.37 cycles', 'ct_sd', [0.85, 1.37]),
    ('Hennechart-Collette [22]', 'Mengovirus, lettuce',
     '0.41-0.55 cycles', 'ct_sd', [0.41, 0.55]),
    ('Lowther [16]', 'ISO 15216 trial, repeatability',
     'r = 0.28-0.74 log10', 'repeatability_limit', [0.28, 0.74]),
    ('Lowther [16]', 'ISO 15216 trial, reproducibility',
     'sR = 0.40-0.50 log10', 'log10_sd', [0.40, 0.50]),
    ('Raymond [17]', 'Dates, two protocol variants',
     '39 +- 11%, 44 +- 4%', 'recovery', [(39.0, 11.0), (44.0, 4.0)]),
    ('Raymond [21]', 'Leafy greens, MNV',
     '28 +- 29%', 'recovery', [(28.0, 29.0)]),
    ('Uhrbrand [19]', 'Shellfish, vMC0',
     '1.8 +- 2.4%', 'recovery', [(1.8, 2.4)]),
]


def convert(kind, values):
    """Put a set of as-reported values on the log10 scale."""
    return np.atleast_1d(CONVERSIONS[kind](values))


def table4_rows(dp=2):
    """Table 4: one row per published entry, converted and range-formatted."""
    rows = []
    for source, assay, reported, kind, values in PUBLISHED:
        conv = np.sort(convert(kind, values))
        lo, hi = float(conv[0]), float(conv[-1])
        span = (f'{lo:.{dp}f}' if round(lo, dp) == round(hi, dp)
                else f'{lo:.{dp}f}-{hi:.{dp}f}')
        rows.append({'source': source, 'assay_and_matrix': assay,
                     'as_reported': reported, 'total_sd_log10': span,
                     'min_log10': round(lo, 4), 'max_log10': round(hi, 4)})
    return rows


def span():
    """Overall range of published control-channel dispersion, log10 units."""
    allv = np.concatenate([convert(k, v) for _, _, _, k, v in PUBLISHED])
    return float(allv.min()), float(allv.max())
