# When does internal-control normalisation help?

Simulation framework and closed-form criterion for spike-in normalisation in
PCR-based quantification. Companion code for the manuscript *A variance-component
criterion for deciding when internal-control normalisation improves molecular
quantification*.

## The result in one line

ΔCt normalisation reduces mean squared error relative to ignoring the control
if and only if

    σs² > σc² + τC²/β²

where σs is shared processing variability, σc is control-specific variability,
τC is control-channel measurement noise in cycles and β is the amplification
slope. More generally the optimal control coefficient is

    θ* = σs² / (σs² + σc² + τC²/β²)

so the conventional unit coefficient is correct only for a noiseless, perfectly
coupled control, and the zero-benefit boundary is the contour where θ* = ½.

## Reproducing the analysis

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run_all.py                # everything, ~4 min
```

`run_all.py` runs the four stages in order, each in its own process so that
each starts from its own seeded generator. Any stage can equally be run alone:

```bash
python3 run_analysis.py           # Table 2, Figures 1-3, ~50 s
python3 run_robustness.py         # Figure 4, assumption violations, ~10 s
python3 run_validation.py         # Table 3, validation designs, ~2 min
python3 run_calibration.py        # Table 4, published dispersions, instant
```

Add `--quick` to `run_all.py`, `run_analysis.py` or `run_robustness.py` for a
reduced-replication smoke test. `run_calibration.py` is deterministic and needs
no flag. `run_validation.py` has no reduced mode; lower `N_STUDIES` or `N_BOOT`
at the top of the file for a quick check.

Requires Python 3.9 or later. The pinned versions in `requirements.txt` are
those the archived results were produced with, recorded in
`results/run_metadata.json`.

Two seeds are used: `SEED = 20260728` in `run_analysis.py` and
`run_robustness.py`, and `SEED = 20260729` in `run_validation.py`;
`run_calibration.py` is deterministic and uses none. Every CSV in `results/`
reproduces bit-for-bit, and has been checked both on the pinned versions and on
Python 3.12 with NumPy 2.4. Figure PNGs are not byte-reproducible across
Matplotlib versions, but the numbers they are drawn from are.

## Figures and tables

Figures are numbered as in the manuscript, in order of first citation in the
text. The numbering lives in one place — the `FIGURES` dictionary at the top of
`run_analysis.py`, which is also written into `run_metadata.json` on every run —
so that renumbering is a single edit rather than a search across scripts.

| # | file | produced by | content |
|---|---|---|---|
| Figure 1 | `figure1.png` | `run_analysis.py` | Cross-validated RMSE of the four estimators across five representative regimes |
| Figure 2 | `figure2.png` | `run_analysis.py` | Two-parameter sensitivity grid over 256 (σs, σc) combinations, with the zero-benefit contour |
| Figure 3 | `figure3.png` | `run_analysis.py` | Dependence of estimator ranking on study size, with the paired M2-M3 contrast alongside |
| Figure 4 | `figure4.png` | `run_robustness.py` | Change in RMSE under each departure from the model's assumptions |
| Table 2 | `table2.csv` | `run_analysis.py` | RMSE by estimator and regime |
| Table 3 | `table3.csv` | `run_validation.py` | Recovery of the variance components and the decision contrast in simulated validation designs |
| Table 4 | `table4.csv` | `run_calibration.py` | Published control-channel dispersions on the log₁₀ scale |

**There is no Figure 5.** Earlier versions of this repository generated a
parameter-plane figure placing published assays as arcs in (σs, σc). That figure
has been withdrawn: a published dispersion constrains the *denominator* of θ\*
without fixing the split between its three components, so plotting it against
the benefit boundary invites the inference it cannot support. The same material
is now Table 4, which reports the constrained quantity and nothing more.

The `results/` directory is committed rather than ignored, so a clone or an
archived release carries the numbers behind every figure and table without
anyone having to re-run anything. Supporting output in `results/`: the raw `grid.csv` and `grid_mcse.csv`,
`samplesize.csv`, `robustness.csv`, an `analytic_check.csv` comparing simulation
against the closed form, `table2_contrasts.csv` and `samplesize_contrasts.csv`
holding the paired between-estimator contrasts, and `run_metadata.json`
recording the seed, the figure manifest and library versions.

## Monte Carlo standard errors

Every reported performance measure carries an MCSE, computed as the
between-study standard deviation divided by the square root of the number of
independent synthetic studies. `spikein.mcse` and `spikein.summarise` implement
this; MCSE columns appear in `table2.csv`, `samplesize.csv`, `robustness.csv`
and `analytic_check.csv`, and `grid_mcse.csv` holds the per-cell MCSE of the
sensitivity grid.

Contrasts between estimators need more care than their levels. All four
estimators are fitted on the same synthetic studies and the same
cross-validation folds, so their per-study errors are strongly positively
correlated, and treating the two means as independent overstates the Monte Carlo
error of a ratio between them. `spikein.ratio_mcse` and
`spikein.pct_change_mcse` therefore apply the delta method to the paired
study-level values,

    Var(R) ≈ [Var(a) − 2R·Cov(a,b) + R²·Var(b)] / (B·mean(b)²)

for R = mean(a)/mean(b) over B studies. The point estimates are unchanged: the
ratio is still formed from the means, so the published percentages are
reproduced exactly and the MCSE is attached to them.

`table2_contrasts.csv` reports M2 against M1, M3 against M1 and M2 against M3
for each representative regime, and `samplesize_contrasts.csv` reports M2
against M3 at each study size, both with paired MCSEs. These are the quantities
that decide whether a margin between two estimators is resolved; a difference
between two separately quoted means is not, and the between-study standard
deviation is a measure of spread across studies rather than of uncertainty in
their difference.

Figure 1 keeps between-study standard deviations as its error bars, matching its
caption; its MCSEs are in `table2.csv`. Figures 3 and 4 show MCSEs directly. The
right-hand panel of Figure 3 shows the paired M2-M3 contrast with bars at twice
its MCSE, which is the threshold the `resolved` column of
`samplesize_contrasts.csv` applies.

## Layout

```
src/spikein.py      generative model, estimator family, cross-validation,
                    closed-form RMSE, Monte Carlo standard errors
src/robustness.py   generative model with assumption violations
src/assess.py       decision tool for your own assay
src/calibrate.py    published dispersion values and unit conversions
run_all.py          runs all four stages in order
run_analysis.py     driver; Table 2 and Figures 1-3
run_robustness.py   driver; Figure 4
run_validation.py   driver; Table 3
run_calibration.py  driver; Table 4
requirements.txt    pinned dependencies
results/            archived outputs; committed, so the release is self-contained
```

## The model

For sample *i*:

| term | meaning |
|---|---|
| `N` | true log₁₀ target concentration, drawn uniformly on [2, 6] |
| `s` | shared processing perturbation, affects target **and** control |
| `t` | target-specific perturbation |
| `c` | control-specific perturbation |
| `eT`, `eC` | measurement noise, one per channel |

    Ct_T = a_T − β(N + s + t) + eT
    Ct_C = a_C − β(K + s + c) + eC

The spike-in carries information about `s` only. Whether normalising by it helps
therefore depends on how much of the target's error budget is shared against how
noisy the control channel is in its own right.

## Estimators

| id | form | assumption |
|---|---|---|
| M1 | `N ~ Ct_T` | control ignored |
| M2 | `N ~ (Ct_T − Ct_C)` | control tracks target exactly; coefficient fixed at 1 |
| M3 | `N ~ Ct_T + Ct_C` | coupling estimated from data |
| M4 | `N ~ Ct_T + Ct_C + Ct_T·Ct_C` | coupling varies with abundance (channels centred first) |

`spikein.py` also provides `theta_moments` (method-of-moments θ from the two
channels) and an oracle θ\* estimator via `estimator_set(..., include_aux=True)`.

M4 centres both channels before forming the product, because on the raw cycle
scale `Ct_T·Ct_C` is roughly 660 and nearly collinear with the main effects.

A penalised variant is retained in `spikein.py` as `Ridge` but is not part of the
reported comparison. On the raw cycle scale the cross-product matrix has diagonal
entries of order 10⁴, so any penalty small enough to be defensible a priori has
no effect, and one large enough to matter simply reproduces M1.

## Robustness

`run_robustness.py` re-runs the comparison under seven departures from the
model's assumptions: unequal amplification efficiency (90% and 80% in the
control channel), heavy-tailed errors, batch perturbations shared only in part
between the channels, variation in the spike-in amount itself, and censoring of
threshold cycles at the detection limit. The sign of the criterion holds in every
case, and every violation widens the margin by which the estimated coefficient
beats fixed ΔCt.

## Assessing your own assay

```python
from assess import assess_normalisation

out = assess_normalisation(target_ct, control_ct, processing_unit)
print(out)
```

Returns the estimated variance components, θ\*, the decision contrast
`D = σs² − σc² − τC²/β²` with a bootstrap interval, and one of three
recommendations: apply ΔCt, do not apply it, or estimate the coefficient.

The design it needs: replicate aliquots of one homogenised material carried
through the complete workflow (the *processing units*), with the spike added at
the intended stage, and target and control measured in replicate within each
aliquot.

On simulated designs of 24 aliquots with three replicates each, across 500
studies per setting (Table 3), the procedure returns the correct recommendation in 100.0%
of studies at a favourable setting and 99.6% at an unfavourable one, with no
directionally incorrect call. Near the boundary the interval is indeterminate
in 89.6% of studies, which is the correct behaviour. Whichever variance
component is genuinely small is biased upward by the non-negativity constraint,
and θ* is well centred but imprecise — at the boundary a design of this size
locates it only to within roughly ±0.27. The sign of the contrast that drives
the decision is far better determined than the components it is built from.
Run `python3 run_validation.py` to reproduce.

## Empirical calibration

`src/calibrate.py` converts published validation statistics onto the log₁₀
scale, and `run_calibration.py` writes the result as Table 4. Three conversions
cover what the literature reports:

- control Ct standard deviation at fixed input → divide by β
- repeatability limit *r* → *s*ᵣ = *r* / 2.8, already on the log₁₀ scale
- recovery as mean ± SD per cent → `SD(log₁₀ R) = √(ln(1 + CV²)) / ln 10`

Each entry records the value *as reported* in the source together with the
conversion applied, so the table is derived from the primary figures rather than
transcribed from them. A reported dispersion constrains `σs² + σc² + τC²/β²`, the
denominator of θ\*, and not the split between the three components, so it bounds
the magnitude of control-channel dispersion without locating an assay relative to
the benefit boundary. The values collected here span 0.04–0.80 log₁₀ units, which brackets the
0.05–0.80 grid used in the simulations.

## Licence and citation

MIT (see `LICENSE`). Citation metadata in `CITATION.cff`.
