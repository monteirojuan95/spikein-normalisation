# Changelog

## 1.3.0 — 2026-07-29

Pre-release audit against the submitted manuscript. No result changes: every
number in every table is unaltered and still reproduces bit-for-bit.

### Fixed

- **Table numbering.** The manuscript cites the validation-design table in the
  Results and the published-dispersion table in the Discussion, so by order of
  first citation they are Tables 3 and 4 respectively. The repository had them
  the other way round. `run_validation.py` now writes `table3.csv` and
  `run_calibration.py` writes `table4.csv`; `calibrate.table3_rows` is renamed
  `table4_rows`. File contents are unchanged.
- **`results/robustness.csv` was malformed.** The scenario labels
  `batch effect, rho = 0.5` and `batch effect, rho = 0` contain an unquoted
  comma, so six of the twenty-four rows carried sixteen fields instead of
  fifteen and any CSV reader silently shifted their columns. The labels now use
  parentheses.
- **`requirements.txt` pinned `scipy`, which nothing imports.** Removed, and
  the manuscript's Implementation section corrected to match.
- **`README` reproduction section.** Claimed a single seeded generator when
  `run_validation.py` uses its own; claimed run times of ~20, ~10 and ~12
  minutes for stages that take ~50 s, ~10 s and ~2 min; and claimed every
  published dispersion falls inside the 0.05–0.80 simulation grid when the
  Raymond dates entry converts to 0.039 and the MNV lettuce entry to 0.804.
- **`run_analysis.py` docstring** claimed to regenerate every table and figure
  in the manuscript. It produces Table 2 and Figures 1-3.
- **`calibrate.PUBLISHED`** labelled the Uhrbrand entry `vMC`; the mengovirus
  strain is vMC₀, as the manuscript has it.

### Added

- **`run_all.py`**, which runs the four stages in order, each in its own
  process so that each starts from its own seeded generator. The manuscript
  states that the analysis is reproduced from a single documented command;
  this makes that true without collapsing the stages into one script.
- **Second panel in Figure 3**, plotting the paired M2-M3 contrast with bars at
  twice its Monte Carlo standard error. The crossover the text argues about is
  about one part in two hundred, and against the distance to M1 on the original
  single-panel plot it was not legible.
- **`CITATION.cff`**: second author, `type`, `abstract`, `keywords`, and
  placeholders for `repository-code` and `doi`.

### Changed

- **Figure 2** now uses the same estimator palette as Figures 1 and 3 in its
  right-hand panel, instead of `Set2`; its legend moves below the axes, where
  it no longer covers the region in which M1 is optimal; and its colourbar is
  drawn with `extend='max'`, since the surface reaches +291% against a scale
  that stops at +200%.

## 1.2.0 — 2026-07-29

Adds the evaluation of the operational decision procedure, which the manuscript
previously asserted without supporting code.

### Added

- **`run_validation.py`** and **Table 4.** Simulates replicate validation
  designs — 24 independently processed aliquots of one homogenised material,
  three technical measurements of each channel per aliquot — and runs
  `assess.assess_normalisation` on each. Reports recovery of the variance
  components, of theta*, and of the decision contrast D, together with the
  proportion of studies in which the bootstrap interval for D yields each of
  the three recommendations. 500 studies per setting, 2000 bootstrap resamples,
  seed 20260729. Writes `results/table4.csv` and
  `results/validation_studies.csv`, the latter holding all 1500 per-study
  records.
- The generative structure this needs differs from `spikein.simulate`, which
  varies the concentration across samples and measures each once. In a
  validation design the concentration is fixed, the processing perturbations
  are drawn per aliquot, and the measurement error is drawn per replicate.

### Fixed

- **`requirements.txt`** pinned versions that cannot be installed on Python 3.9
  and listed `pandas`, which nothing imports. Now pinned to the versions the
  archived results were produced with, matching `results/run_metadata.json`.
- **`src/assess.py`** docstring advertised a `true_concentration` argument the
  function does not accept.
- **`src/spikein.py`** docstring described M6 and M7 as comparisons the
  Discussion refers to. Neither is instantiated by any driver script, and the
  Discussion states that the moment shortcut was not evaluated.
- **`run_analysis.py`** docstring omitted `table2_contrasts.csv` and
  `samplesize_contrasts.csv` from the list of files it writes.
- **README** documented commands as `python`, which does not resolve on macOS
  or most Linux distributions, and omitted `run_validation.py`. Its summary of
  the validation-design behaviour predated the numbers now in Table 4 and
  described the component bias in the wrong direction: the non-negativity
  constraint biases whichever component is genuinely small *upward*, not
  downward.

## 1.1.0 — 2026-07-29

Brings the repository into line with the submitted manuscript.

### Removed

- **Figure 5 and the code that produced it.** `run_calibration.py` no longer
  draws the parameter plane, and `calibrate.arc()` — whose only consumer was
  that figure — has been deleted. A published dispersion constrains the
  denominator of θ\* without fixing the split between its three components, so
  plotting it against the benefit boundary invited an inference it cannot
  support. The material is now Table 3.

### Added

- **Monte Carlo standard errors**, as described in the Methods, on every
  reported performance measure. `spikein.mcse` and `spikein.summarise` compute
  the between-study standard deviation divided by the square root of the number
  of studies. MCSE columns now appear in `table2.csv`, `samplesize.csv`,
  `robustness.csv` and `analytic_check.csv`, and `grid_mcse.csv` holds the
  per-cell MCSE of the sensitivity grid.
- **`spikein.ratio_mcse` / `spikein.pct_change_mcse`** for contrasts between
  estimators. The estimators share synthetic studies and cross-validation folds,
  so their per-study errors are strongly correlated; the delta method is applied
  to the paired study-level values rather than treating the two means as
  independent. Point estimates are unchanged, so every percentage reported in the
  manuscript is reproduced exactly.
- **`table2_contrasts.csv`** — M2 vs M1, M3 vs M1 and M2 vs M3 per regime, with
  paired MCSEs. These are the percentages quoted in the Results.
- **`samplesize_contrasts.csv`** — M2 vs M3 at each study size, with paired
  MCSEs and a flag for whether the margin exceeds twice its Monte Carlo error.
  The crossover in Figure 3 turns on margins of a few thousandths of a log₁₀
  unit, which only the paired contrast can resolve.
- **`FIGURES` manifest** in `run_analysis.py`, mirrored into
  `run_metadata.json`, fixing the figure numbering in one place and recording
  which script produces each panel.
- **`CHANGELOG.md`** (this file).

### Changed

- **`run_calibration.py` writes `table3.csv`**, replacing `calibration.csv` and
  `figure5.png`.
- **`calibrate.PUBLISHED` now stores values as reported in each source**
  (threshold-cycle SD, repeatability limit, or recovery mean ± SD) together with
  the conversion applied, and derives the log₁₀ column from them. Previously the
  already-converted numbers were hard-coded, so the conversion functions were
  never exercised on the published data and the table could not be checked
  against its primary sources.
  - **Corrects one row.** Raymond [21], leafy greens with MNV, was stored as
    0.098–0.464 log₁₀. Converting the reported 28 ± 29% recovery gives 0.37,
    which is the value printed in the manuscript. The other ten rows were
    unaffected.
- **`calibrate.from_repeatability_limit`** added for the ISO 15216 rows, which
  report a repeatability limit *r* rather than a standard deviation
  (*s*ᵣ = *r* / 2.8, already on the log₁₀ scale and so not divided by β).
- **`robustness.compare`** returns per-study values rather than their means, so
  that the caller can attach MCSEs and form contrasts on paired studies.
- **Figures 3 and 4** now display MCSEs. Figure 1 keeps between-study standard
  deviations, matching its caption.
- **README** documents the figure-to-script mapping, the MCSE definitions and
  the absence of Figure 5.

## 1.0.0 — 2026-07-28

Initial release.
