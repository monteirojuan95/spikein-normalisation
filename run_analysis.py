#!/usr/bin/env python3
"""Table 2 and Figures 1-3. Run via run_all.py to regenerate the whole analysis.

    python run_analysis.py            # full run
    python run_analysis.py --quick    # reduced replication, for a smoke test

Writes to results/: table2.csv, table2_contrasts.csv, figure1.png,
figure2.png, figure3.png, grid.csv, grid_mcse.csv, samplesize.csv,
samplesize_contrasts.csv, analytic_check.csv and run_metadata.json.

Every reported performance measure carries a Monte Carlo standard error,
computed as the between-study standard deviation divided by the square root of
the number of independent synthetic studies, as stated in the Methods. Figure 1
retains between-study standard deviations as its error bars, matching its
caption; the MCSEs are in table2.csv.
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import spikein as sp

SEED = 20260728
OUT = os.path.join(os.path.dirname(__file__), 'results')

REGIMES = {'A': (0.60, 0.10), 'B': (0.35, 0.25),
           'C': (0.10, 0.10), 'D': (0.10, 0.50), 'E': (0.60, 0.50)}
GRID = np.linspace(0.05, 0.80, 16)
NS = [15, 20, 30, 40, 60, 100, 200]
COLOURS = {'M1 naive': '#9e9e9e', 'M2 dCt': '#c0392b', 'M3 partial': '#2980b9',
           'M4 interaction': '#8e44ad'}

# Manuscript figure numbering, in order of first citation in the text.
# Numbers live here so that renumbering is a one-line change; the file names
# written to results/ follow from it. There is no figure 5: the parameter-plane
# figure was withdrawn and its content is now Table 4 (see run_calibration.py).
FIGURES = {
    1: ('figure1.png', 'run_analysis.py',   'RMSE across five representative regimes'),
    2: ('figure2.png', 'run_analysis.py',   'Two-parameter sensitivity grid, 16 x 16'),
    3: ('figure3.png', 'run_analysis.py',   'Dependence on study size'),
    4: ('figure4.png', 'run_robustness.py', 'Departures from the model assumptions'),
}


def fig_path(n):
    return os.path.join(OUT, FIGURES[n][0])


def four(ss, sc, rng):
    """Estimator family without the penalised variant (see manuscript)."""
    return [e for e in sp.estimator_set(ss, sc, rng=rng) if e.name != 'M5 ridge']


def studies(n, ss, sc, est, n_studies, repeats, rng):
    """Per-study CV RMSE for each estimator: (n_studies, n_estimators)."""
    runs, names = [], [e.name for e in est]
    for _ in range(n_studies):
        N, tT, tC = sp.simulate(n, ss, sc, rng=rng)
        res, names = sp.cv_rmse(N, tT, tC, est, V=5, repeats=repeats, rng=rng)
        runs.append([res[k] for k in names])
    return np.asarray(runs, float), names


def run_regimes(n_studies, repeats, rng):
    """Table 2 and Figure 1: five representative regimes."""
    out, raw, names = {}, {}, None
    for r, (ss, sc) in REGIMES.items():
        runs, names = studies(40, ss, sc, four(ss, sc, rng), n_studies, repeats, rng)
        out[r] = sp.summarise(runs, names)
        raw[r] = {k: runs[:, j] for j, k in enumerate(names)}
    return out, raw, names


# Contrasts quoted in the Results, as (label, numerator, denominator).
CONTRASTS = [('M2_vs_M1', 'M2 dCt', 'M1 naive'),
             ('M3_vs_M1', 'M3 partial', 'M1 naive'),
             ('M2_vs_M3', 'M2 dCt', 'M3 partial')]


def regime_contrasts(raw):
    """Percentage change between estimators, with the MCSE of the paired ratio."""
    recs = []
    for r in REGIMES:
        rec = {'regime': r}
        for lab, num, den in CONTRASTS:
            pct, se = sp.pct_change_mcse(raw[r][num], raw[r][den])
            rec[f'{lab}_pct'] = round(pct, 2)
            rec[f'{lab}_pct_mcse'] = round(se, 2)
        recs.append(rec)
    return recs


def run_grid(n_studies, repeats, rng):
    """Figure 2: 16 x 16 sensitivity grid, with the MCSE of each cell."""
    rel = np.zeros((len(GRID), len(GRID)))
    rel_se = np.zeros((len(GRID), len(GRID)))
    best = np.empty((len(GRID), len(GRID)), dtype=object)
    for i, sc in enumerate(GRID):
        for j, ss in enumerate(GRID):
            runs, names = studies(40, ss, sc, four(ss, sc, rng),
                                  n_studies, repeats, rng)
            col = {k: runs[:, c] for c, k in enumerate(names)}
            rel[i, j], rel_se[i, j] = sp.pct_change_mcse(col['M2 dCt'],
                                                         col['M1 naive'])
            m = {k: v.mean() for k, v in col.items()}
            best[i, j] = min(m, key=m.get)
    return rel, rel_se, best


def run_samplesize(n_studies, repeats, rng):
    """Figure 3: sample-size dependence at three regimes."""
    out, raw = {}, {}
    for label, (ss, sc) in {'favourable': (0.50, 0.15),
                            'intermediate': (0.35, 0.25),
                            'marginal': (0.30, 0.30)}.items():
        curves, cols = {}, {}
        for n in NS:
            runs, names = studies(n, ss, sc, four(ss, sc, rng),
                                  n_studies, repeats, rng)
            curves[n] = sp.summarise(runs, names)
            cols[n] = {k: runs[:, j] for j, k in enumerate(names)}
        out[label], raw[label] = curves, cols
    return out, raw


def samplesize_contrasts(raw):
    """M2 against M3 at each study size, on the paired studies.

    The crossover in Figure 3 turns on a margin of a few thousandths of a log10
    unit. Because both estimators see the same studies and folds, the paired
    contrast is the quantity that decides whether that margin is resolved; the
    difference of two independently-quoted means is not.
    """
    recs = []
    for label, cols in raw.items():
        for n in NS:
            pct, se = sp.pct_change_mcse(cols[n]['M2 dCt'], cols[n]['M3 partial'])
            recs.append({'regime': label, 'n': n,
                         'M2_vs_M3_pct': round(pct, 3),
                         'M2_vs_M3_pct_mcse': round(se, 3),
                         'resolved': abs(pct) > 2 * se})
    return recs


def analytic_check(rows):
    """Compare simulation with the closed form, inflated to the training-fold size."""
    lines = []
    for r, (ss, sc) in REGIMES.items():
        an = sp.analytic_rmse(ss, sc, n=40, V=5)
        for k, v in an.items():
            sim = rows[r][k]['mean']
            lines.append({'regime': r, 'estimator': k, 'analytic': round(v, 4),
                          'simulated': round(sim, 4),
                          'simulated_mcse': round(rows[r][k]['mcse'], 4),
                          'deviation_pct': round(100 * (v - sim) / sim, 2)})
    return lines


def write_csv(path, header, records):
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')
        for rec in records:
            f.write(','.join(str(v) for v in rec) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ns, rp = (10, 2) if a.quick else (200, 5)
    ng, rg = (3, 1) if a.quick else (40, 3)
    nss, rss = (10, 2) if a.quick else (120, 3)

    rng = np.random.default_rng(SEED)
    print(f'seed {SEED}; {"quick" if a.quick else "full"} run')

    print('  representative regimes ...')
    rows, raw, names = run_regimes(ns, rp, rng)

    head = ['regime'] + [f'{k}{sfx}' for k in names
                         for sfx in ('', '_sd', '_mcse')]
    write_csv(f'{OUT}/table2.csv', head,
              [[r] + [f'{rows[r][k][f]:.4f}' for k in names
                      for f in ('mean', 'sd', 'mcse')] for r in REGIMES])
    worst_se = max(rows[r][k]['mcse'] for r in REGIMES for k in names)
    print(f'    largest MCSE in Table 2: {worst_se:.4f} log10 units')

    con = regime_contrasts(raw)
    write_csv(f'{OUT}/table2_contrasts.csv', list(con[0]),
              [list(d.values()) for d in con])
    for d in con:
        print(f"    {d['regime']}  M2 vs M1 {d['M2_vs_M1_pct']:+7.1f}% "
              f"(MCSE {d['M2_vs_M1_pct_mcse']:.2f})   M2 vs M3 "
              f"{d['M2_vs_M3_pct']:+6.1f}% (MCSE {d['M2_vs_M3_pct_mcse']:.2f})")

    rec = analytic_check(rows)
    write_csv(f'{OUT}/analytic_check.csv', list(rec[0]),
              [list(d.values()) for d in rec])
    worst = max(abs(d['deviation_pct']) for d in rec)
    print(f'    closed form agrees to within {worst:.1f}%')

    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.20
    for i, k in enumerate(names):
        ax.bar(np.arange(5) + (i - 1.5) * w,
               [rows[r][k]['mean'] for r in REGIMES], w,
               yerr=[rows[r][k]['sd'] for r in REGIMES], capsize=3,
               label=k, color=COLOURS[k])
    ax.set_xticks(range(5)); ax.set_xticklabels(REGIMES)
    ax.set_xlabel('Simulated regime')
    ax.set_ylabel('Cross-validated RMSE (log$_{10}$ copies/mL)')
    ax.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(); fig.savefig(fig_path(1), dpi=200); plt.close(fig)

    print('  sensitivity grid ...')
    rel, rel_se, best = run_grid(ng, rg, rng)
    np.savetxt(f'{OUT}/grid.csv', rel, delimiter=',', fmt='%.3f')
    np.savetxt(f'{OUT}/grid_mcse.csv', rel_se, delimiter=',', fmt='%.3f')
    print(f'    largest MCSE on the grid: {rel_se.max():.2f} points')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    im = axes[0].imshow(rel, origin='lower', cmap='RdBu_r', vmin=-100, vmax=200,
                        extent=[GRID[0], GRID[-1], GRID[0], GRID[-1]], aspect='auto')
    axes[0].contour(GRID, GRID, rel, levels=[0], colors='k', linewidths=1.6)
    axes[0].plot(GRID, GRID, '--', color='0.4', lw=0.9)
    axes[0].set_title('Relative RMSE, $\\Delta$Ct vs naive (%)')
    fig.colorbar(im, ax=axes[0], extend='max')   # surface reaches +291%
    # Same estimator colours as Figures 1 and 3, so a reader does not have to
    # re-learn the palette between panels.
    labs = [k for k in COLOURS if any(best[i, j] == k
                                      for i in range(16) for j in range(16))]
    codes = np.array([[labs.index(best[i, j]) for j in range(16)] for i in range(16)])
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    axes[1].imshow(codes, origin='lower', cmap=ListedColormap([COLOURS[l] for l in labs]),
                   vmin=-0.5, vmax=len(labs) - 0.5,
                   extent=[GRID[0], GRID[-1], GRID[0], GRID[-1]], aspect='auto')
    axes[1].plot(GRID, GRID, '--', color='0.9', lw=0.9)
    axes[1].set_title('Best-performing estimator')
    axes[1].legend(handles=[Patch(facecolor=COLOURS[l], label=l) for l in labs],
                   frameon=False, fontsize=8, loc='upper center', ncol=len(labs),
                   bbox_to_anchor=(0.5, -0.18))
    for ax in axes:
        ax.set_xlabel('Shared processing variability  $\\sigma_s$ (log$_{10}$)')
        ax.set_ylabel('Control-specific variability  $\\sigma_c$ (log$_{10}$)')
    fig.tight_layout(); fig.savefig(fig_path(2), dpi=200); plt.close(fig)

    print('  sample-size dependence ...')
    ss_out, ss_raw = run_samplesize(nss, rss, rng)
    head = ['regime', 'n'] + [f'{k}{sfx}' for k in names for sfx in ('', '_mcse')]
    write_csv(f'{OUT}/samplesize.csv', head,
              [[lab, n] + [f'{curves[n][k][f]:.4f}' for k in names
                           for f in ('mean', 'mcse')]
               for lab, curves in ss_out.items() for n in NS])

    sc_rec = samplesize_contrasts(ss_raw)
    write_csv(f'{OUT}/samplesize_contrasts.csv', list(sc_rec[0]),
              [list(d.values()) for d in sc_rec])
    fav = [d for d in sc_rec if d['regime'] == 'favourable']
    print('    favourable regime, M2 vs M3 on paired studies:')
    for d in fav:
        flag = 'resolved' if d['resolved'] else 'within MC error'
        print(f"      n={d['n']:>3}  {d['M2_vs_M3_pct']:+6.2f}% "
              f"(MCSE {d['M2_vs_M3_pct_mcse']:.2f})  {flag}")

    # Two panels. The left shows the levels; the right shows the paired M2-M3
    # contrast, which is the quantity the text actually argues about. On the
    # left panel alone the crossover is a few thousandths of a log10 unit and
    # is swamped by the distance to M1, so the claim is not legible.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for k in names:
        axes[0].errorbar(NS, [ss_out['favourable'][n][k]['mean'] for n in NS],
                         yerr=[ss_out['favourable'][n][k]['mcse'] for n in NS],
                         fmt='o-', label=k, color=COLOURS[k], ms=4,
                         elinewidth=0.8, capsize=2)
    axes[0].set_ylabel('CV RMSE (log$_{10}$)')
    axes[0].set_title('Cross-validated error')
    axes[0].legend(frameon=False, fontsize=9)

    fav = {d['n']: d for d in sc_rec if d['regime'] == 'favourable'}
    pct = np.array([fav[n]['M2_vs_M3_pct'] for n in NS], float)
    se = np.array([fav[n]['M2_vs_M3_pct_mcse'] for n in NS], float)
    axes[1].axhline(0, color='0.6', lw=0.9)
    axes[1].errorbar(NS, pct, yerr=2 * se, fmt='o-', color='#34495e', ms=4,
                     elinewidth=0.9, capsize=3)
    axes[1].set_ylabel('$\\Delta$Ct vs estimated coefficient (%)')
    axes[1].set_title('Paired contrast, M2 against M3')
    axes[1].annotate('$\\Delta$Ct better', xy=(0.03, 0.06),
                     xycoords='axes fraction', fontsize=8, color='0.35')
    axes[1].annotate('estimated coefficient better', xy=(0.03, 0.92),
                     xycoords='axes fraction', fontsize=8, color='0.35')
    for ax in axes:
        ax.set_xscale('log'); ax.set_xticks(NS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel('Samples per study (n)')
    fig.tight_layout(); fig.savefig(fig_path(3), dpi=200); plt.close(fig)

    with open(f'{OUT}/run_metadata.json', 'w') as f:
        json.dump({'seed': SEED, 'numpy': np.__version__,
                   'matplotlib': matplotlib.__version__,
                   'python': sys.version.split()[0],
                   'studies_per_regime': ns, 'cv_repeats': rp,
                   'studies_per_grid_cell': ng, 'studies_per_sample_size': nss,
                   'mcse': 'between-study SD / sqrt(number of studies)',
                   'figures': {str(n): {'file': v[0], 'script': v[1],
                                        'content': v[2]}
                               for n, v in FIGURES.items()}},
                  f, indent=2)
    print(f'done -> {OUT}/')


if __name__ == '__main__':
    main()
