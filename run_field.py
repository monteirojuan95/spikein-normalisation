#!/usr/bin/env python3
"""Application of the criterion to field surveillance data (Figure 5, Table 7).

Data: CentrEau-COVID Quebec City dataset, v1.04 (Therrien et al. 2021,
doi:10.5281/zenodo.5597158; github.com/modelEAU/CentrEau-COVID-Dataset-Qc,
CC-BY 4.0). Two water resource recovery facilities, 15 Jan - 7 Oct 2021. BRSV
is spiked before extraction; SARS-CoV-2 N2, PMMoV and BRSV are run as a triplex
RT-qPCR in 2-3 wells per sample. The criterion is src/assess.py, unchanged.

The dataset is not redistributed here. To obtain it:

    git clone https://github.com/modelEAU/CentrEau-COVID-Dataset-Qc data/qc
    unzip data/qc/data/centreau_qc.zip -d data/centreau_qc

    python3 run_field.py                      # reads data/centreau_qc/
    python3 run_field.py --data <directory>   # or any directory with the CSVs

Outputs in results/: field_table.csv (Table 7), field_external.csv,
field_meta.json, figure5.png/.pdf/.eps. Seed 20261107.
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.join(HERE, 'data', 'centreau_qc'))
ap.add_argument('--out', default=os.path.join(HERE, 'results'))
args = ap.parse_args()
DATA, OUT = args.data, args.out
if not os.path.exists(os.path.join(DATA, 'WWMeasure.csv')):
    sys.exit(f'dataset not found in {DATA}; see the instructions at the top of run_field.py')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, 'src'))
from assess import assess_normalisation, _unit_stats, _components

SEED = 20261107
BETA = 3.32
SWITCH = pd.Timestamp('2021-05-04')      # first flow-proportional composite
K_TREND = 15                              # samples in the common rolling-median trend
TARGETS = {'covn2': 'SARS-CoV-2 N2', 'npmmov': 'PMMoV'}
SITES = {'qc_01': 'East', 'qc_02': 'West'}
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- data
w = pd.read_csv(f'{DATA}/WWMeasure.csv', low_memory=False)
s = pd.read_csv(f'{DATA}/Sample.csv')
c = pd.read_csv(f'{DATA}/CovidPublicHealthData.csv', parse_dates=['date'])
ct = w[w.unit == 'ct'].copy()
n_flag = int(ct.qualityFlag.astype(str).eq('True').sum())
ct = ct[~ct.qualityFlag.astype(str).eq('True')]
wells = (ct.pivot_table(index=['sampleID', 'analysisDate', 'index'], columns='type', values='value')
         .reset_index().merge(s[['sampleID', 'siteID', 'collection', 'dateTimeEnd']], on='sampleID'))
wells['date'] = pd.to_datetime(wells.dateTimeEnd.str[:10])
wells['unit'] = wells.sampleID + '|' + wells.analysisDate
wells['period'] = np.where(wells.date < SWITCH, 'P1', 'P2')
cases = c[c.type == 'conf'].set_index('date').value.sort_index().clip(lower=0)
inc_fwd = cases.rolling(7, center=True).mean().shift(-3)   # mean of days 0..6 after sampling


def common_detrend(d, tcol):
    """Subtract a centred rolling median (over samples, in date order) from
    target and control alike, so slow drifts common to both, and slow drifts
    in either that happen to coincide, are removed before estimation."""
    u = d.groupby('unit').agg(T=(tcol, 'mean'), C=('nbrsv', 'mean'), date=('date', 'first')).sort_values('date')
    trT = u['T'].rolling(K_TREND, center=True, min_periods=5).median()
    trC = u['C'].rolling(K_TREND, center=True, min_periods=5).median()
    o = d[d.unit.isin(u.index)].copy()
    o['T_use'] = o[tcol] - o.unit.map(trT)
    o['C_use'] = o['nbrsv'] - o.unit.map(trC)
    return o.dropna(subset=['T_use', 'C_use'])


def block_ci(d, B=4000, r_routine=2):
    """Weekly block bootstrap of D; tau_C^2 re-estimated in each resample."""
    order = np.unique(d.unit.values)
    mT, mC, n, ssT, ssC = _unit_stats(d.T_use.values, d.C_use.values, d.unit.values)
    dates = d.groupby('unit').date.first().reindex(order)
    wk = ((dates - dates.min()).dt.days // 7).values
    weeks = np.unique(wk)
    out = []
    for _ in range(B):
        idx = np.concatenate([np.where(wk == p)[0] for p in rng.choice(weeks, size=len(weeks))])
        tau = ssC[idx].sum() / (n[idx] - 1).sum()
        s2, c2 = _components(mT[idx], mC[idx], (1 / n[idx]).mean(), tau, BETA)
        out.append(s2 - c2 - tau / (r_routine * BETA ** 2))
    return np.percentile(out, [2.5, 97.5])


rows, ext = [], []
for target, tname in TARGETS.items():
    for site, sname in SITES.items():
        for period in ['P1', 'P2']:
            d0 = wells[(wells.siteID == site) & (wells.period == period)].dropna(subset=[target, 'nbrsv'])
            for est in ['raw', 'detrended']:
                d = d0.assign(T_use=d0[target], C_use=d0.nbrsv) if est == 'raw' else common_detrend(d0, target)
                a = assess_normalisation(d.T_use.values, d.C_use.values, d.unit.values,
                                         r_routine=2, n_boot=4000, seed=SEED)
                lo, hi = block_ci(d)
                rows.append(dict(target=tname, site=sname, period=period, estimation=est,
                                 samples=a.n_units, wells=len(d), tau_T2=a.tau_T2, tau_C2=a.tau_C2,
                                 sigma_s2=a.sigma_s2, sigma_c2=a.sigma_c2, theta_star=a.theta_star,
                                 D=a.contrast, D_lo=lo, D_hi=hi, t=a.test_t, df=a.test_df, p=a.test_p,
                                 call=a.recommendation.split('.')[0]))
            # external check: N2 only (PMMoV is not expected to track incidence)
            if target == 'covn2':
                th = rows[-1]['theta_star']          # theta* from the detrended fit
                u = d0.groupby('unit').agg(T=('covn2', 'mean'), C=('nbrsv', 'mean'), date=('date', 'first'))
                u['L'] = np.log10(u.date.map(inc_fwd) + 1); u = u.dropna()
                X = {'none': -u['T'].values, 'unit (dCt)': -(u['T'] - u['C']).values,
                     'estimated theta*': -(u['T'] - th * u['C']).values}
                r = lambda x, l: np.corrcoef(x, l)[0, 1]
                obs = {k: r(v, u.L.values) for k, v in X.items()}
                wk = ((u.date - u.date.min()).dt.days // 7).values; weeks = np.unique(wk)
                bd = {k: [] for k in X if k != 'none'}
                for _ in range(4000):
                    idx = np.concatenate([np.where(wk == p)[0] for p in rng.choice(weeks, size=len(weeks))])
                    r0 = r(X['none'][idx], u.L.values[idx])
                    for k in bd:
                        bd[k].append(r(X[k][idx], u.L.values[idx]) - r0)
                for k in X:
                    lo, hi = (np.nan, np.nan) if k == 'none' else np.percentile(bd[k], [2.5, 97.5])
                    ext.append(dict(site=sname, period=period, normalisation=k, theta_used=1.0 if k == 'unit (dCt)' else (th if k != 'none' else 0.0),
                                    samples=len(u), r=obs[k], diff_vs_none=obs[k] - obs['none'], diff_lo=lo, diff_hi=hi))

tab = pd.DataFrame(rows); tab.to_csv(f'{OUT}/field_table.csv', index=False, float_format='%.10g')
ex = pd.DataFrame(ext); ex.to_csv(f'{OUT}/field_external.csv', index=False, float_format='%.10g')

# descriptive and batch structure
desc = {}
for p in ['P1', 'P2']:
    x = wells[wells.period == p]
    u = x.groupby('unit').agg(C=('nbrsv', 'mean'), ad=('analysisDate', 'first'))
    within = u.groupby('ad').C.transform(lambda v: v - v.mean()).var()
    lag = (pd.to_datetime(x.analysisDate) - x.date).dt.days
    desc[p] = dict(samples=int(x.unit.nunique()), wells=int(len(x)), wells_per_sample=round(len(x) / x.unit.nunique(), 2),
                   analysis_batches=int(x.analysisDate.nunique()), extraction_lag_days_median=float(lag.median()),
                   extraction_lag_days_max=int(lag.max()), n2_nondetect_wells=int(x.covn2.isna().sum()),
                   brsv_var_between_batches=round(1 - within / u.C.var(), 2),
                   collection=x.drop_duplicates('unit').collection.value_counts().to_dict())
wr = wells.dropna(subset=['covn2', 'nbrsv', 'npmmov']).copy()
for col in ['covn2', 'nbrsv', 'npmmov']:
    wr[col] = wr[col] - wr.groupby('unit')[col].transform('mean')
meta = dict(source='CentrEau-COVID Quebec City, Zenodo 10.5281/zenodo.5597158 (v1.04)', WWMeasure_md5=hashlib.md5(open(os.path.join(DATA, 'WWMeasure.csv'), 'rb').read()).hexdigest(), qc_flagged_ct_removed=n_flag,
            wells=int(len(wells)), samples=int(wells.unit.nunique()), period_switch=str(SWITCH.date()),
            trend_window_samples=K_TREND, seed=SEED, beta=BETA, periods=desc,
            within_well_corr=wr[['covn2', 'nbrsv', 'npmmov']].corr().round(3).to_dict())
json.dump(meta, open(f'{OUT}/field_meta.json', 'w'), indent=1, default=str)

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
_fam = 'Inter' if any('Inter' == f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': _fam, 'font.size': 9, 'ps.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'pdf.fonttype': 42})
BLUE, ORANGE, INK, MUTED, GRID = '#34495e', '#2980b9', '#000000', '0.35', '0.9'

fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], height_ratios=[1, 1, 1], wspace=0.42, hspace=0.35)

# A: decision contrast
axA = fig.add_subplot(gs[:, 0])
cells = [(t, si, p) for t in TARGETS.values() for si in SITES.values() for p in ['P1', 'P2']]
ypos = np.arange(len(cells))[::-1].astype(float)
for (t, si, p), y in zip(cells, ypos):
    for est, col, off, mk in [('raw', BLUE, 0.17, 'o'), ('detrended', ORANGE, -0.17, 's')]:
        r_ = tab[(tab.target == t) & (tab.site == si) & (tab.period == p) & (tab.estimation == est)].iloc[0]
        axA.plot([r_.D_lo, r_.D_hi], [y + off] * 2, color=col, lw=1.6, solid_capstyle='round')
        axA.plot(r_.D, y + off, mk, color=col, ms=5.5, mec='white', mew=0.8)
axA.axvline(0, color=MUTED, lw=0.8)
axA.set_yticks(ypos)
axA.set_yticklabels([f'{t.replace("SARS-CoV-2 ", "")}  {si}  {p}' for t, si, p in cells])
axA.set_ylim(-0.7, len(cells) - 0.3)
for y in [3.5]:
    axA.axhline(y, color=GRID, lw=0.8)
axA.set_xlabel('Decision contrast D (log$_{10}$ units$^2$)\n< 0: ΔCt adds error   |   > 0: ΔCt removes error')
axA.grid(axis='x', color=GRID, lw=0.6); axA.set_axisbelow(True)
axA.legend(handles=[Line2D([], [], color=BLUE, marker='o', lw=1.6, mec='white', label='raw threshold cycles'),
                    Line2D([], [], color=ORANGE, marker='s', lw=1.6, mec='white', label='common trend removed')],
           loc='upper left', frameon=False, fontsize=7.5, ncol=2, bbox_to_anchor=(-0.02, -0.115))
axA.set_title('A  Criterion by target, plant and period', loc='left', fontsize=9, fontweight='bold', color=INK)

# B: weekly series, both plants pooled
u = wells.groupby('unit').agg(T=('covn2', 'mean'), C=('nbrsv', 'mean'), date=('date', 'first'))
wk = u.set_index('date').resample('W').mean()
inc_w = cases.resample('W').sum().reindex(wk.index)
axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
series = [(np.log10(inc_w + 1), 'log$_{10}$ weekly cases', 'Regional incidence (Capitale-Nationale)'),
          (wk['T'], 'Ct', 'SARS-CoV-2 N2 (target)'),
          (wk['C'], 'Ct', 'BRSV spike (control)')]
for ax, (y, lab, ttl) in zip(axes, series):
    ax.plot(y.index, y.values, color=BLUE, lw=1.6)
    ax.set_ylabel(lab); ax.set_title(ttl, loc='left', fontsize=8, color=INK)
    ax.axvspan(pd.Timestamp('2021-01-10'), SWITCH, color='#f0efec', zorder=0, lw=0)
    for ev in [pd.Timestamp('2021-08-02')]:
        ax.axvline(ev, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(pd.Timestamp('2021-01-10'), pd.Timestamp('2021-10-10'))
for ax in axes[1:]:
    ax.invert_yaxis()                  # lower Ct = more template
for ax in axes[:2]:
    ax.set_xticklabels([])
for ax in axes:
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[2, 4, 6, 8, 10]))
axes[2].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))
axes[0].text(pd.Timestamp('2021-01-14'), axes[0].get_ylim()[0] + 0.05 * np.ptp(axes[0].get_ylim()), 'P1', color=MUTED, fontsize=7.5)
axes[0].text(SWITCH + pd.Timedelta(days=4), axes[0].get_ylim()[0] + 0.05 * np.ptp(axes[0].get_ylim()), 'P2', color=MUTED, fontsize=7.5)
fig.text(axes[0].get_position().x0, 0.935, 'B  Weekly means, both plants', fontsize=9, fontweight='bold', color=INK)
for ext in ('png', 'pdf', 'eps'):
    fig.savefig(f'{OUT}/figure5.{ext}', dpi=200 if ext == 'png' else None, bbox_inches='tight')

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 30)
print(json.dumps(meta['periods'], indent=1, default=str))
print(tab.round(3).to_string())
print(ex.round(3).to_string())
