"""
SIR Model Fitting - MDM3 Fashion Trends Project
Run after data_loader.py (needs output/ml_ready_data.csv)

SIR model applied to fashion diffusion:
    S = susceptible  (potential adopters not yet into the trend)
    I = infected     (people currently into the trend, proxied by Google Trends)
    R = recovered    (people who have moved on)

    dS/dt = -beta * S * I
    dI/dt =  beta * S * I - gamma * I
    dR/dt =  gamma * I

Fitted parameters per trend: beta, gamma, I0
Derived: R0 = beta / gamma (trend virality index)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

os.makedirs("output", exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.size':        10,
})

ERA_COLORS = {
    'Pre-social': '#A0A0A0',
    'Tumblr':     '#888780',
    'Instagram':  '#D4537E',
    'TikTok':     '#1D9E75',
}

ACTIVE_THRESHOLD = 0.35   # match Alicia's validation script: active = above 35% of peak
GAP_TOLERANCE    = 1      # allow 1-month gaps inside an active run (same as myValadation.py)

# Broad bounds keep the optimiser away from numerically pathological
# solutions such as gamma -> 0 and implausibly huge R0 values.
BETA_MIN,  BETA_MAX  = 1e-3, 20.0
GAMMA_MIN, GAMMA_MAX = 1e-3, 3.0
I0_MIN,    I0_MAX    = 1e-4, 0.95


def logit(p):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


PARAM_BOUNDS = [
    (np.log(BETA_MIN),  np.log(BETA_MAX)),
    (np.log(GAMMA_MIN), np.log(GAMMA_MAX)),
    (logit(I0_MIN),     logit(I0_MAX)),
]


# clip trend to its active cycle using the same 35% threshold as Alicia's validation script

def _fill_gaps(mask, tol=GAP_TOLERANCE):
    """Fill short False gaps inside True runs (matches myValadation.py logic)."""
    mask = list(mask)
    i = 0
    while i < len(mask):
        if not mask[i]:
            start = i
            while i < len(mask) and not mask[i]:
                i += 1
            end = i - 1
            left  = start > 0 and mask[start - 1]
            right = end < len(mask) - 1 and mask[end + 1]
            if left and right and (end - start + 1) <= tol:
                for j in range(start, end + 1):
                    mask[j] = True
        else:
            i += 1
    return mask


def _longest_run(mask):
    """Return (length, start_idx, end_idx) of the longest True run."""
    best_len, best_s, best_e = 0, None, None
    cur_len, cur_s = 0, 0
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_s = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_s, best_e = cur_len, cur_s, i - 1
            cur_len = 0
    if cur_len > best_len:
        best_len, best_s, best_e = cur_len, cur_s, len(mask) - 1
    return best_len, best_s, best_e


DATE_COLS_RAW = ["Month", "Date", "date", "month", "Week", "Time"]
FOLDERS = {
    'Micro': 'data/Microtrends',
    'Macro': 'data/Macrotrends',
    'Mega':  'data/MegaTrends',
}


def _load_raw_csv(path):
    """Load a raw Google Trends CSV → normalised numpy array."""
    import glob as _glob
    for skip in (0, 1):
        try:
            raw = pd.read_csv(path, skiprows=skip)
            dc = next((c for c in DATE_COLS_RAW if c in raw.columns), None)
            if dc is None:
                continue
            vc = next((c for c in raw.columns if c != dc and
                       pd.to_numeric(raw[c], errors='coerce').notna().sum() >= len(raw) * 0.5), None)
            if vc is None:
                continue
            vals = pd.to_numeric(raw[vc], errors='coerce').fillna(0).values.astype(float)
            lo, hi = vals.min(), vals.max()
            norm = (vals - lo) / (hi - lo) if hi > lo else np.zeros(len(vals))
            dates = pd.to_datetime(raw[dc], errors='coerce')
            return norm, dates.values
        except Exception:
            continue
    return None, None


def _window_from_raw(norm):
    """Return (start_idx, end_idx) of longest run above 35% in raw normalised data."""
    active = _fill_gaps((norm >= ACTIVE_THRESHOLD).tolist())
    length, start_idx, end_idx = _longest_run(active)
    if length < 1 or start_idx is None:
        return None, None
    return start_idx, end_idx


# load smoothed data (for SIR fitting quality) + era assignment
print("Loading output/ml_ready_data.csv ...")
df = pd.read_csv("output/ml_ready_data.csv")
df['date'] = pd.to_datetime(df['date'])

peaks = df.loc[df.groupby('trend')['gt_normalised'].idxmax(), ['trend','date']].copy()
peaks['peak_year'] = peaks['date'].dt.year
def assign_era(y):
    if y < 2012:   return 'Pre-social'
    elif y < 2017: return 'Tumblr'
    elif y < 2020: return 'Instagram'
    else:          return 'TikTok'
peaks['era'] = peaks['peak_year'].apply(assign_era)
df = df.merge(peaks[['trend','era']], on='trend')

# clip each trend using RAW CSV boundaries (matches Alicia's categorisation exactly)
print("\nBuilding trend windows from raw CSVs...")
trend_windows = {}
for cat, folder in FOLDERS.items():
    import glob as _glob, os as _os
    for path in sorted(_glob.glob(_os.path.join(folder, '*.csv'))):
        stem = _os.path.splitext(_os.path.basename(path))[0]
        if 'combined' in stem.lower():
            continue
        norm_raw, _ = _load_raw_csv(path)
        if norm_raw is None:
            continue
        si, ei = _window_from_raw(norm_raw)
        if si is None:
            print(f"  Skipped: {stem:<22} (window too short)")
            continue
        # slice the smoothed series using the raw-derived boundaries
        sub = df[df['trend'] == stem].sort_values('period').reset_index(drop=True)
        if len(sub) == 0:
            continue
        si = min(si, len(sub) - 1)
        ei = min(ei, len(sub) - 1)
        clipped = sub.iloc[si:ei + 1].copy().reset_index(drop=True)
        if len(clipped) < 1:
            continue
        clipped['normalised'] = clipped['gt_normalised']
        trend_windows[stem] = clipped

trends = list(trend_windows.keys())
print(f"  {len(trends)} trends retained\n")


# SIR solver - RK4 with 4 sub-steps per period, fully inlined for speed

def solve_sir(beta, gamma, I0, n_periods, sub=4):
    dt    = 1.0 / sub
    total = (n_periods - 1) * sub + 1
    h6    = dt / 6.0
    h2    = dt / 2.0

    S, I = max(1.0 - I0, 1e-9), float(I0)
    I_out    = np.empty(n_periods)
    I_out[0] = I
    out_idx  = 1

    for step in range(1, total):
        bSI = beta * S * I
        k1s = -bSI;              k1i = bSI - gamma * I
        S2  = S + h2 * k1s;     I2  = I + h2 * k1i
        bSI = beta * S2 * I2
        k2s = -bSI;              k2i = bSI - gamma * I2
        S3  = S + h2 * k2s;     I3  = I + h2 * k2i
        bSI = beta * S3 * I3
        k3s = -bSI;              k3i = bSI - gamma * I3
        S4  = S + dt * k3s;     I4  = I + dt * k3i
        bSI = beta * S4 * I4
        k4s = -bSI;              k4i = bSI - gamma * I4

        S += h6 * (k1s + 2.0*k2s + 2.0*k3s + k4s)
        I += h6 * (k1i + 2.0*k2i + 2.0*k3i + k4i)
        if S < 0.0: S = 0.0
        if I < 0.0: I = 0.0

        if step % sub == 0 and out_idx < n_periods:
            I_out[out_idx] = I
            out_idx += 1

    return I_out


def solve_sir_full(beta, gamma, I0, n_periods, sub=4):
    """Returns (S, I, R) arrays - used for compartment plots."""
    dt    = 1.0 / sub
    total = (n_periods - 1) * sub + 1
    h6    = dt / 6.0
    h2    = dt / 2.0

    S, I, R = max(1.0 - I0, 1e-9), float(I0), 0.0
    S_out, I_out, R_out = np.empty(n_periods), np.empty(n_periods), np.empty(n_periods)
    S_out[0], I_out[0], R_out[0] = S, I, R
    out_idx = 1

    for step in range(1, total):
        bSI = beta * S * I
        k1s = -bSI;              k1i = bSI - gamma * I;  k1r = gamma * I
        S2  = S + h2*k1s;       I2  = I + h2*k1i
        bSI = beta * S2 * I2
        k2s = -bSI;              k2i = bSI - gamma*I2;   k2r = gamma * I2
        S3  = S + h2*k2s;       I3  = I + h2*k2i
        bSI = beta * S3 * I3
        k3s = -bSI;              k3i = bSI - gamma*I3;   k3r = gamma * I3
        S4  = S + dt*k3s;       I4  = I + dt*k3i
        bSI = beta * S4 * I4
        k4s = -bSI;              k4i = bSI - gamma*I4;   k4r = gamma * I4

        S += h6 * (k1s + 2.0*k2s + 2.0*k3s + k4s)
        I += h6 * (k1i + 2.0*k2i + 2.0*k3i + k4i)
        R += h6 * (k1r + 2.0*k2r + 2.0*k3r + k4r)
        if S < 0.0: S = 0.0
        if I < 0.0: I = 0.0
        if R < 0.0: R = 0.0

        if step % sub == 0 and out_idx < n_periods:
            S_out[out_idx] = S
            I_out[out_idx] = I
            R_out[out_idx] = R
            out_idx += 1

    return S_out, I_out, R_out


def objective(params, observed):
    """Sum of squared residuals. Log/logit transforms keep params positive."""
    log_beta, log_gamma, logit_I0 = params
    beta  = np.exp(log_beta)
    gamma = np.exp(log_gamma)
    I0    = inv_logit(logit_I0)
    I_pred = solve_sir(beta, gamma, I0, len(observed))
    if I_pred is None or np.any(~np.isfinite(I_pred)):
        return 1e10
    return float(np.sum((I_pred - observed) ** 2))


# multi-start fitting - 6 deterministic starts + 20 random

def fit_sir(observed, n_random_starts=20, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(observed)

    starts = [
        [np.log(0.05), np.log(0.02), logit(0.01)],
        [np.log(0.15), np.log(0.06), logit(0.01)],
        [np.log(0.40), np.log(0.15), logit(0.01)],
        [np.log(0.80), np.log(0.30), logit(0.01)],
        [np.log(0.20), np.log(0.08), logit(0.15)],
        [np.log(0.20), np.log(0.20), logit(0.01)],
    ]
    for _ in range(n_random_starts):
        starts.append([
            rng.uniform(np.log(BETA_MIN), np.log(min(3.0, BETA_MAX))),
            rng.uniform(np.log(max(0.01, GAMMA_MIN)), np.log(min(3.0, GAMMA_MAX))),
            rng.uniform(logit(max(0.001, I0_MIN)), logit(min(0.5, I0_MAX))),
        ])

    best_loss   = np.inf
    best_params = None

    for x0 in starts:
        res = minimize(objective, x0=x0, args=(observed,), method='L-BFGS-B',
                       bounds=PARAM_BOUNDS,
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_loss:
            best_loss   = res.fun
            best_params = res.x

    log_beta, log_gamma, logit_I0 = best_params
    beta  = np.exp(log_beta)
    gamma = np.exp(log_gamma)
    I0    = inv_logit(logit_I0)
    R0    = beta / gamma
    rmse  = np.sqrt(best_loss / n)
    I_fit = solve_sir(beta, gamma, I0, n)

    return {"beta": beta, "gamma": gamma, "I0": I0, "R0": R0, "rmse": rmse, "fit": I_fit}


print("\nFitting SIR model...")
print("=" * 65)

RESULTS = {}
for name in trends:
    sub      = trend_windows[name]
    observed = sub['gt_normalised'].values.astype(float)

    res = fit_sir(observed)
    res["observed"] = observed
    res["n"]        = len(observed)
    res["freq"]     = 'monthly'
    res["era"]      = sub['era'].values[0] if 'era' in sub.columns else 'Unknown'
    res["category"] = sub['category'].values[0] if 'category' in sub.columns else 'Unknown'
    RESULTS[name]   = res

    print(f"\n  {name}  [{res['era']}]")
    print(f"    beta  : {res['beta']:.4f}")
    print(f"    gamma : {res['gamma']:.4f}")
    print(f"    I0    : {res['I0']:.4f}")
    print(f"    R0    : {res['R0']:.3f}")
    print(f"    RMSE  : {res['rmse']:.4f}")

print("\n" + "=" * 65)


# bootstrap confidence intervals - 300 resamples, warm-start from fitted params

print("\nBootstrap confidence intervals (300 resamples)...")
N_BOOT = 300
rng    = np.random.default_rng(0)


def moving_block_resample(series, rng, block_len=None):
    """Resample a time series in short contiguous blocks to preserve ordering."""
    n = len(series)
    if n == 0:
        return np.array([], dtype=float)
    if block_len is None:
        block_len = max(2, min(6, int(round(np.sqrt(n)))))

    pieces = []
    total = 0
    while total < n:
        start = int(rng.integers(0, n))
        idx = (np.arange(block_len) + start) % n
        block = series[idx]
        pieces.append(block)
        total += len(block)
    return np.concatenate(pieces)[:n]


def percentile_or_nan(values, q):
    return float(np.percentile(values, q)) if values else float('nan')

for name, res in RESULTS.items():
    observed = res["observed"]
    n        = res["n"]
    fitted   = res["fit"]
    boot_beta, boot_gamma, boot_R0 = [], [], []
    x0_warm = np.array([
        np.log(res["beta"]),
        np.log(res["gamma"]),
        logit(np.clip(res["I0"], I0_MIN, I0_MAX)),
    ])
    centered_resid = observed - fitted
    centered_resid = centered_resid - np.mean(centered_resid)

    for _ in range(N_BOOT):
        pseudo_obs = np.clip(fitted + moving_block_resample(centered_resid, rng), 0.0, 1.0)
        try:
            r = minimize(objective, x0=x0_warm, args=(pseudo_obs,), method='L-BFGS-B',
                         bounds=PARAM_BOUNDS,
                         options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if not r.success or np.any(~np.isfinite(r.x)):
                continue
            b_beta  = np.exp(r.x[0])
            b_gamma = np.exp(r.x[1])
            if not (np.isfinite(b_beta) and np.isfinite(b_gamma) and b_gamma > 0):
                continue
            boot_beta.append(float(b_beta))
            boot_gamma.append(float(b_gamma))
            boot_R0.append(float(b_beta / b_gamma))
        except Exception:
            pass

    res["ci_beta"]  = (percentile_or_nan(boot_beta,  2.5), percentile_or_nan(boot_beta,  97.5))
    res["ci_gamma"] = (percentile_or_nan(boot_gamma, 2.5), percentile_or_nan(boot_gamma, 97.5))
    res["ci_R0"]    = (percentile_or_nan(boot_R0,    2.5), percentile_or_nan(boot_R0,    97.5))

    print(f"  {name}")
    print(f"    beta = {res['beta']:.4f}  95% CI [{res['ci_beta'][0]:.4f}, {res['ci_beta'][1]:.4f}]")
    print(f"    gamma = {res['gamma']:.4f}  95% CI [{res['ci_gamma'][0]:.4f}, {res['ci_gamma'][1]:.4f}]")
    print(f"    R0 = {res['R0']:.3f}   95% CI [{res['ci_R0'][0]:.3f}, {res['ci_R0'][1]:.3f}]")


# residual analysis - Ljung-Box test for autocorrelation in residuals

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as acf_fn

print("\nResidual analysis (Ljung-Box test)...")
RESIDUALS = {}
for name, res in RESULTS.items():
    resid = res["observed"] - res["fit"]
    RESIDUALS[name] = resid
    max_lag = max(1, min(10, len(resid) // 2 - 1))
    if len(resid) < 4 or max_lag < 1:
        print(f"  {name}: too short for Ljung-Box (n={len(resid)}) -- skipped")
        continue
    lb = acorr_ljungbox(resid, lags=[max_lag], return_df=True)
    p  = lb['lb_pvalue'].values[0]
    verdict = "white noise" if p > 0.05 else "autocorrelated (model misfit)"
    print(f"  {name}: Ljung-Box p = {p:.4f}  --> {verdict}")

print("\nFigure 9: Residuals...")
names_ord = list(RESIDUALS.keys())
n_trends  = len(names_ord)
ncols     = 4
nrows_res = int(np.ceil(n_trends / ncols))
fig, axes = plt.subplots(nrows_res * 2, ncols, figsize=(ncols * 4, nrows_res * 5))
axes_flat = axes.flatten()

for i, name in enumerate(names_ord):
    ax    = axes_flat[i]
    resid = RESIDUALS[name]
    color = ERA_COLORS.get(RESULTS[name]["era"], '#888888')
    t     = np.arange(len(resid))
    ax.bar(t, resid, color=color, alpha=0.65, width=0.8)
    ax.axhline(0, color='black', lw=0.9)
    ax.set_title(f'{name}', fontweight='bold', fontsize=9)
    ax.set_xlabel("Period")
    ax.set_ylabel("Residual")

for i, name in enumerate(names_ord):
    ax     = axes_flat[nrows_res * ncols + i]
    resid  = RESIDUALS[name]
    color  = ERA_COLORS.get(RESULTS[name]["era"], '#888888')
    n_lags = min(15, len(resid) // 2 - 1)
    acf_r  = acf_fn(resid, nlags=n_lags, fft=True)
    conf   = 1.96 / np.sqrt(len(resid))
    lags   = np.arange(len(acf_r))
    ax.bar(lags, acf_r, color=color, alpha=0.7, width=0.6)
    ax.axhline( conf, color='black', lw=1.0, linestyle='--', label='95% CI')
    ax.axhline(-conf, color='black', lw=1.0, linestyle='--')
    ax.axhline(0,     color='black', lw=0.5)
    ax.set_title(f'{name} ACF', fontsize=8, fontweight='bold')
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend(fontsize=7, loc='upper right')

for j in range(n_trends, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("SIR model residual analysis\nTop: residuals    Bottom: ACF (flat = white noise)",
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("output/fig9_residuals.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: output/fig9_residuals.png")


# figure 6 - SIR fits

print("\nFigure 6: SIR fits...")
n_fits   = len(RESULTS)
ncols6   = 4
nrows6   = int(np.ceil(n_fits / ncols6))
fig, axes = plt.subplots(nrows6, ncols6, figsize=(ncols6 * 4, nrows6 * 4))
axes_flat6 = axes.flatten() if n_fits > 1 else [axes]

for i, (name, res) in enumerate(RESULTS.items()):
    ax = axes_flat6[i]
    color = ERA_COLORS.get(RESULTS[name]["era"], '#888888')
    t     = np.arange(res["n"], dtype=float)
    unit  = "Months" if res["freq"] == "monthly" else "Weeks"

    ax.plot(t, res["observed"], color=color, lw=1.5, alpha=0.65, label="Observed (normalised)")
    if res["fit"] is not None:
        ax.plot(t, res["fit"], color='black', lw=2.5, linestyle='--',
                label=f"SIR fit  (R0 = {res['R0']:.2f},  RMSE = {res['rmse']:.3f})")

    ax.text(0.03, 0.03,
            f"beta = {res['beta']:.3f}\ngamma = {res['gamma']:.3f}\nR0 = {res['R0']:.2f}\nI0 = {res['I0']:.3f}",
            transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_title(f'{name}  [{res["era"]}]', fontweight='bold', fontsize=10)
    ax.set_xlabel(f"{unit} from series start")
    ax.set_ylabel("Normalised interest [0,1]")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8, loc='upper right')

for j in range(n_fits, len(axes_flat6)):
    axes_flat6[j].set_visible(False)

fig.suptitle("SIR model fits to Google Trends fashion diffusion data",
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("output/fig6_sir_fits.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: output/fig6_sir_fits.png")


# figure 7 - R0 comparison

print("\nFigure 7: R0 comparison...")
fig, ax = plt.subplots(figsize=(9, 5))

names  = list(RESULTS.keys())
R0s    = [RESULTS[n]["R0"]  for n in names]
colors = [ERA_COLORS.get(RESULTS[n]["era"], '#888888') for n in names]

bars = ax.barh(names, R0s, color=colors, alpha=0.85, height=0.45)
ax.axvline(1.0, color='black', lw=1.2, linestyle='--', alpha=0.6, label='R0 = 1 (epidemic threshold)')

for bar, val in zip(bars, R0s):
    ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2,
            f'R0 = {val:.2f}', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel("Basic reproduction number  R0 = beta / gamma", fontsize=11)
ax.set_title("Fashion trend virality across platform eras", fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, max(R0s) * 1.25)
plt.tight_layout()
plt.savefig("output/fig7_R0_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: output/fig7_R0_comparison.png")


# figure 8 - S, I, R compartments over time

print("\nFigure 8: SIR compartments...")
ncols8   = 4
nrows8   = int(np.ceil(n_fits / ncols8))
fig, axes = plt.subplots(nrows8, ncols8, figsize=(ncols8 * 4, nrows8 * 4))
axes_flat8 = axes.flatten() if n_fits > 1 else [axes]

for i, (name, res) in enumerate(RESULTS.items()):
    ax = axes_flat8[i]
    color = ERA_COLORS.get(RESULTS[name]["era"], '#888888')
    n     = res["n"]
    unit  = "Months" if res["freq"] == "monthly" else "Weeks"
    t     = np.arange(n, dtype=float)

    S_t, I_t, R_t = solve_sir_full(res["beta"], res["gamma"], res["I0"], n)
    ax.plot(t, S_t, color='steelblue', lw=2,   label='S (susceptible)')
    ax.plot(t, I_t, color=color,       lw=2.5, label='I (infected / interested)')
    ax.plot(t, R_t, color='#888780',   lw=2,   label='R (recovered / moved on)')
    ax.plot(t, res["observed"], color=color, lw=1.2, alpha=0.4, linestyle=':', label='Observed')

    ax.set_title(f'{name}  R0={res["R0"]:.2f}', fontweight='bold', fontsize=10)
    ax.set_xlabel(f"{unit} from series start")
    ax.set_ylabel("Population fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='center right')

for j in range(n_fits, len(axes_flat8)):
    axes_flat8[j].set_visible(False)

fig.suptitle("SIR compartment trajectories", fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("output/fig8_sir_compartments.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: output/fig8_sir_compartments.png")


# export results to CSV

rows = [
    {
        "trend":       name,
        "era":         res["era"],
        "freq":        res["freq"],
        "beta":        round(res["beta"],         4),
        "beta_ci_lo":  round(res["ci_beta"][0],   4),
        "beta_ci_hi":  round(res["ci_beta"][1],   4),
        "gamma":       round(res["gamma"],        4),
        "gamma_ci_lo": round(res["ci_gamma"][0],  4),
        "gamma_ci_hi": round(res["ci_gamma"][1],  4),
        "R0":          round(res["R0"],           4),
        "R0_ci_lo":    round(res["ci_R0"][0],     4),
        "R0_ci_hi":    round(res["ci_R0"][1],     4),
        "I0":          round(res["I0"],           4),
        "rmse":        round(res["rmse"],         4),
    }
    for name, res in RESULTS.items()
]
pd.DataFrame(rows).to_csv("output/sir_parameters.csv", index=False)
print("Exported: output/sir_parameters.csv")


# ── EARLY PREDICTION ──────────────────────────────────────────────────────────
# Fit SIR to first OBS_MONTHS only, then extrapolate forward and compare
# to what actually happened (the model never sees that future data).

OBS_MONTHS = 6

print("\n\nEarly prediction: fitting SIR to first 6 months only...")
print("=" * 65)

EARLY_RESULTS = {}
for name, res in RESULTS.items():
    observed = res["observed"]
    n        = res["n"]

    if n <= OBS_MONTHS + 2:
        print(f"  Skipped {name:<22} (only {n} months total, need >{OBS_MONTHS + 2})")
        continue

    early_obs = observed[:OBS_MONTHS]
    early_res = fit_sir(early_obs)

    predicted_full  = solve_sir(early_res["beta"], early_res["gamma"], early_res["I0"], n)
    held_pred       = predicted_full[OBS_MONTHS:]
    held_actual     = observed[OBS_MONTHS:]
    rmse_holdout    = float(np.sqrt(np.mean((held_pred - held_actual) ** 2)))
    rmse_early      = float(np.sqrt(np.mean((predicted_full[:OBS_MONTHS] - early_obs) ** 2)))

    ss_res = float(np.sum((held_pred - held_actual) ** 2))
    ss_tot = float(np.sum((held_actual - held_actual.mean()) ** 2))
    r2_holdout = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    EARLY_RESULTS[name] = {
        "observed":       observed,
        "predicted_full": predicted_full,
        "rmse_holdout":   rmse_holdout,
        "rmse_early":     rmse_early,
        "r2_holdout":     r2_holdout,
        "era":            res["era"],
        "category":       res.get("category", "Unknown"),
        "n":              n,
    }
    cat = res.get("category", "Unknown")
    print(f"  {name:<22} [{cat:<5}]  fit RMSE: {rmse_early:.4f}   holdout RMSE: {rmse_holdout:.4f}   R²: {r2_holdout:+.3f}")

# overall
mean_rmse = float(np.mean([v["rmse_holdout"] for v in EARLY_RESULTS.values()]))
mean_r2   = float(np.nanmean([v["r2_holdout"] for v in EARLY_RESULTS.values()]))
print(f"\n  Overall  mean holdout RMSE: {mean_rmse:.4f}   mean R²: {mean_r2:+.3f}")

# split by category
print(f"\n  Holdout RMSE by trend category:")
for cat in ["Micro", "Macro", "Mega"]:
    vals_rmse = [v["rmse_holdout"] for v in EARLY_RESULTS.values() if v["category"] == cat]
    vals_r2   = [v["r2_holdout"]   for v in EARLY_RESULTS.values() if v["category"] == cat]
    if vals_rmse:
        print(f"    {cat:<6}  RMSE mean={np.mean(vals_rmse):.4f}  "
              f"R² mean={np.nanmean(vals_r2):+.3f}   n={len(vals_rmse)}")

print("=" * 65)


# figure 10 - early prediction grid

print("\nFigure 10: Early prediction grid...")
names_ep  = list(EARLY_RESULTS.keys())
n_ep      = len(names_ep)
ncols_ep  = 4
nrows_ep  = int(np.ceil(n_ep / ncols_ep))

fig, axes = plt.subplots(nrows_ep, ncols_ep,
                         figsize=(ncols_ep * 4, nrows_ep * 3.5))
axes_flat_ep = axes.flatten() if n_ep > 1 else [axes]

for i, name in enumerate(names_ep):
    ax  = axes_flat_ep[i]
    er  = EARLY_RESULTS[name]
    col = ERA_COLORS.get(er["era"], '#888888')
    n   = er["n"]
    t   = np.arange(n)

    # full actual curve
    ax.plot(t, er["observed"], color=col, lw=1.8, alpha=0.75, label="Actual")

    # shade the training window
    ax.axvspan(-0.5, OBS_MONTHS - 0.5, alpha=0.10, color='steelblue')

    # early observed points (highlighted)
    ax.plot(t[:OBS_MONTHS], er["observed"][:OBS_MONTHS],
            color='steelblue', lw=2.2, alpha=0.95, label=f"Fit window ({OBS_MONTHS}mo)")

    # SIR extrapolation (full curve)
    ax.plot(t, er["predicted_full"], color='black', lw=1.8,
            linestyle='--', label="SIR prediction")

    # vertical cut-off line
    ax.axvline(OBS_MONTHS - 0.5, color='steelblue', lw=1.0,
               linestyle=':', alpha=0.7)

    ax.set_title(f"{name}  [{er['category']}]\nRMSE={er['rmse_holdout']:.3f}  R²={er['r2_holdout']:+.2f}",
                 fontsize=8, fontweight='bold')
    ax.set_xlabel("Months from trend start", fontsize=7)
    ax.set_ylabel("Normalised interest", fontsize=7)
    ax.set_ylim(-0.05, 1.18)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=6, loc='upper right')

for j in range(n_ep, len(axes_flat_ep)):
    axes_flat_ep[j].set_visible(False)

fig.suptitle(
    "SIR early prediction — fit to first 6 months only, then extrapolate forward\n"
    "Blue shading = training window   Dashed = SIR forecast   Solid = actual",
    fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("output/fig10_sir_early_prediction.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: output/fig10_sir_early_prediction.png")


print("\nDone. Figures: fig6, fig7, fig8, fig9, fig10")
