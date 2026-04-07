"""
SIR Model Fitting - MDM3 Fashion Trends Project
Run after data_processing.py (needs output/processed_trends.csv)

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

TREND_COLORS = {
    "Soft grunge":     "#888780",
    "Millennial pink": "#D4537E",
    "Cottagecore":     "#1D9E75",
    "Dark academia":   "#534AB7",
}


# load data

print("Loading output/processed_trends.csv ...")
df = pd.read_csv("output/processed_trends.csv")
df['date'] = pd.to_datetime(df['date'])
trends = list(df['trend'].unique())
print(f"  Trends: {trends}")


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
    I0    = 1.0 / (1.0 + np.exp(-logit_I0))
    I_pred = solve_sir(beta, gamma, I0, len(observed))
    if I_pred is None or np.any(~np.isfinite(I_pred)):
        return 1e10
    return float(np.sum((I_pred - observed) ** 2))


# multi-start fitting - 6 deterministic starts + 20 random

def fit_sir(observed, n_random_starts=20, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(observed)
    logit = lambda p: np.log(p / (1 - p))

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
            rng.uniform(np.log(0.01), np.log(3.0)),
            rng.uniform(np.log(0.01), np.log(3.0)),
            rng.uniform(logit(0.001), logit(0.5)),
        ])

    best_loss   = np.inf
    best_params = None

    for x0 in starts:
        res = minimize(objective, x0=x0, args=(observed,), method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_loss:
            best_loss   = res.fun
            best_params = res.x

    log_beta, log_gamma, logit_I0 = best_params
    beta  = np.exp(log_beta)
    gamma = np.exp(log_gamma)
    I0    = 1.0 / (1.0 + np.exp(-logit_I0))
    R0    = beta / gamma
    rmse  = np.sqrt(best_loss / n)
    I_fit = solve_sir(beta, gamma, I0, n)

    return {"beta": beta, "gamma": gamma, "I0": I0, "R0": R0, "rmse": rmse, "fit": I_fit}


print("\nFitting SIR model...")
print("=" * 65)

RESULTS = {}
for name in trends:
    sub      = df[df['trend'] == name].sort_values('period')
    observed = sub['normalised'].values.astype(float)
    freq     = sub['freq'].values[0] if 'freq' in sub.columns else 'monthly'

    res = fit_sir(observed)
    res["observed"] = observed
    res["n"]        = len(observed)
    res["freq"]     = freq
    res["era"]      = sub['era'].values[0]
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

for name, res in RESULTS.items():
    observed = res["observed"]
    n        = res["n"]
    boot_beta, boot_gamma, boot_R0 = [], [], []

    logit = lambda p: np.log(p / (1.0 - p))
    x0_warm = np.array([
        np.log(res["beta"]),
        np.log(res["gamma"]),
        logit(np.clip(res["I0"], 1e-6, 1 - 1e-6)),
    ])

    for _ in range(N_BOOT):
        idx      = rng.integers(0, n, size=n)
        resample = observed[idx]
        try:
            r = minimize(objective, x0=x0_warm, args=(resample,), method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            b_beta  = np.exp(r.x[0])
            b_gamma = np.exp(r.x[1])
            boot_beta.append(b_beta)
            boot_gamma.append(b_gamma)
            boot_R0.append(b_beta / b_gamma)
        except Exception:
            pass

    res["ci_beta"]  = (np.percentile(boot_beta,  2.5), np.percentile(boot_beta,  97.5))
    res["ci_gamma"] = (np.percentile(boot_gamma, 2.5), np.percentile(boot_gamma, 97.5))
    res["ci_R0"]    = (np.percentile(boot_R0,    2.5), np.percentile(boot_R0,    97.5))

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
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    p  = lb['lb_pvalue'].values[0]
    verdict = "white noise" if p > 0.05 else "autocorrelated (model misfit)"
    print(f"  {name}: Ljung-Box p = {p:.4f}  --> {verdict}")

print("\nFigure 9: Residuals...")
names_ord = list(RESIDUALS.keys())
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, name in zip(axes[0], names_ord):
    resid = RESIDUALS[name]
    color = TREND_COLORS[name]
    t     = np.arange(len(resid))
    ax.bar(t, resid, color=color, alpha=0.65, width=0.8)
    ax.axhline(0, color='black', lw=0.9)
    ax.set_title(f'{name}', fontweight='bold', fontsize=10)
    ax.set_xlabel("Period")
    ax.set_ylabel("Residual (observed - fit)")

for ax, name in zip(axes[1], names_ord):
    resid  = RESIDUALS[name]
    color  = TREND_COLORS[name]
    n_lags = min(15, len(resid) // 2 - 1)
    acf_r  = acf_fn(resid, nlags=n_lags, fft=True)
    conf   = 1.96 / np.sqrt(len(resid))
    lags   = np.arange(len(acf_r))
    ax.bar(lags, acf_r, color=color, alpha=0.7, width=0.6)
    ax.axhline( conf, color='black', lw=1.0, linestyle='--', label='95% CI')
    ax.axhline(-conf, color='black', lw=1.0, linestyle='--')
    ax.axhline(0,     color='black', lw=0.5)
    ax.set_title(f'{name} - Residual ACF', fontsize=9, fontweight='bold')
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend(fontsize=7, loc='upper right')

fig.suptitle("SIR model residual analysis\nTop: residuals    Bottom: ACF (flat = white noise)",
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("output/fig9_residuals.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("  Saved: output/fig9_residuals.png")


# figure 6 - SIR fits

print("\nFigure 6: SIR fits...")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, (name, res) in zip(axes, RESULTS.items()):
    color = TREND_COLORS[name]
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

fig.suptitle("SIR model fits to Google Trends fashion diffusion data",
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("output/fig6_sir_fits.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("  Saved: output/fig6_sir_fits.png")


# figure 7 - R0 comparison

print("\nFigure 7: R0 comparison...")
fig, ax = plt.subplots(figsize=(9, 5))

names  = list(RESULTS.keys())
R0s    = [RESULTS[n]["R0"]   for n in names]
colors = [TREND_COLORS[n]    for n in names]

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
plt.show()
plt.close()
print("  Saved: output/fig7_R0_comparison.png")


# figure 8 - S, I, R compartments over time

print("\nFigure 8: SIR compartments...")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, (name, res) in zip(axes, RESULTS.items()):
    color = TREND_COLORS[name]
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

fig.suptitle("SIR compartment trajectories", fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("output/fig8_sir_compartments.png", dpi=150, bbox_inches='tight')
plt.show()
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
print("\nDone. Figures: fig6, fig7, fig8, fig9")
