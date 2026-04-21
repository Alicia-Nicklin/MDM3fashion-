"""
Agent-Based Model — SIR-driven wave version
MDM3 Fashion Trends Project

Key change from previous version:
  The ABM wave (external social pressure) is now derived directly from the
  solved SIR I(t) curve using the fitted beta/gamma/I0 for that trend,
  rather than a hand-crafted Gaussian or logistic shape.

  Logic:
    1. Load fitted SIR parameters from sir_parameters.csv  (Tiggy's output)
    2. Solve I(t) using those parameters                   (RK4, same solver as SIR script)
    3. Rescale I(t) → wave that modulates agent join pressure
    4. Run ABM — agents still have individual randomness, but timing now
       comes from the fitted SIR trajectory, not a manually placed peak
    5. Compare simulated vs real: print RMSE + correlation per trend

Timescale note:
  SIR was fitted on monthly data.  The ABM therefore runs on monthly steps
  (n_steps = number of months in the clipped active window).  Axis labels
  say "Months" throughout for consistency.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# 1) SETUP
# =========================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SIR_PATH   = os.path.join(OUTPUT_DIR, "sir_parameters.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTIVE_THRESHOLD = 0.35   # matches Tiggy's SIR script and Alicia's validation
GAP_TOLERANCE    = 1      # allow 1-month gaps inside an active run
BUFFER           = 5      # months of context either side of active window
N_PEOPLE         = 1000
NOISE            = 0.005

np.random.seed(42)

ERA_COLORS = {
    "Pre-social": "#A0A0A0",
    "Tumblr":     "#888780",
    "Instagram":  "#D4537E",
    "TikTok":     "#1D9E75",
}

SEARCH_DIRS = {
    "Microtrends": os.path.join(DATA_DIR, "Microtrends"),
    "Macrotrends": os.path.join(DATA_DIR, "Macrotrends"),
    "MegaTrends":  os.path.join(DATA_DIR, "MegaTrends"),
}

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        10,
})

# =========================================================
# 2) HELPERS
# =========================================================

def safe_normalise(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.zeros_like(x)


def minmax_scale(val, vmin, vmax, out_min, out_max):
    """Map a scalar from [vmin, vmax] → [out_min, out_max]."""
    if vmax <= vmin:
        return 0.5 * (out_min + out_max)
    z = np.clip((val - vmin) / (vmax - vmin), 0, 1)
    return out_min + z * (out_max - out_min)


def clean_era_label(era):
    if pd.isna(era):
        return "Unknown"
    return str(era).split("(")[0].strip()


def normalise_name(s):
    return str(s).lower().replace("_", "").replace("-", "").replace(" ", "").strip()


def _fill_gaps(mask, tol=GAP_TOLERANCE):
    """Fill short False gaps inside True runs (mirrors SIR script logic)."""
    mask = list(mask)
    i = 0
    while i < len(mask):
        if not mask[i]:
            start = i
            while i < len(mask) and not mask[i]:
                i += 1
            end = i - 1
            left  = start > 0            and mask[start - 1]
            right = end < len(mask) - 1  and mask[end + 1]
            if left and right and (end - start + 1) <= tol:
                for j in range(start, end + 1):
                    mask[j] = True
        else:
            i += 1
    return np.array(mask, dtype=bool)


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


MANUAL_NAME_MAP = {
    "Soft grunge": "Tumblr Grunge",
}

def find_trend_file(trend_name):
    """Search SEARCH_DIRS for a CSV matching trend_name (fuzzy stem match)."""
    lookup = MANUAL_NAME_MAP.get(trend_name, trend_name)
    target = normalise_name(lookup)
    for folder_name, folder_path in SEARCH_DIRS.items():
        if not os.path.exists(folder_path):
            continue
        for path in glob.glob(os.path.join(folder_path, "*.csv")):
            stem = os.path.splitext(os.path.basename(path))[0]
            if normalise_name(stem) == target:
                return folder_name, path
    return None, None

# =========================================================
# 3) SIR SOLVER  (RK4, identical to Tiggy's implementation)
# =========================================================

def solve_sir_I(beta, gamma, I0, n_periods, sub=4):
    """
    Return the I(t) trajectory from the SIR model.
    Uses RK4 with `sub` sub-steps per period — same as Tiggy's solver.
    n_periods = number of monthly steps to simulate.
    """
    dt   = 1.0 / sub
    h6   = dt / 6.0
    h2   = dt / 2.0
    total = (n_periods - 1) * sub + 1

    S, I = max(1.0 - I0, 1e-9), float(I0)
    I_out = np.empty(n_periods)
    I_out[0] = I
    out_idx = 1

    for step in range(1, total):
        bSI = beta * S * I
        k1s = -bSI;           k1i = bSI - gamma * I
        S2 = S + h2*k1s;      I2  = I + h2*k1i
        bSI = beta * S2 * I2
        k2s = -bSI;           k2i = bSI - gamma * I2
        S3 = S + h2*k2s;      I3  = I + h2*k2i
        bSI = beta * S3 * I3
        k3s = -bSI;           k3i = bSI - gamma * I3
        S4 = S + dt*k3s;      I4  = I + dt*k3i
        bSI = beta * S4 * I4
        k4s = -bSI;           k4i = bSI - gamma * I4

        S += h6 * (k1s + 2*k2s + 2*k3s + k4s)
        I += h6 * (k1i + 2*k2i + 2*k3i + k4i)
        S = max(S, 0.0)
        I = max(I, 0.0)

        if step % sub == 0 and out_idx < n_periods:
            I_out[out_idx] = I
            out_idx += 1

    return I_out

# =========================================================
# 4) SIR-DERIVED WAVE
# =========================================================

def make_sir_wave(beta, gamma, I0, n_steps, peak_height):
    """
    Solve the SIR model → normalise I(t) → build the wave that modulates
    agent join pressure.

        wave[t] = 1 + peak_height * normalised_I(t)

    This replaces ALL hand-crafted Gaussian / logistic shapes.
    The timing now comes from the actual fitted SIR trajectory.
    """
    I_sir = solve_sir_I(beta, gamma, I0, n_steps)
    I_norm = safe_normalise(I_sir)        # rescale to [0, 1]
    wave = 1.0 + peak_height * I_norm     # stay above 1 so baseline adoption continues
    # light smoothing to reduce step-function jitter from short series
    if n_steps >= 5:
        from scipy.ndimage import uniform_filter1d
        wave = uniform_filter1d(wave, size=3)
    wave = np.clip(wave, 0.15, None)
    return wave, I_sir

# =========================================================
# 5) ABM FUNCTION  (wave is now an input, not computed inside)
# =========================================================

def simulate_trend(
    n_people,
    n_steps,
    wave,           # SIR-derived pressure profile, shape (n_steps,)
    social_weight,
    start_chance,
    fade_base,
    late_drop,
    start_share,
    peak_step,      # index of wave peak — derived from SIR, not from real data
    noise=NOISE,
    start_delay=0,
):
    """
    Run one ABM simulation.

    Agents are either 0 (not adopted) or 1 (adopted).
    At each monthly step:
      - Susceptible agents join with probability driven by `wave[t]` and
        current adoption share (social contagion).
      - Adopted agents leave with a base rate that accelerates after the peak.

    `wave` replaces the old hand-crafted Gaussian / logistic shapes.
    `peak_step` is the index at which wave peaks (from SIR I(t)), not argmax(real).
    """
    state = np.zeros(n_people, dtype=int)
    seeded = max(1, int(start_share * n_people))
    state[:seeded] = 1
    np.random.shuffle(state)

    series = np.zeros(n_steps)
    uptake_weight = np.random.uniform(0.9, 1.1, n_people)
    fade_weight   = np.random.uniform(0.9, 1.1, n_people)

    for i in range(n_steps):
        if i < start_delay:
            series[i] = 0.0
            continue

        current_share = state.mean()

        join_rate  = (start_chance + social_weight * (current_share ** 1.1)) * wave[i]
        join_rate += np.random.normal(0, noise)

        # adoption pressure drops sharply after the SIR peak
        if i > peak_step:
            join_rate *= 0.20

        if i > peak_step:
            extra_fade = late_drop * ((i - peak_step) / max(1, n_steps - peak_step))
        else:
            extra_fade = 0.0

        leave_rate  = fade_base + extra_fade
        leave_rate += np.random.normal(0, noise / 2)

        join_prob  = np.clip(join_rate  * uptake_weight, 0, 0.95)
        leave_prob = np.clip(leave_rate * fade_weight,   0, 0.95)

        draw_join  = np.random.random(n_people)
        draw_leave = np.random.random(n_people)

        state[(state == 0) & (draw_join  < join_prob)]  = 1
        state[(state == 1) & (draw_leave < leave_prob)] = 0

        series[i] = state.mean()

    if series.max() > 0:
        series /= series.max()

    return series

# =========================================================
# 6) LOAD SIR PARAMETERS AND BUILD PER-TREND ABM PARAMS
# =========================================================

if not os.path.exists(SIR_PATH):
    raise FileNotFoundError(
        f"Could not find {SIR_PATH}\n"
        "Run Tiggy's SIR fitting script first to generate sir_parameters.csv."
    )

sir_df = pd.read_csv(SIR_PATH)
required = {"trend", "beta", "gamma", "R0", "I0", "era"}
missing  = required - set(sir_df.columns)
if missing:
    raise ValueError(f"sir_parameters.csv is missing columns: {missing}")

beta_min,  beta_max  = sir_df["beta"].min(),  sir_df["beta"].max()
gamma_min, gamma_max = sir_df["gamma"].min(), sir_df["gamma"].max()
r0_min,    r0_max    = sir_df["R0"].min(),    sir_df["R0"].max()

print(f"\nLoaded {len(sir_df)} trends from {SIR_PATH}")
print(f"  beta  range: [{beta_min:.4f}, {beta_max:.4f}]")
print(f"  gamma range: [{gamma_min:.4f}, {gamma_max:.4f}]")
print(f"  R0    range: [{r0_min:.3f},  {r0_max:.3f}]")

# =========================================================
# 7) RUN ALL TRENDS
# =========================================================

summary_rows = []

print("\n" + "=" * 70)
print(f"{'Trend':<24} {'Era':<12} {'RMSE':>7} {'Corr':>7} {'R0':>6}")
print("=" * 70)

for _, row in sir_df.iterrows():
    trend_name = str(row["trend"]).strip()
    beta       = float(row["beta"])
    gamma      = float(row["gamma"])
    r0         = float(row["R0"])
    i0         = float(row["I0"])
    era        = clean_era_label(row["era"])

    folder_name, csv_path = find_trend_file(trend_name)
    if csv_path is None:
        print(f"  Skipped {trend_name}: CSV not found")
        continue

    # --- load real data ---
    df_raw = pd.read_csv(csv_path)
    real   = pd.to_numeric(df_raw.iloc[:, 1], errors="coerce").fillna(0).values.astype(float)
    real   = safe_normalise(real)
    n_raw  = len(real)

    # --- active window (mirrors SIR script) ---
    active_mask = _fill_gaps(real >= ACTIVE_THRESHOLD, tol=GAP_TOLERANCE)
    run_len, run_start, run_end = _longest_run(active_mask)

    if run_start is None:
        print(f"  Skipped {trend_name}: no active window")
        continue

    start_idx = max(0, run_start - BUFFER)
    end_idx   = min(n_raw - 1, run_end + BUFFER)
    n_steps   = end_idx - start_idx + 1   # monthly steps in clipped window

    # delay before first active month
    start_delay = max(0, run_start - start_idx - 2)

    # --- SIR-derived wave (KEY CHANGE) ---
    # Scale peak_height from R0 (higher virality → taller wave)
    peak_height = minmax_scale(r0, r0_min, r0_max, 1.8, 3.6)
    wave, I_sir = make_sir_wave(beta, gamma, i0, n_steps, peak_height)

    # peak_step from SIR I(t), not from real data (KEY CHANGE)
    peak_step = int(np.argmax(I_sir))

    # --- scale other ABM params from SIR ---
    social_weight = minmax_scale(r0,    r0_min,    r0_max,    0.18, 0.58)
    start_chance  = minmax_scale(beta,  beta_min,  beta_max,  0.002, 0.018)
    fade_base     = minmax_scale(gamma, gamma_min, gamma_max, 0.015, 0.12)
    late_drop     = minmax_scale(gamma, gamma_min, gamma_max, 0.05,  0.30)
    start_share   = float(np.clip(i0, 0.005, 0.15))

    # --- run ABM ---
    fake = simulate_trend(
        n_people     = N_PEOPLE,
        n_steps      = n_steps,
        wave         = wave,
        social_weight= social_weight,
        start_chance = start_chance,
        fade_base    = fade_base,
        late_drop    = late_drop,
        start_share  = start_share,
        peak_step    = peak_step,
        start_delay  = start_delay,
    )

    # --- clip real data to same window ---
    real_clip = real[start_idx:end_idx + 1]
    fake_clip = fake                          # already n_steps long
    x_clip    = np.arange(start_idx, end_idx + 1)

    # --- metrics ---
    rmse_abm = float(np.sqrt(np.mean((real_clip - fake_clip) ** 2)))
    if len(real_clip) > 1 and real_clip.std() > 0 and fake_clip.std() > 0:
        corr_abm = float(np.corrcoef(real_clip, fake_clip)[0, 1])
    else:
        corr_abm = float("nan")

    print(f"  {trend_name:<24} {era:<12} {rmse_abm:>7.4f} {corr_abm:>7.3f} {r0:>6.2f}")

    summary_rows.append({
        "trend":       trend_name,
        "era":         era,
        "folder":      folder_name,
        "R0":          round(r0,       3),
        "beta":        round(beta,     4),
        "gamma":       round(gamma,    4),
        "I0":          round(i0,       4),
        "abm_rmse":    round(rmse_abm, 4),
        "abm_corr":    round(corr_abm, 4) if np.isfinite(corr_abm) else None,
        "n_months":    n_steps,
        "peak_step_sir": peak_step,
    })

    # --- plot ---
    color = ERA_COLORS.get(era, "#888888")
    clean = trend_name.replace(" ", "_").lower()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # left: ABM vs real
    ax = axes[0]
    ax.plot(x_clip, real_clip, label="Real trend",           lw=2,   color=color, alpha=0.8)
    ax.plot(x_clip, fake_clip, label="SIR-driven ABM",       lw=2,   color="black", linestyle="--")
    ax.axhline(ACTIVE_THRESHOLD, color="grey", linestyle=":", lw=1,  label="Threshold = 0.35")
    ax.set_title(f"{trend_name}  [{era}]\nRMSE = {rmse_abm:.4f}   corr = {corr_abm:.3f}",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Months from series start")
    ax.set_ylabel("Normalised interest [0, 1]")
    ax.set_ylim(-0.05, 1.18)
    ax.legend(fontsize=8)

    # right: SIR wave that drove the ABM
    ax2 = axes[1]
    t_sir = np.arange(n_steps)
    I_norm_plot = safe_normalise(I_sir)
    ax2.plot(t_sir, I_norm_plot, color=color, lw=2,   label="SIR I(t) [normalised]")
    ax2.plot(t_sir, (wave - 1) / peak_height, color="black", lw=1.5,
             linestyle=":", label="Wave (rescaled)")
    ax2.axvline(peak_step, color="grey", linestyle="--", lw=1, label=f"SIR peak (t={peak_step})")
    ax2.set_title(f"SIR trajectory driving wave\nβ={beta:.3f}  γ={gamma:.3f}  R0={r0:.2f}  I0={i0:.3f}",
                  fontsize=10)
    ax2.set_xlabel("Months from trend start")
    ax2.set_ylabel("Fraction / wave intensity")
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(fontsize=8)

    plt.suptitle("SIR-driven agent-based diffusion model", fontsize=11, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"{folder_name}_{clean}_sir_abm.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

print("=" * 70)

# =========================================================
# 8) SUMMARY STATISTICS
# =========================================================

summary_df = pd.DataFrame(summary_rows)

if len(summary_df):
    mean_rmse = summary_df["abm_rmse"].mean()
    mean_corr = summary_df["abm_corr"].mean(skipna=True)
    print(f"\nOverall  mean RMSE = {mean_rmse:.4f}   mean corr = {mean_corr:.3f}")

    print("\nBy era:")
    for era, grp in summary_df.groupby("era"):
        print(f"  {era:<12}  RMSE={grp['abm_rmse'].mean():.4f}  "
              f"corr={grp['abm_corr'].mean(skipna=True):.3f}  n={len(grp)}")

    print("\nBy folder (trend category):")
    for folder, grp in summary_df.groupby("folder"):
        print(f"  {folder:<14}  RMSE={grp['abm_rmse'].mean():.4f}  "
              f"corr={grp['abm_corr'].mean(skipna=True):.3f}  n={len(grp)}")

    out_csv = os.path.join(OUTPUT_DIR, "abm_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nExported summary: {out_csv}")

print("\nDone.")
