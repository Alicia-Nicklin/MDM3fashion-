"""
Agent-Based Model — SIR-driven wave + Optuna calibration (final version)
MDM3 Fashion Trends Project

Logic:
  1. Load fitted SIR parameters from output/sir_parameters.csv (Tiggy's output)
  2. Solve I(t) using those parameters (RK4, same solver as SIR script)
  3. Build wave from SIR I(t) — no hand-crafted Gaussian or logistic shapes
  4. Use Optuna to tune ABM hyperparameters globally across all matched trends
     (SIR structure is FIXED — Optuna only calibrates the agent-level response)
  5. Run final ABM with best parameters, print RMSE + correlation per trend,
     summarise by era / category, export abm_summary.csv

Why Optuna:
  Supervisor feedback: before claiming ABM has limitations, check the fit is not
  just due to poor parameter choices. Optuna gives systematic calibration so we
  can say "even after tuning, the ABM still struggled on X" — a defensible claim.

  Optuna tunes ONLY:  social_weight, start_chance, fade_base, late_drop,
                      start_share, post_peak_suppression, noise
  Optuna does NOT touch: beta, gamma, I0, R0, the SIR wave shape, or any
                          clipping / windowing logic.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: optuna not installed. Run:  pip install optuna")
    print("         Falling back to SIR-scaled defaults.\n")

# =========================================================
# 1) SETUP
# =========================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SIR_PATH   = os.path.join(OUTPUT_DIR, "sir_parameters.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTIVE_THRESHOLD = 0.35
GAP_TOLERANCE    = 1
BUFFER           = 5
N_PEOPLE         = 1000

# Optuna settings
# Start with N_TRIALS=40, N_RUNS=2 — fast enough to finish in a few minutes.
# If it completes comfortably, increase to N_TRIALS=100, N_RUNS=3 and rerun.
N_TRIALS    = 40    # increase to 100 if runtime is fine
N_RUNS      = 2     # ABM runs per trial (averaged to reduce stochastic noise)
N_CI_RUNS   = 30    # runs for final CI bands (mean + 2.5th/97.5th percentile)

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

MANUAL_NAME_MAP = {}   # names now match filenames directly after rerunning data_loader + SIR

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
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def minmax_scale(val, vmin, vmax, out_min, out_max):
    if vmax <= vmin:
        return 0.5 * (out_min + out_max)
    return out_min + np.clip((val - vmin) / (vmax - vmin), 0, 1) * (out_max - out_min)


def clean_era_label(era):
    if pd.isna(era):
        return "Unknown"
    return str(era).split("(")[0].strip()


def normalise_name(s):
    return str(s).lower().replace("_", "").replace("-", "").replace(" ", "").strip()


def _fill_gaps(mask, tol=GAP_TOLERANCE):
    mask = list(mask)
    i = 0
    while i < len(mask):
        if not mask[i]:
            start = i
            while i < len(mask) and not mask[i]:
                i += 1
            end = i - 1
            left  = start > 0           and mask[start - 1]
            right = end < len(mask) - 1 and mask[end + 1]
            if left and right and (end - start + 1) <= tol:
                for j in range(start, end + 1):
                    mask[j] = True
        else:
            i += 1
    return np.array(mask, dtype=bool)


def _longest_run(mask):
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


def find_trend_file(trend_name):
    lookup = MANUAL_NAME_MAP.get(trend_name, trend_name)
    target = normalise_name(lookup)
    for folder_name, folder_path in SEARCH_DIRS.items():
        if not os.path.exists(folder_path):
            continue
        for path in glob.glob(os.path.join(folder_path, "*.csv")):
            stem = os.path.splitext(os.path.basename(path))[0]
            if "combined" in stem.lower():
                continue
            if normalise_name(stem) == target:
                return folder_name, path
    return None, None


def infer_trend_type(folder_name):
    return folder_name if folder_name in SEARCH_DIRS else "Macrotrends"

# =========================================================
# 3) SIR SOLVER  (RK4 — identical to Tiggy's implementation)
# =========================================================

def solve_sir_I(beta, gamma, I0, n_periods, sub=4):
    """Return the I(t) trajectory from the fitted SIR model."""
    dt  = 1.0 / sub
    h6  = dt / 6.0
    h2  = dt / 2.0
    S, I = max(1.0 - I0, 1e-9), float(I0)
    I_out = np.empty(n_periods)
    I_out[0] = I
    out_idx = 1

    for step in range(1, (n_periods - 1) * sub + 1):
        bSI = beta * S * I
        k1s = -bSI;             k1i = bSI - gamma * I
        S2 = S + h2*k1s;        I2  = I + h2*k1i
        bSI = beta * S2 * I2
        k2s = -bSI;             k2i = bSI - gamma*I2
        S3 = S + h2*k2s;        I3  = I + h2*k2i
        bSI = beta * S3 * I3
        k3s = -bSI;             k3i = bSI - gamma*I3
        S4 = S + dt*k3s;        I4  = I + dt*k3i
        bSI = beta * S4 * I4
        k4s = -bSI;             k4i = bSI - gamma*I4

        S = max(S + h6*(k1s + 2*k2s + 2*k3s + k4s), 0.0)
        I = max(I + h6*(k1i + 2*k2i + 2*k3i + k4i), 0.0)

        if step % sub == 0 and out_idx < n_periods:
            I_out[out_idx] = I
            out_idx += 1

    return I_out

# =========================================================
# 4) SIR-DERIVED WAVE  (structure fixed — Optuna does not touch this)
# =========================================================

def build_sir_wave(beta, gamma, I0, n_steps):
    """
    Build the ABM driving wave directly from the fitted SIR infected curve.
    No class-specific shaping, no manual lifecycle edits.
    Wave is the normalised SIR I(t) — purely Tiggy's fitted parameters.
    """
    sir_I = solve_sir_I(beta, gamma, I0, n_steps)
    wave  = safe_normalise(sir_I)
    return wave, sir_I

# =========================================================
# 5) ABM  (wave is a pre-computed input, not built inside)
# =========================================================

def simulate_trend(
    wave, sir_I, trend_type, n_steps,
    social_weight, start_chance, fade_base, late_drop, start_share,
    post_peak_suppression, noise,
    start_delay=0,
    seed=None,
):
    """
    Run one ABM simulation over n_steps monthly steps.

    Parameters controlled by Optuna (ABM-side only):
      social_weight, start_chance, fade_base, late_drop, start_share,
      post_peak_suppression, noise

    Parameters fixed from SIR (never tuned):
      wave, sir_I, peak_step (derived from sir_I)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    peak_step = int(np.argmax(sir_I))   # from SIR — never from real data

    state = np.zeros(N_PEOPLE, dtype=int)
    state[:max(1, int(start_share * N_PEOPLE))] = 1
    rng.shuffle(state)

    uptake_w = rng.uniform(0.9, 1.1, N_PEOPLE)
    fade_w   = rng.uniform(0.9, 1.1, N_PEOPLE)
    series   = np.zeros(n_steps)

    for i in range(n_steps):
        if i < start_delay:
            continue

        share = state.mean()

        join_rate  = (start_chance + social_weight * (share ** 1.1)) * wave[i]
        join_rate += rng.normal(0, noise)

        if i > peak_step:
            join_rate *= post_peak_suppression

        extra_fade = (
            late_drop * (i - peak_step) / max(1, n_steps - peak_step)
            if i > peak_step else 0.0
        )
        leave_rate  = fade_base + extra_fade + rng.normal(0, noise / 2)

        join_prob  = np.clip(join_rate  * uptake_w, 0, 0.95)
        leave_prob = np.clip(leave_rate * fade_w,   0, 0.95)

        rj = rng.random(N_PEOPLE)
        rl = rng.random(N_PEOPLE)
        state[(state == 0) & (rj < join_prob)]  = 1
        state[(state == 1) & (rl < leave_prob)] = 0

        series[i] = state.mean()

    if series.max() > 0:
        series /= series.max()

    return series


def run_abm_mean(wave, sir_I, trend_type, n_steps, params, start_delay, n_runs=N_RUNS):
    """Run ABM n_runs times and return the mean (reduces stochastic noise)."""
    runs = [
        simulate_trend(
            wave=wave, sir_I=sir_I, trend_type=trend_type, n_steps=n_steps,
            start_delay=start_delay, seed=42 + k, **params
        )
        for k in range(n_runs)
    ]
    return np.mean(runs, axis=0)

# =========================================================
# 6) LOAD SIR PARAMETERS + BUILD TREND DATASET
# =========================================================

if not os.path.exists(SIR_PATH):
    raise FileNotFoundError(
        f"Could not find {SIR_PATH}\n"
        "Run Tiggy's SIR fitting script first."
    )

sir_df = pd.read_csv(SIR_PATH)
required = {"trend", "beta", "gamma", "R0", "I0", "era"}
missing  = required - set(sir_df.columns)
if missing:
    raise ValueError(f"sir_parameters.csv is missing columns: {missing}")

# Drop degenerate SIR fits before doing anything else.
# These are trends where the optimiser failed — either gamma collapsed to zero
# producing absurd R0 values, or the series was too short to fit meaningfully.
n_before = len(sir_df)
sir_df = sir_df[
    (sir_df["R0"]    >  0)    &   # sanity check
    (sir_df["R0"]    < 500)   &   # excludes blown-up fits where gamma -> 0
    (sir_df["gamma"] > 0.005) &   # directly removes the root cause of R0 explosions
    (sir_df["rmse"]  < 0.20)      # excludes default-start failures / too-short series
].reset_index(drop=True)
n_dropped = n_before - len(sir_df)
print(f"\nLoaded {n_before} trends from sir_parameters.csv")
print(f"Dropped {n_dropped} degenerate fits (R0 >= 500, gamma <= 0.005, or RMSE >= 0.20)")
print(f"Retained {len(sir_df)} trends for ABM\n")

beta_min,  beta_max  = sir_df["beta"].min(),  sir_df["beta"].max()
gamma_min, gamma_max = sir_df["gamma"].min(), sir_df["gamma"].max()
r0_min,    r0_max    = sir_df["R0"].min(),    sir_df["R0"].max()

# Pre-build per-trend dataset (real series, wave, clipping info)
# so both Optuna and the final run use exactly the same data
TRENDS = []

for _, row in sir_df.iterrows():
    trend_name = str(row["trend"]).strip()
    beta  = float(row["beta"])
    gamma = float(row["gamma"])
    r0    = float(row["R0"])
    i0    = float(row["I0"])
    era   = clean_era_label(row["era"])

    folder_name, csv_path = find_trend_file(trend_name)
    if csv_path is None:
        continue

    trend_type = infer_trend_type(folder_name)

    df_raw = pd.read_csv(csv_path)
    real   = pd.to_numeric(df_raw.iloc[:, 1], errors="coerce").fillna(0).values.astype(float)
    real   = safe_normalise(real)
    n_raw  = len(real)

    active_mask = _fill_gaps(real >= ACTIVE_THRESHOLD)
    _, run_start, run_end = _longest_run(active_mask)
    if run_start is None:
        continue

    start_idx   = max(0, run_start - BUFFER)
    end_idx     = min(n_raw - 1, run_end + BUFFER)
    n_steps     = end_idx - start_idx + 1
    start_delay = max(0, (run_start - start_idx) - 2)

    wave, sir_I = build_sir_wave(beta, gamma, i0, n_steps)
    real_clip   = real[start_idx:end_idx + 1]

    # SIR-scaled defaults (used as Optuna search-space centre)
    TRENDS.append({
        "name":        trend_name,
        "era":         era,
        "folder":      folder_name,
        "trend_type":  trend_type,
        "beta":        beta,
        "gamma":       gamma,
        "R0":          r0,
        "I0":          i0,
        "n_steps":     n_steps,
        "start_delay": start_delay,
        "wave":        wave,
        "sir_I":       sir_I,
        "real_clip":   real_clip,
        "start_idx":   start_idx,
        # SIR-informed defaults (Optuna starts near these)
        "sw_default":  minmax_scale(r0,    r0_min,    r0_max,    0.18, 0.58),
        "sc_default":  minmax_scale(beta,  beta_min,  beta_max,  0.002, 0.018),
        "fb_default":  minmax_scale(gamma, gamma_min, gamma_max, 0.015, 0.12),
        "ld_default":  minmax_scale(gamma, gamma_min, gamma_max, 0.05,  0.30),
        "ss_default":  float(np.clip(i0, 0.005, 0.15)),
    })

print(f"Matched {len(TRENDS)} trends to CSV files.\n")

# =========================================================
# 7) OPTUNA — global tuning across all matched trends
#    Tunes: ABM agent-level controls only
#    Fixed: SIR wave shape, beta, gamma, I0, clipping logic
# =========================================================

def abm_objective(trial):
    """
    Optuna objective: mean RMSE across all trends using one global
    parameter set.  The SIR wave for each trend is pre-computed and
    fixed — Optuna only moves the agent-level knobs.
    """
    params = {
        "social_weight":         trial.suggest_float("social_weight",         0.10, 0.70),
        "start_chance":          trial.suggest_float("start_chance",          0.001, 0.025),
        "fade_base":             trial.suggest_float("fade_base",             0.008, 0.15),
        "late_drop":             trial.suggest_float("late_drop",             0.02,  0.40),
        "start_share":           trial.suggest_float("start_share",           0.003, 0.18),
        "post_peak_suppression": trial.suggest_float("post_peak_suppression", 0.10,  0.55),
        "noise":                 trial.suggest_float("noise",                 0.001, 0.015),
    }

    total_rmse = 0.0
    for t in TRENDS:
        fake = run_abm_mean(
            wave=t["wave"], sir_I=t["sir_I"],
            trend_type=t["trend_type"],
            n_steps=t["n_steps"],
            params=params,
            start_delay=t["start_delay"],
            n_runs=2,   # 2 runs inside objective for speed; 3 for final eval
        )
        total_rmse += float(np.sqrt(np.mean((t["real_clip"] - fake) ** 2)))

    return total_rmse / max(1, len(TRENDS))


if OPTUNA_AVAILABLE:
    print(f"Running Optuna ({N_TRIALS} trials, global tuning)...")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(abm_objective, n_trials=N_TRIALS, show_progress_bar=False)

    BEST_PARAMS = study.best_params
    print(f"\nOptuna complete. Best mean RMSE: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in BEST_PARAMS.items():
        print(f"  {k:<28} {v:.5f}")
else:
    # Fallback: use mid-range SIR-scaled defaults
    BEST_PARAMS = {
        "social_weight":         0.38,
        "start_chance":          0.010,
        "fade_base":             0.05,
        "late_drop":             0.18,
        "start_share":           0.02,
        "post_peak_suppression": 0.28,
        "noise":                 0.005,
    }
    print("Using fallback defaults (install optuna for proper tuning).")

# =========================================================
# 8) FINAL RUN — best parameters, full metrics, plots
# =========================================================

summary_rows = []

print("\n" + "=" * 82)
print(f"{'Trend':<26} {'Era':<12} {'Cat':<14} {'RMSE':>7} {'Corr':>7} {'R0':>6} {'Band':>8}")
print("=" * 82)

for t in TRENDS:
    # --- run N_CI_RUNS times to get simulation uncertainty bands ---
    all_runs = np.array([
        simulate_trend(
            wave=t["wave"], sir_I=t["sir_I"],
            trend_type=t["trend_type"],
            n_steps=t["n_steps"],
            start_delay=t["start_delay"],
            seed=200 + k,
            **BEST_PARAMS,
        )
        for k in range(N_CI_RUNS)
    ])  # shape: (N_CI_RUNS, n_steps)

    mean_sim = all_runs.mean(axis=0)
    lo_sim   = np.percentile(all_runs, 2.5,  axis=0)
    hi_sim   = np.percentile(all_runs, 97.5, axis=0)

    real_clip = t["real_clip"]

    # metrics computed on the mean trajectory
    rmse_abm = float(np.sqrt(np.mean((real_clip - mean_sim) ** 2)))
    if len(real_clip) > 1 and real_clip.std() > 0 and mean_sim.std() > 0:
        corr_abm = float(np.corrcoef(real_clip, mean_sim)[0, 1])
    else:
        corr_abm = float("nan")

    # fraction of real data points that fall inside the 95% simulation band
    inside = float(np.mean((real_clip >= lo_sim) & (real_clip <= hi_sim)))

    print(f"  {t['name']:<26} {t['era']:<12} {t['folder']:<14} "
          f"{rmse_abm:>7.4f} {corr_abm:>7.3f} {t['R0']:>6.2f}  "
          f"in-band={inside:.0%}")

    summary_rows.append({
        "trend":          t["name"],
        "era":            t["era"],
        "category":       t["folder"],
        "beta":           round(t["beta"],  4),
        "gamma":          round(t["gamma"], 4),
        "R0":             round(t["R0"],    3),
        "I0":             round(t["I0"],    4),
        "n_months":       t["n_steps"],
        "abm_rmse":       round(rmse_abm, 4),
        "abm_corr":       round(corr_abm, 4) if np.isfinite(corr_abm) else None,
        "ci_coverage":    round(inside, 3),
        **{k: round(v, 5) for k, v in BEST_PARAMS.items()},
    })

    # --- plot ---
    color      = ERA_COLORS.get(t["era"], "#888888")
    clean_name = t["name"].replace(" ", "_").lower()
    x_clip     = np.arange(t["start_idx"], t["start_idx"] + t["n_steps"])
    peak_step  = int(np.argmax(t["sir_I"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # left: ABM mean + 95% simulation band vs real
    ax = axes[0]
    ax.fill_between(x_clip, lo_sim, hi_sim,
                    color="black", alpha=0.12, label="95% simulation interval")
    ax.plot(x_clip, real_clip, label="Real trend",      lw=2,   color=color, alpha=0.85)
    ax.plot(x_clip, mean_sim,  label="ABM mean",        lw=2,   color="black", linestyle="--")
    ax.axhline(ACTIVE_THRESHOLD, color="grey", linestyle=":", lw=1, label="Threshold = 0.35")
    ax.set_title(
        f"{t['name']}  [{t['era']}]\n"
        f"RMSE = {rmse_abm:.4f}   corr = {corr_abm:.3f}   in-band = {inside:.0%}",
        fontweight="bold", fontsize=10
    )
    ax.set_xlabel("Months from series start")
    ax.set_ylabel("Normalised interest [0, 1]")
    ax.set_ylim(-0.05, 1.18)
    ax.legend(fontsize=8)

    # right: SIR I(t) that drove the wave
    ax2 = axes[1]
    ts  = np.arange(t["n_steps"])
    ax2.plot(ts, safe_normalise(t["sir_I"]), color=color, lw=2,
             label="SIR I(t) [normalised]")
    ax2.plot(ts, safe_normalise(t["wave"]),  color="black", lw=1.5,
             linestyle=":", label="Wave (normalised)")
    ax2.axvline(peak_step, color="grey", linestyle="--", lw=1,
                label=f"SIR peak (t = {peak_step})")
    ax2.set_title(
        f"SIR trajectory → wave\n"
        f"β={t['beta']:.3f}  γ={t['gamma']:.3f}  R0={t['R0']:.2f}  I0={t['I0']:.3f}",
        fontsize=10
    )
    ax2.set_xlabel("Months from trend start")
    ax2.set_ylabel("Normalised value")
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(fontsize=8)

    plt.suptitle(
        f"SIR-driven ABM  |  Optuna-calibrated  |  95% simulation interval ({N_CI_RUNS} runs)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{t['folder']}_{clean_name}_sir_optuna_abm.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

print("=" * 82)

# =========================================================
# 9) SUMMARY STATISTICS
# =========================================================

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    mean_rmse  = summary_df["abm_rmse"].mean()
    mean_corr  = pd.to_numeric(summary_df["abm_corr"], errors="coerce").mean()
    mean_cov   = pd.to_numeric(summary_df["ci_coverage"], errors="coerce").mean()
    print(f"\nOverall   RMSE = {mean_rmse:.4f}   corr = {mean_corr:.3f}   "
          f"CI coverage = {mean_cov:.0%}   n = {len(summary_df)}")

    print("\nBy era:")
    for era, grp in summary_df.groupby("era"):
        print(f"  {era:<14}  RMSE = {grp['abm_rmse'].mean():.4f}  "
              f"corr = {pd.to_numeric(grp['abm_corr'], errors='coerce').mean():.3f}  "
              f"CI cov = {pd.to_numeric(grp['ci_coverage'], errors='coerce').mean():.0%}  "
              f"n = {len(grp)}")

    print("\nBy category:")
    for cat, grp in summary_df.groupby("category"):
        print(f"  {cat:<14}  RMSE = {grp['abm_rmse'].mean():.4f}  "
              f"corr = {pd.to_numeric(grp['abm_corr'], errors='coerce').mean():.3f}  "
              f"CI cov = {pd.to_numeric(grp['ci_coverage'], errors='coerce').mean():.0%}  "
              f"n = {len(grp)}")

    out_csv = os.path.join(OUTPUT_DIR, "abm_summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\nExported: {out_csv}")

    # save best params separately so they can be reported easily
    if OPTUNA_AVAILABLE:
        params_df = pd.DataFrame([{
            "param": k, "value": round(v, 5),
            "optuna_trials": N_TRIALS,
            "mean_rmse": round(study.best_value, 4),
        } for k, v in BEST_PARAMS.items()])
        params_path = os.path.join(OUTPUT_DIR, "abm_optuna_best_params.csv")
        params_df.to_csv(params_path, index=False)
        print(f"Exported: {params_path}")

print("\nDone.")
