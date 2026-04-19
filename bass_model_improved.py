"""
Fashion Trend Curve Forecasting — Improved Bass + Log-Normal Pipeline
======================================================================
Fixes over the original bass_diffusion.py:
  1. Loads ALL of Alicia's data (Macrotrends / Microtrends / MegaTrends)
  2. Class-aware priors — p/q bounds and training window depend on Micro/Macro/Mega
  3. Pre-ignition trimming — ignores flat zeros before a trend starts
  4. Dual model fitting — tries Bass AND log-normal, picks whichever fits better
  5. Bootstrap confidence bands on the forecast
  6. Summary dashboard per trend class
  7. Saves per-trend forecast CSVs for downstream use

"""

import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import lognorm

warnings.filterwarnings("ignore")

OUT = "bass_output"
os.makedirs(OUT, exist_ok=True)

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0d0d",
    "axes.facecolor":   "#1a1a1a",
    "axes.grid":        True,
    "grid.alpha":       0.15,
    "grid.color":       "#444",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "gray",
    "ytick.color":      "gray",
    "font.size":        9,
})

COLOURS = {"Micro": "#e63946", "Macro": "#457b9d", "Mega": "#2d6a4f"}

DATE_COLS = ["Month", "Date", "date", "month", "Week", "Time"]

# ── class-aware configuration ─────────────────────────────────────────────────
# Each class gets its own:
#   train_frac   — how much of the active (post-ignition) period to train on
#   p_bounds     — (min, max) for innovation coefficient
#   q_bounds     — (min, max) for imitation coefficient
#   p0_grid      — starting points for optimiser

CLASS_CONFIG = {
    "Micro": {
        "train_frac": 0.5,          # need at least half the spike to fit
        "p_bounds":   (0.01, 0.9),  # Micro trends spread fast via innovation/viral
        "q_bounds":   (0.05, 2.5),
        "p0_grid":    [[0.1, 0.8], [0.2, 1.0], [0.05, 0.5], [0.3, 1.5]],
    },
    "Macro": {
        "train_frac": 0.45,
        "p_bounds":   (0.005, 0.5),
        "q_bounds":   (0.05, 1.5),
        "p0_grid":    [[0.03, 0.5], [0.05, 0.4], [0.01, 0.8], [0.08, 0.3]],
    },
    "Mega": {
        "train_frac": 0.4,          # long trends — can fit on less
        "p_bounds":   (0.001, 0.2),  # slow-burning, low innovation rate
        "q_bounds":   (0.01, 1.0),
        "p0_grid":    [[0.01, 0.3], [0.02, 0.5], [0.005, 0.4], [0.03, 0.2]],
    },
}

RMSE_THRESHOLD = 0.12   # if best model RMSE > this, flag as poor fit


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def find_date_col(df):
    for c in DATE_COLS:
        if c in df.columns:
            return c
    return None

def find_value_col(df, date_col):
    for col in df.columns:
        if col == date_col:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() >= len(df) * 0.5:
            return col
    return None

def load_single_csv(path):
    for skip in (0, 1):
        try:
            df = pd.read_csv(path, skiprows=skip)
            dc = find_date_col(df)
            if dc is None: continue
            vc = find_value_col(df, dc)
            if vc is None: continue
            df = df[[dc, vc]].copy()
            df.columns = ["date", "value"]
            df["date"]  = pd.to_datetime(df["date"],  errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().sort_values("date").reset_index(drop=True)
            if len(df) > 10:
                return df
        except Exception:
            pass
    return None

def normalise(s):
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)

def load_all_trends(data_folder):
    folder_label = {
        "Macrotrends": "Macro",
        "MegaTrends":  "Mega",
        "Microtrends": "Micro",
    }
    trends = {}
    for folder, label in folder_label.items():
        fp = os.path.join(data_folder, folder)
        if not os.path.isdir(fp):
            print(f"[WARN] not found: {fp}")
            continue
        csvs = [f for f in glob.glob(os.path.join(fp, "*.csv"))
                if "combined" not in os.path.basename(f).lower()
                and not os.path.basename(f).startswith(".")]
        for path in csvs:
            name = os.path.splitext(os.path.basename(path))[0]
            df = load_single_csv(path)
            if df is not None:
                df["value_norm"] = normalise(df["value"])
                trends[name] = {"df": df, "label": label}
    print(f"[INFO] Loaded {len(trends)} trends.\n")
    return trends


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PRE-IGNITION TRIMMING
# ═══════════════════════════════════════════════════════════════════════════

def trim_pre_ignition(series, threshold=0.05):
    """
    Find the first point where the trend actually starts (value > threshold).
    Returns (trimmed_series, offset) where offset is the number of months trimmed.
    This stops the model wasting its training budget fitting flat zeros.
    """
    arr = np.array(series)
    for i, v in enumerate(arr):
        if v > threshold:
            # start a few months before ignition for context
            start = max(0, i - 2)
            return arr[start:], start
    return arr, 0  # if never ignites, return as-is


# ═══════════════════════════════════════════════════════════════════════════
# 3.  BASS MODEL
# ═══════════════════════════════════════════════════════════════════════════

def bass_incremental(t, p, q):
    M = 1.0
    e = np.exp(-(p + q) * t)
    denom = (1 + (q / p) * e) ** 2
    return M * ((p + q) ** 2 / p) * e / denom

def bass_cumulative(t, p, q):
    M = 1.0
    e = np.exp(-(p + q) * t)
    return M * (1 - e) / (1 + (q / p) * e)

def fit_bass_model(active_series, label):
    """Fit Bass model using class-aware priors."""
    cfg = CLASS_CONFIG[label]
    n = len(active_series)
    n_train = max(6, int(n * cfg["train_frac"]))
    t_train = np.arange(n_train, dtype=float)
    t_full  = np.arange(n, dtype=float)

    # fit on cumulative (more stable)
    y_raw = active_series[:n_train]
    y_cum = np.cumsum(y_raw)
    y_cum_norm = y_cum / y_cum.max() if y_cum.max() > 0 else y_cum

    best_result, best_rmse = None, np.inf

    for p0 in cfg["p0_grid"]:
        try:
            popt, pcov = curve_fit(
                bass_cumulative,
                t_train, y_cum_norm,
                p0=p0,
                bounds=([cfg["p_bounds"][0], cfg["q_bounds"][0]],
                        [cfg["p_bounds"][1], cfg["q_bounds"][1]]),
                maxfev=30000
            )
            pred = bass_cumulative(t_train, *popt)
            rmse = np.sqrt(np.mean((pred - y_cum_norm) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_result = (popt, pcov)
        except Exception:
            continue

    if best_result is None:
        return None

    p, q = best_result[0]
    perr = np.sqrt(np.diag(best_result[1]))

    # scale incremental predictions to match observed peak
    pred_full = bass_incremental(t_full, p, q)
    obs_peak  = active_series.max()
    mod_peak  = pred_full.max()
    if mod_peak > 0:
        pred_full = pred_full * (obs_peak / mod_peak)

    train_rmse = np.sqrt(np.mean((pred_full[:n_train] - active_series[:n_train]) ** 2))
    test_rmse  = np.sqrt(np.mean((pred_full[n_train:] - active_series[n_train:]) ** 2)) \
                 if n > n_train else train_rmse

    return {
        "model":       "Bass",
        "p":           p, "q": q,
        "p_err":       perr[0], "q_err": perr[1],
        "pq_ratio":    q / p,
        "peak_month":  np.log(q / p) / (p + q) if q > p else 0,
        "n_train":     n_train,
        "n_total":     n,
        "pred_full":   pred_full,
        "train_rmse":  train_rmse,
        "test_rmse":   test_rmse,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  LOG-NORMAL MODEL (better for asymmetric spikes)
# ═══════════════════════════════════════════════════════════════════════════

def lognormal_curve(t, mu, sigma, scale):
    """Log-normal PDF scaled by `scale`."""
    t = np.maximum(t, 1e-6)
    return scale * np.exp(-((np.log(t) - mu) ** 2) / (2 * sigma ** 2)) / (t * sigma * np.sqrt(2 * np.pi))

def fit_lognormal_model(active_series, label):
    """Fit a log-normal curve — better for sharp asymmetric Micro/Macro trends."""
    cfg = CLASS_CONFIG[label]
    n = len(active_series)
    n_train = max(6, int(n * cfg["train_frac"]))
    t_train = np.arange(1, n_train + 1, dtype=float)
    t_full  = np.arange(1, n + 1, dtype=float)

    y_train = active_series[:n_train]

    peak_idx = np.argmax(active_series[:n_train]) + 1
    p0 = [np.log(max(peak_idx, 1)), 0.5, active_series.max() * 2]

    best_result, best_rmse = None, np.inf

    for sigma_init in [0.3, 0.5, 0.8, 1.0]:
        for scale_init in [active_series.max(), active_series.max() * 2, active_series.max() * 3]:
            try:
                popt, _ = curve_fit(
                    lognormal_curve,
                    t_train, y_train,
                    p0=[np.log(max(peak_idx, 1)), sigma_init, scale_init],
                    bounds=([0, 0.1, 0], [np.log(n + 1) + 1, 3.0, 10.0]),
                    maxfev=20000
                )
                pred = lognormal_curve(t_train, *popt)
                rmse = np.sqrt(np.mean((pred - y_train) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_result = popt
            except Exception:
                continue

    if best_result is None:
        return None

    pred_full  = lognormal_curve(t_full, *best_result)
    train_rmse = np.sqrt(np.mean((pred_full[:n_train] - active_series[:n_train]) ** 2))
    test_rmse  = np.sqrt(np.mean((pred_full[n_train:] - active_series[n_train:]) ** 2)) \
                 if n > n_train else train_rmse

    return {
        "model":      "LogNormal",
        "mu":         best_result[0],
        "sigma":      best_result[1],
        "scale":      best_result[2],
        "n_train":    n_train,
        "n_total":    n,
        "pred_full":  pred_full,
        "train_rmse": train_rmse,
        "test_rmse":  test_rmse,
        "peak_month": np.exp(best_result[0]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  BOOTSTRAP CONFIDENCE BANDS
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_confidence(active_series, fit_fn, label, n_boot=100):
    """
    Resample residuals from the best fit and refit n_boot times.
    Returns (lower_5th_percentile, upper_95th_percentile) arrays.
    """
    result = fit_fn(active_series, label)
    if result is None:
        return None, None

    residuals   = active_series - result["pred_full"]
    n           = len(active_series)
    boot_preds  = []

    for _ in range(n_boot):
        noise        = np.random.choice(residuals, size=n, replace=True)
        boot_series  = np.clip(result["pred_full"] + noise, 0, 1)
        boot_result  = fit_fn(boot_series, label)
        if boot_result is not None:
            boot_preds.append(boot_result["pred_full"])

    if len(boot_preds) < 10:
        return None, None

    boot_arr = np.array(boot_preds)
    return np.percentile(boot_arr, 5, axis=0), np.percentile(boot_arr, 95, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  FIT BOTH MODELS AND PICK WINNER
# ═══════════════════════════════════════════════════════════════════════════

def fit_best_model(active_series, label):
    """Try Bass and LogNormal, return whichever has lower test RMSE."""
    bass_res = fit_bass_model(active_series, label)
    lnrm_res = fit_lognormal_model(active_series, label)

    if bass_res is None and lnrm_res is None:
        return None
    if bass_res is None:
        return lnrm_res
    if lnrm_res is None:
        return bass_res

    # pick by test RMSE
    return bass_res if bass_res["test_rmse"] <= lnrm_res["test_rmse"] else lnrm_res


# ═══════════════════════════════════════════════════════════════════════════
# 7.  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_trend(name, label, df_row, result, lo, hi, ax):
    """Plot a single trend's observed data + model fit + confidence band."""
    colour  = COLOURS[label]
    obs     = df_row["value_norm"].values
    offset  = result.get("offset", 0)
    n_train = result["n_train"]
    n_total = result["n_total"]
    t_full  = np.arange(n_total)
    t_obs   = np.arange(offset, offset + len(obs))

    # observed
    ax.plot(t_obs, obs, color=colour, alpha=0.6, linewidth=1.3, label="Observed")

    # training window shade
    ax.axvspan(offset, offset + n_train - 1,
               alpha=0.08, color="steelblue")
    ax.axvline(offset + n_train - 1, color="steelblue",
               linestyle=":", alpha=0.5, linewidth=1)

    t_model = np.arange(offset, offset + n_total)

    # confidence band
    if lo is not None and hi is not None:
        ax.fill_between(t_model, lo, hi, alpha=0.18, color=colour)

    # fit on train
    ax.plot(t_model[:n_train], result["pred_full"][:n_train],
            "w--", linewidth=1.8, alpha=0.85,
            label=f"{result['model']} fit (train RMSE={result['train_rmse']:.3f})")

    # forecast
    ax.plot(t_model[n_train:], result["pred_full"][n_train:],
            color=colour, linewidth=2, linestyle="--",
            label=f"Forecast (test RMSE={result['test_rmse']:.3f})")

    # peak marker
    peak_m = result.get("peak_month", None)
    if peak_m is not None and 0 < peak_m < n_total:
        ax.axvline(offset + peak_m, color="orange",
                   linestyle="--", alpha=0.4, linewidth=1)

    quality = "⚠ poor fit" if result["test_rmse"] > RMSE_THRESHOLD else "✓"
    ax.set_title(f"{name}  [{label}]  {quality}", fontsize=8, color="white")
    ax.set_ylim(-0.05, 1.2)
    ax.legend(fontsize=6, loc="upper right",
              facecolor="#2a2a2a", edgecolor="none", labelcolor="white")
    ax.set_xlabel("Month (from 2004)", fontsize=7)


def plot_class_grid(all_results, trends, label, max_per_page=9):
    """Plot a grid of all trends for a given class."""
    items = [(n, r) for n, r in all_results.items()
             if trends[n]["label"] == label]

    if not items:
        return

    n_cols = 3
    n_rows = min(3, int(np.ceil(len(items) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(15, n_rows * 4))
    fig.patch.set_facecolor("#0d0d0d")
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, (name, res) in zip(axes_flat, items[:max_per_page]):
        ax.set_facecolor("#1a1a1a")
        plot_trend(name, label, trends[name]["df"], res["result"],
                   res["lo"], res["hi"], ax)

    # hide unused panels
    for ax in list(axes_flat)[len(items):]:
        ax.set_visible(False)

    fig.suptitle(f"{label} Trends — Curve Fitting & Forecast",
                 color="white", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT, f"forecast_{label.lower()}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_rmse(all_results, trends):
    """Bar chart of test RMSE for all trends, coloured by class."""
    names  = list(all_results.keys())
    rmses  = [all_results[n]["result"]["test_rmse"] for n in names]
    models = [all_results[n]["result"]["model"]     for n in names]
    cols   = [COLOURS[trends[n]["label"]]           for n in names]

    order  = np.argsort(rmses)
    names  = [names[i]  for i in order]
    rmses  = [rmses[i]  for i in order]
    models = [models[i] for i in order]
    cols   = [cols[i]   for i in order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.28)))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#1a1a1a")

    bars = ax.barh(names, rmses, color=cols, alpha=0.85)
    ax.axvline(RMSE_THRESHOLD, color="white", linestyle="--",
               linewidth=0.8, alpha=0.6, label=f"Poor fit threshold ({RMSE_THRESHOLD})")

    for bar, model, rmse in zip(bars, models, rmses):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{model}  {rmse:.3f}", va="center", color="white", fontsize=7)

    ax.set_xlabel("Test RMSE (lower = better forecast)", color="white")
    ax.set_title("Forecast Quality — All Trends", color="white", fontsize=12)
    ax.tick_params(colors="white", labelsize=7)
    ax.legend(labelcolor="white", facecolor="#2a2a2a", edgecolor="none")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # legend for classes
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=l)
                  for l, c in COLOURS.items()]
    ax.legend(handles=legend_els, labelcolor="white",
              facecolor="#2a2a2a", edgecolor="none", loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUT, "forecast_rmse_summary.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_class_archetypes(all_results, trends):
    """
    Average predicted curve per class — shows the archetypal shape
    the model has learned for each trend type.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d0d0d")

    for ax, label in zip(axes, ["Micro", "Macro", "Mega"]):
        ax.set_facecolor("#1a1a1a")
        colour = COLOURS[label]
        preds  = []

        for name, res in all_results.items():
            if trends[name]["label"] != label:
                continue
            # normalise to 0-1 length for averaging
            p = res["result"]["pred_full"]
            t_norm = np.linspace(0, 1, len(p))
            preds.append((t_norm, p))

        if not preds:
            continue

        # interpolate all to common 100-point grid
        t_common = np.linspace(0, 1, 100)
        interp_preds = []
        for t_norm, p in preds:
            interp_preds.append(np.interp(t_common, t_norm, p))

        arr  = np.array(interp_preds)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)

        ax.fill_between(t_common, mean - std, mean + std,
                        alpha=0.2, color=colour)
        ax.plot(t_common, mean, color=colour, linewidth=2.5,
                label=f"Mean {label} curve (n={len(preds)})")

        # individual trends faint
        for p in interp_preds:
            ax.plot(t_common, p, color=colour, alpha=0.12, linewidth=0.8)

        ax.set_title(f"{label} Archetype", color="white", fontsize=11)
        ax.set_xlabel("Normalised time (0=start, 1=end)", color="white")
        ax.set_ylabel("Normalised interest", color="white")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8, facecolor="#2a2a2a",
                  edgecolor="none", labelcolor="white")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    fig.suptitle("Archetypal Curve Shapes by Trend Class",
                 color="white", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT, "archetype_curves.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="data")
    parser.add_argument("--n_boot", type=int, default=80,
                        help="Bootstrap iterations for confidence bands (default 80)")
    args = parser.parse_args()

    trends = load_all_trends(args.data_folder)
    if not trends:
        raise RuntimeError("No trends loaded — check data_folder path.")

    all_results = {}
    summary_rows = []

    print(f"{'Trend':30s} {'Class':6s} {'Model':10s} "
          f"{'Train RMSE':12s} {'Test RMSE':10s} {'Status'}")
    print("-" * 80)

    for name, info in trends.items():
        label  = info["label"]
        series = info["df"]["value_norm"].values.copy()

        # trim pre-ignition zeros
        active, offset = trim_pre_ignition(series, threshold=0.05)

        if len(active) < 8:
            print(f"  [SKIP] {name} — too short after trimming")
            continue

        # fit best model
        result = fit_best_model(active, label)
        if result is None:
            print(f"  [FAIL] {name} — fitting failed")
            continue

        result["offset"] = offset

        # bootstrap confidence bands
        fit_fn  = (fit_bass_model if result["model"] == "Bass"
                   else fit_lognormal_model)
        lo, hi  = bootstrap_confidence(active, fit_fn, label, n_boot=args.n_boot)

        all_results[name] = {"result": result, "lo": lo, "hi": hi}

        status = "⚠" if result["test_rmse"] > RMSE_THRESHOLD else "✓"
        print(f"  {name:28s} {label:6s} {result['model']:10s} "
              f"{result['train_rmse']:.4f}       {result['test_rmse']:.4f}     {status}")

        summary_rows.append({
            "trend":       name,
            "label":       label,
            "model":       result["model"],
            "train_rmse":  round(result["train_rmse"], 4),
            "test_rmse":   round(result["test_rmse"],  4),
            "peak_month":  round(result.get("peak_month", 0), 1),
            "n_train":     result["n_train"],
            "n_total":     result["n_total"],
            "good_fit":    result["test_rmse"] <= RMSE_THRESHOLD,
        })

        # save per-trend forecast CSV
        pred = result["pred_full"]
        t_idx = np.arange(offset, offset + len(pred))
        forecast_df = pd.DataFrame({
            "month_index": t_idx,
            "predicted":   pred,
            "lo_5":        lo  if lo  is not None else [np.nan] * len(pred),
            "hi_95":       hi  if hi  is not None else [np.nan] * len(pred),
        })
        forecast_df.to_csv(
            os.path.join(OUT, f"{name}_forecast.csv"), index=False)

    # summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT, "forecast_summary.csv"), index=False)

    # ── stats ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label in ["Micro", "Macro", "Mega"]:
        sub = summary_df[summary_df["label"] == label]
        if sub.empty: continue
        good = sub["good_fit"].sum()
        print(f"  {label:6s}  n={len(sub):2d}  "
              f"good fits={good}/{len(sub)}  "
              f"mean test RMSE={sub['test_rmse'].mean():.4f}  "
              f"Bass wins={( sub['model']=='Bass').sum()}  "
              f"LogNormal wins={(sub['model']=='LogNormal').sum()}")

    # ── plots ──────────────────────────────────────────────────────────────
    print("\n[INFO] Generating plots...")
    for label in ["Micro", "Macro", "Mega"]:
        plot_class_grid(all_results, trends, label)

    plot_summary_rmse(all_results, trends)
    plot_class_archetypes(all_results, trends)

    print(f"\nAll outputs saved to: ./{OUT}/")


if __name__ == "__main__":
    main()