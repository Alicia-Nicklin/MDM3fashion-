"""
gam1_generalist.py
==================
GAM 1 — Generalist model, completely label-free.

Uses all_trends_combined.csv which contains all 45 trends in one file
with NO category column. The model has absolutely no knowledge of whether
any trend is Micro, Macro, or Mega — it learns purely from lifecycle
shape, seasonality, and momentum features.

Train/test split:
  - 9 trends held out entirely (3 from each category, but the model
    does not know this — the split is just by trend name)
  - Model trained on remaining 36 trends
  - Evaluation: seed on first 40%, rollout forecast remaining 60%

This is the honest baseline. GAM 2 (gam2_specialist.py) adds category
knowledge and should improve on these results.

How to run:
    pip install pygam pandas numpy matplotlib scikit-learn
    python gam1_generalist.py

Required files:
    data/all_trends_combined.csv

Outputs (written to gam_output/):
    gam1_results.csv
    fig_gam1_fits.png
    fig_gam1_rmse.png
    fig_gam1_partial_effects.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, l
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")

os.makedirs("gam_output", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9,
})

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Single combined CSV — no category column, completely label-free
ALL_TRENDS_CSV = os.path.join("data", "all_trends_combined.csv")

# Held-out test trends — removed entirely from training
# The model does not know these are split by category
TEST_TRENDS = [
    "teddycoat", "vsco", "coastalgrandmother",   # (Micro)
    "Darkacademia", "chokers", "Normcore",         # (Macro)
    "hipster fashion", "TUMBLR", "Pastelgrunge",   # (Mega)
]

ROLLOUT_FRAC  = 0.4   # seed on first 40% of active window, forecast remaining 60%
ACTIVE_BUFFER = 3     # months either side of active window kept for training context
THRESHOLD     = 0.35  # active window = value_norm >= 0.35, consistent with myValadation.py

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

# Only used for colouring plots — not passed to the model
KNOWN_CATS = {
    "teddycoat": "Micro", "vsco": "Micro", "coastalgrandmother": "Micro",
    "Darkacademia": "Macro", "chokers": "Macro", "Normcore": "Macro",
    "hipster fashion": "Mega", "TUMBLR": "Mega", "Pastelgrunge": "Mega",
}
CAT_COLOURS  = {"Micro": "#E24B4A", "Macro": "#457b9d", "Mega": "#2d6a4f",
                "Unknown": "#888780"}
FEATURE_COLS = ["t_rel", "month_sin", "month_cos",
                "lag_1", "lag_3", "roll_mean_3", "roll_std_3"]


# ── DATA ──────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(ALL_TRENDS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {df['trend_name'].nunique()} trends loaded")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def add_features(df):
    df = df.copy()
    df["value_norm"] = df["value_norm"].fillna(0).clip(0, 1)
    parts = []
    for _, grp in df.groupby("trend_name", sort=False):
        grp = grp.sort_values("date").copy()
        active = grp[grp["value_norm"] >= THRESHOLD]["date"]
        if len(active) >= 2:
            start = active.iloc[0]
            dur   = max((active.iloc[-1] - start).days / 30.44, 1.0)
        else:
            start = grp["date"].iloc[0]
            dur   = 1.0
        grp["t_rel"]       = (((grp["date"] - start).dt.days / 30.44) / dur).clip(-0.5, 2.5)
        m = grp["date"].dt.month
        grp["month_sin"]   = np.sin(2 * np.pi * m / 12)
        grp["month_cos"]   = np.cos(2 * np.pi * m / 12)
        grp["lag_1"]       = grp["value_norm"].shift(1).fillna(0)
        grp["lag_3"]       = grp["value_norm"].shift(3).fillna(0)
        grp["roll_mean_3"] = grp["value_norm"].shift(1).rolling(3, min_periods=1).mean().fillna(0)
        grp["roll_std_3"]  = grp["value_norm"].shift(1).rolling(3, min_periods=1).std().fillna(0)
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def trim_to_active(df, buffer=ACTIVE_BUFFER):
    """Trim each trend to its active window plus a small buffer either side.
    Prevents the model being dominated by flat-zero rows outside the trend."""
    parts = []
    for _, grp in df.groupby("trend_name", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True).copy()
        active_idx = grp[grp["value_norm"] >= THRESHOLD].index.tolist()
        if len(active_idx) == 0:
            continue
        lo = max(0, active_idx[0] - buffer)
        hi = min(len(grp) - 1, active_idx[-1] + buffer)
        parts.append(grp.iloc[lo:hi + 1])
    return pd.concat(parts, ignore_index=True)


# ── MODEL ─────────────────────────────────────────────────────────────────────

def fit_gam(X, y):
    """
    Single GAM — no category knowledge.
    gridsearch finds the best smoothing penalty automatically.

    Feature index:
      0  t_rel        smooth, 20 splines
      1  month_sin    smooth,  8 splines
      2  month_cos    smooth,  8 splines
      3  lag_1        smooth, 12 splines
      4  lag_3        smooth, 10 splines
      5  roll_mean_3  smooth,  8 splines
      6  roll_std_3   linear
    """
    gam = LinearGAM(
        s(0, n_splines=20) +
        s(1, n_splines=8)  +
        s(2, n_splines=8)  +
        s(3, n_splines=12) +
        s(4, n_splines=10) +
        s(5, n_splines=8)  +
        l(6)
    )
    gam.gridsearch(X, y, progress=False)
    return gam


# ── EVALUATION ────────────────────────────────────────────────────────────────

def rollout(gam, trend_df):
    """
    Seed on first 40% of the active window, forecast remaining 60%.
    RMSE computed on forecast portion of active window only.
    """
    grp    = trend_df.sort_values("date").reset_index(drop=True).copy()
    actual = grp["value_norm"].values.copy()
    pred   = actual.copy()

    active_idx = grp[grp["value_norm"] >= THRESHOLD].index.tolist()
    if len(active_idx) >= 2:
        first_pos = active_idx[0]
        last_pos  = active_idx[-1]
    else:
        first_pos, last_pos = 0, len(grp) - 1

    active_len = last_pos - first_pos + 1
    n_seed     = max(2, int(active_len * ROLLOUT_FRAC))
    seed_end   = first_pos + n_seed

    for i in range(seed_end, last_pos + 1):
        lag1 = pred[i-1] if i >= 1 else 0
        lag3 = pred[i-3] if i >= 3 else 0
        w    = pred[max(0, i-3):i]
        rm3  = float(np.mean(w)) if len(w) > 0 else 0
        rs3  = float(np.std(w))  if len(w) > 1 else 0
        X    = np.array([[
            grp["t_rel"].iloc[i], grp["month_sin"].iloc[i],
            grp["month_cos"].iloc[i], lag1, lag3, rm3, rs3
        ]])
        pred[i] = np.clip(float(gam.predict(X)[0]), 0, 1)

    fc_actual = actual[seed_end:last_pos + 1]
    fc_pred   = pred[seed_end:last_pos + 1]

    return {
        "pred":      pred,
        "actual":    actual,
        "first_pos": first_pos,
        "seed_end":  seed_end,
        "last_pos":  last_pos,
        "rmse":      float(np.sqrt(mean_squared_error(fc_actual, fc_pred))),
        "mae":       float(mean_absolute_error(fc_actual, fc_pred)),
    }


# ── FIGURES ───────────────────────────────────────────────────────────────────

def plot_fits(forecasts, results):
    names = list(forecasts.keys())
    n = len(names); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        r    = forecasts[name]
        cat  = KNOWN_CATS.get(name, "Unknown")   # only for plot colour
        col  = CAT_COLOURS.get(cat, "#888")
        t    = np.arange(len(r["actual"]))
        rmse = results[results["trend_name"] == name]["RMSE"].iloc[0]

        fp = r["first_pos"]; se = r["seed_end"]; lp = r["last_pos"]
        ax.axvspan(fp, se, alpha=0.07, color="steelblue")
        ax.axvline(se, color="steelblue", lw=1, linestyle=":", alpha=0.6)
        ax.plot(t,       r["actual"],     color=col,      lw=1.8, alpha=0.6, label="Observed")
        ax.plot(t[fp:se], r["pred"][fp:se], color="black", lw=2, linestyle="--", label="GAM fit (seed)")
        ax.plot(t[se:lp+1], r["pred"][se:lp+1], color="crimson", lw=2, linestyle="--",
                label=f"Forecast RMSE={rmse:.3f}")
        ax.text(0.97, 0.97, cat, transform=ax.transAxes, fontsize=8,
                ha="right", va="top", color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.3)
        ax.set_xlabel("Months", fontsize=8)
        ax.set_ylabel("Normalised interest [0,1]", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "GAM 1 — Generalist model (no category labels)\n"
        "Trained on all 36 trends unlabelled   |   Blue = seed 40%   |   Crimson = forecast 60%",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("gam_output/fig_gam1_fits.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam1_fits.png")


def plot_rmse(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Per trend — colour by known category (display only, not model input)
    cols = [CAT_COLOURS.get(KNOWN_CATS.get(n, "Unknown"), "#888")
            for n in results["trend_name"]]
    ax1.bar(range(len(results)), results["RMSE"], color=cols,
            alpha=0.85, edgecolor="black", lw=0.5, width=0.6)
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(results["trend_name"], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Forecast RMSE (lower = better)")
    ax1.set_title("GAM 1 — per-trend RMSE\n(no category labels used in training)",
                  fontweight="bold")
    for cat, col in CAT_COLOURS.items():
        if cat != "Unknown":
            ax1.bar(0, 0, color=col, label=cat, alpha=0.85)
    ax1.legend(fontsize=9)

    # Per category average (for display)
    results["display_cat"] = results["trend_name"].map(
        lambda n: KNOWN_CATS.get(n, "Unknown"))
    cat_avg = results.groupby("display_cat")["RMSE"].mean().reset_index()
    c2 = [CAT_COLOURS.get(c, "#888") for c in cat_avg["display_cat"]]
    bars = ax2.bar(cat_avg["display_cat"], cat_avg["RMSE"], color=c2,
                   alpha=0.85, edgecolor="black", lw=0.5, width=0.5)
    for bar, v in zip(bars, cat_avg["RMSE"]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Average RMSE")
    ax2.set_ylim(0, max(cat_avg["RMSE"]) * 1.3)
    ax2.set_title("GAM 1 — RMSE by category\n(categories shown for reference only)",
                  fontweight="bold")

    plt.tight_layout()
    plt.savefig("gam_output/fig_gam1_rmse.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam1_rmse.png")


def plot_partial_effects(gam):
    labels = ["t_rel", "month_sin", "month_cos",
              "lag_1", "lag_3", "roll_mean_3", "roll_std_3"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    axes = axes.flatten()
    for i, (ax, label) in enumerate(zip(axes, labels)):
        try:
            XX  = gam.generate_X_grid(term=i)
            pdp = gam.partial_dependence(term=i, X=XX)
            ax.plot(XX[:, i], pdp, color="#457b9d", lw=2)
            ax.axhline(0, color="#999", lw=0.8, linestyle="--")
            ax.set_title(label, fontsize=9, fontweight="bold")
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("Effect on prediction", fontsize=8)
        except Exception:
            ax.text(0.5, 0.5, f"n/a\n{label}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
    axes[-1].set_visible(False)
    fig.suptitle(
        "GAM 1 — partial effects per feature\n"
        "Flat line = no effect   |   Curve = shapes the prediction",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("gam_output/fig_gam1_partial_effects.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam1_partial_effects.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GAM 1 — Generalist (no category labels)")
    print("=" * 60)

    print("\nLoading data...")
    df = add_features(load_data())

    train_df = df[~df["trend_name"].isin(TEST_TRENDS)].copy()
    test_df  = df[ df["trend_name"].isin(TEST_TRENDS)].copy()

    # Trim training data to active windows — stops zeros dominating the fit
    train_trimmed = trim_to_active(train_df)
    print(f"\nTrain: {train_df['trend_name'].nunique()} trends — "
          f"{len(train_df)} rows → {len(train_trimmed)} rows after trimming to active windows")
    print(f"Test:  {sorted(test_df['trend_name'].unique())}")

    print("\nFitting generalist GAM...")
    gam = fit_gam(train_trimmed[FEATURE_COLS].values,
                  train_trimmed["value_norm"].values)
    print(f"  Pseudo-R²: {gam.statistics_['pseudo_r2']['McFadden']:.3f}")

    print("\nRollout forecast (seed 40% of active window → forecast 60%)...")
    records = {}; forecasts = {}
    for name, grp in test_df.groupby("trend_name"):
        r = rollout(gam, grp)
        forecasts[name] = r
        records[name] = {
            "trend_name":    name,
            "active_months": r["last_pos"] - r["first_pos"] + 1,
            "n_seed":        r["seed_end"] - r["first_pos"],
            "RMSE":          round(r["rmse"], 4),
            "MAE":           round(r["mae"],  4),
            "model":         "GAM1_generalist",
        }

    results = pd.DataFrame(records.values())
    results["category"] = results["trend_name"].map(
        lambda n: KNOWN_CATS.get(n, "Unknown"))
    results.to_csv("gam_output/gam1_results.csv", index=False)

    print("\nResults:")
    print(results[["trend_name", "category", "active_months", "n_seed", "RMSE", "MAE"]].to_string(index=False))
    print("\nAverage RMSE by category (reference only):")
    print(results.groupby("category")[["RMSE", "MAE"]].mean().round(3).to_string())
    print("\nSaved: gam_output/gam1_results.csv")

    print("\nGenerating figures...")
    plot_fits(forecasts, results)
    plot_rmse(results)
    plot_partial_effects(gam)

    print("\n" + "=" * 60)
    print("GAM 1 complete. Run gam2_specialist.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()