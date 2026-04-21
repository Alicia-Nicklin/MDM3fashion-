"""
GAM1_trialtrel.py
=================
GAM 1 — Generalist model, no leakage version.

Replaces t_rel (which requires knowing the trend end date in advance) with
months_since_active_start — the number of months since the trend first crossed
the 0.35 activity threshold. This is fully observable at any forecast step:
you only need to know when the trend became active, not when it ends.

Key differences from GAM1_notrimmed.py:
  - Feature: months_active instead of t_rel
  - No dependency on trend_validation_results.csv
  - Seed/evaluate on active window only (not full 268-month series)
  - RMSE measured on active window forecast portion only

Outputs (written to gam_output/):
    gam1_trialtrel_results.csv
    fig_gam1_trialtrel_fits.png
    fig_gam1_trialtrel_rmse.png
    fig_gam1_trialtrel_partial_effects.png
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

ALL_TRENDS_CSV = os.path.join("data", "all_trends_combined.csv")

THRESHOLD     = 0.35
ROLLOUT_FRAC  = 0.4

TEST_TRENDS = [
    "teddycoat", "vsco", "coastalgrandmother",
    "Darkacademia", "chokers", "Normcore",
    "hipster fashion", "TUMBLR", "Pastelgrunge",
]

KNOWN_CATS = {
    "teddycoat": "Micro", "vsco": "Micro", "coastalgrandmother": "Micro",
    "Darkacademia": "Macro", "chokers": "Macro", "Normcore": "Macro",
    "hipster fashion": "Mega", "TUMBLR": "Mega", "Pastelgrunge": "Mega",
}
CAT_COLOURS = {"Micro": "#E24B4A", "Macro": "#457b9d", "Mega": "#2d6a4f",
               "Unknown": "#888780"}

FEATURE_COLS = ["months_active", "month_sin", "month_cos",
                "lag_1", "lag_3", "roll_mean_3", "roll_std_3"]


# ── DATA ──────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(ALL_TRENDS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["value_norm"] = df["value_norm"].fillna(0).clip(0, 1)
    print(f"  {df['trend_name'].nunique()} trends loaded")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def add_features(df):
    """
    months_active: months elapsed since the trend first crossed THRESHOLD.
                   Negative before the trend becomes active.
                   Does NOT require knowing when the trend ends — no leakage.
    """
    df = df.copy()
    parts = []
    for _, grp in df.groupby("trend_name", sort=False):
        grp = grp.sort_values("date").copy()

        active_rows = grp[grp["value_norm"] >= THRESHOLD]
        if len(active_rows) == 0:
            first_active = grp["date"].iloc[0]
        else:
            first_active = active_rows["date"].iloc[0]

        grp["months_active"] = (grp["date"] - first_active).dt.days / 30.44

        m = grp["date"].dt.month
        grp["month_sin"]   = np.sin(2 * np.pi * m / 12)
        grp["month_cos"]   = np.cos(2 * np.pi * m / 12)
        grp["lag_1"]       = grp["value_norm"].shift(1).fillna(0)
        grp["lag_3"]       = grp["value_norm"].shift(3).fillna(0)
        grp["roll_mean_3"] = grp["value_norm"].shift(1).rolling(3, min_periods=1).mean().fillna(0)
        grp["roll_std_3"]  = grp["value_norm"].shift(1).rolling(3, min_periods=1).std().fillna(0)
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


# ── MODEL ─────────────────────────────────────────────────────────────────────

def fit_gam(X, y):
    """
    Feature index:
      0  months_active  smooth, 20 splines
      1  month_sin      smooth,  8 splines
      2  month_cos      smooth,  8 splines
      3  lag_1          smooth, 12 splines
      4  lag_3          smooth, 10 splines
      5  roll_mean_3    smooth,  8 splines
      6  roll_std_3     linear
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
    Seed on first ROLLOUT_FRAC of the active window.
    Forecast remaining 60% of active window autoregressively.
    RMSE evaluated on forecast portion of active window only.
    """
    grp    = trend_df.sort_values("date").reset_index(drop=True).copy()
    actual = grp["value_norm"].values.copy()

    active_idx = grp[grp["value_norm"] >= THRESHOLD].index
    if len(active_idx) < 2:
        first_pos, last_pos = 0, len(grp) - 1
    else:
        first_pos = int(active_idx[0])
        last_pos  = int(active_idx[-1])

    n_active = last_pos - first_pos + 1
    seed_end = first_pos + max(1, int(np.floor(n_active * ROLLOUT_FRAC)))

    pred = actual.copy()
    for i in range(seed_end, last_pos + 1):
        lag1 = pred[i-1] if i >= 1 else 0
        lag3 = pred[i-3] if i >= 3 else 0
        w    = pred[max(0, i-3):i]
        rm3  = float(np.mean(w)) if len(w) > 0 else 0
        rs3  = float(np.std(w))  if len(w) > 1 else 0
        X = np.array([[
            grp["months_active"].iloc[i], grp["month_sin"].iloc[i],
            grp["month_cos"].iloc[i], lag1, lag3, rm3, rs3
        ]])
        pred[i] = np.clip(float(gam.predict(X)[0]), 0, 1)

    forecast_actual = actual[seed_end:last_pos + 1]
    forecast_pred   = pred[seed_end:last_pos + 1]
    rmse = float(np.sqrt(mean_squared_error(forecast_actual, forecast_pred))) \
           if len(forecast_actual) > 0 else np.nan
    mae  = float(mean_absolute_error(forecast_actual, forecast_pred)) \
           if len(forecast_actual) > 0 else np.nan

    return {
        "pred":      pred,
        "actual":    actual,
        "first_pos": first_pos,
        "seed_end":  seed_end,
        "last_pos":  last_pos,
        "n_active":  n_active,
        "n_seed":    seed_end - first_pos,
        "rmse":      rmse,
        "mae":       mae,
    }


# ── FIGURES ───────────────────────────────────────────────────────────────────

def plot_fits(forecasts, results):
    names = list(forecasts.keys())
    cols = 3; rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        r   = forecasts[name]
        cat = KNOWN_CATS.get(name, "Unknown")
        col = CAT_COLOURS.get(cat, "#888")
        t   = np.arange(len(r["actual"]))
        fp  = r["first_pos"]; se = r["seed_end"]; lp = r["last_pos"]
        rmse = results[results["trend_name"] == name]["RMSE"].iloc[0]

        ax.axvspan(fp, se, alpha=0.07, color="steelblue")
        ax.axvline(se, color="steelblue", lw=1, linestyle=":", alpha=0.6)
        ax.plot(t,        r["actual"],       color=col,      lw=1.8, alpha=0.6, label="Observed")
        ax.plot(t[fp:se], r["pred"][fp:se],  color="black",  lw=2,   linestyle="--", label="GAM fit (seed)")
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

    for ax in axes[len(names):]:
        ax.set_visible(False)

    fig.suptitle(
        "GAM 1 Trial — months_active replaces t_rel (no end-date leakage)\n"
        "Seed 40% active window   |   Crimson = forecast 60%",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("gam_output/fig_gam1_trialtrel_fits.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam1_trialtrel_fits.png")


def plot_rmse(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cols = [CAT_COLOURS.get(KNOWN_CATS.get(n, "Unknown"), "#888")
            for n in results["trend_name"]]
    ax1.bar(range(len(results)), results["RMSE"], color=cols,
            alpha=0.85, edgecolor="black", lw=0.5, width=0.6)
    ax1.set_xticks(range(len(results)))
    ax1.set_xticklabels(results["trend_name"], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Forecast RMSE (lower = better)")
    ax1.set_title("GAM 1 Trial — per-trend RMSE\n(months_active, no end-date leakage)",
                  fontweight="bold")
    for cat, col in CAT_COLOURS.items():
        if cat != "Unknown":
            ax1.bar(0, 0, color=col, label=cat, alpha=0.85)
    ax1.legend(fontsize=9)

    cat_avg = results.groupby("category")["RMSE"].mean().reset_index()
    c2 = [CAT_COLOURS.get(c, "#888") for c in cat_avg["category"]]
    bars = ax2.bar(cat_avg["category"], cat_avg["RMSE"], color=c2,
                   alpha=0.85, edgecolor="black", lw=0.5, width=0.5)
    for bar, v in zip(bars, cat_avg["RMSE"]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Average RMSE")
    ax2.set_ylim(0, max(cat_avg["RMSE"]) * 1.3)
    ax2.set_title("GAM 1 Trial — RMSE by category", fontweight="bold")

    plt.tight_layout()
    plt.savefig("gam_output/fig_gam1_trialtrel_rmse.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam1_trialtrel_rmse.png")


def plot_partial_effects(gam):
    labels = FEATURE_COLS
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
        "GAM 1 Trial — partial effects per feature\n"
        "months_active shows learned lifecycle shape without end-date leakage",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("gam_output/fig_gam1_trialtrel_partial_effects.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam1_trialtrel_partial_effects.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GAM 1 Trial — months_active replaces t_rel (no leakage)")
    print("=" * 60)

    print("\nLoading data...")
    df = add_features(load_data())

    train_df = df[~df["trend_name"].isin(TEST_TRENDS)].copy()
    test_df  = df[ df["trend_name"].isin(TEST_TRENDS)].copy()

    print(f"\nTrain: {train_df['trend_name'].nunique()} trends — {len(train_df)} rows")
    print(f"Test:  {sorted(test_df['trend_name'].unique())}")

    print("\nFitting GAM (months_active, no t_rel)...")
    gam = fit_gam(train_df[FEATURE_COLS].values,
                  train_df["value_norm"].values)
    print(f"  Pseudo-R²: {gam.statistics_['pseudo_r2']['McFadden']:.3f}")

    print("\nRollout forecast (seed 40% of active window → forecast 60%)...")
    records = {}; forecasts = {}
    for name, grp in test_df.groupby("trend_name"):
        r = rollout(gam, grp)
        forecasts[name] = r
        records[name] = {
            "trend_name":    name,
            "category":      KNOWN_CATS.get(name, "Unknown"),
            "active_months": r["n_active"],
            "n_seed":        r["n_seed"],
            "RMSE":          round(r["rmse"], 4),
            "MAE":           round(r["mae"],  4),
        }

    results = pd.DataFrame(records.values())
    results.to_csv("gam_output/gam1_trialtrel_results.csv", index=False)

    print("\nResults:")
    print(results[["trend_name", "category", "active_months", "n_seed", "RMSE", "MAE"]].to_string(index=False))
    print("\nPer-category average:")
    print(results.groupby("category")[["RMSE", "MAE"]].mean().round(3).to_string())

    print("\nGenerating figures...")
    plot_fits(forecasts, results)
    plot_rmse(results)
    plot_partial_effects(gam)

    print("\n" + "=" * 60)
    print("GAM 1 Trial complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
