"""
GAM3_hierarchical.py
====================
GAM 3 — Single hierarchical model with category-specific lifecycle shapes.

Instead of one blind model (GAM1) or three data-starved separate models (GAM2),
this trains ONE GAM on all 36 training trends simultaneously with:

  Shared terms (learned from all 36 trends):
    - month_sin / month_cos  — universal seasonality
    - lag_1 / lag_3          — universal momentum
    - roll_mean_3            — universal stability
    - roll_std_3             — universal volatility
    - category intercept     — baseline level per category

  Category-specific terms (each learned from its own trends, but within one model):
    - t_rel_micro  — Micro lifecycle shape  (non-zero only for Micro trends)
    - t_rel_macro  — Macro lifecycle shape  (non-zero only for Macro trends)
    - t_rel_mega   — Mega lifecycle shape   (non-zero only for Mega trends)

This solves the core problem of GAM2: instead of 12 trends per specialist,
all 36 trends inform the shared parameters while category shape is still
learned separately. At test time the known category determines which
t_rel column is populated.

Required:
    data/all_trends_with_classes.csv
    gam_output/gam1_results.csv       (run GAM1_generalised.py first)
    gam_output/gam2_trimmed_results.csv (run GAM2_activetrimmed.py first)

Outputs (gam_output/):
    gam3_results.csv
    fig_gam3_fits.png
    fig_gam3_partial_effects.png
    fig_gam1_gam2_gam3_comparison.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f, l
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

ALL_TRENDS_CSV = os.path.join("data", "all_trends_with_classes.csv")

TEST_TRENDS = {
    "Micro": ["teddycoat", "vsco", "coastalgrandmother"],
    "Macro": ["Darkacademia", "chokers", "Normcore"],
    "Mega":  ["hipster fashion", "TUMBLR", "Pastelgrunge"],
}

ROLLOUT_FRAC  = 0.4
ACTIVE_BUFFER = 3
THRESHOLD     = 0.35

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CAT_COLOURS  = {"Micro": "#E24B4A", "Macro": "#457b9d", "Mega": "#2d6a4f"}
ALL_TEST     = [t for names in TEST_TRENDS.values() for t in names]
TRUE_CATS    = {t: cat for cat, trends in TEST_TRENDS.items() for t in trends}
CAT_ENCODE   = {"Micro": 0, "Macro": 1, "Mega": 2}

# Feature indices:
#   0: t_rel_micro   1: t_rel_macro   2: t_rel_mega
#   3: month_sin     4: month_cos
#   5: lag_1         6: lag_3
#   7: roll_mean_3   8: roll_std_3
#   9: category_enc
FEATURE_COLS = ["t_rel_micro", "t_rel_macro", "t_rel_mega",
                "month_sin", "month_cos",
                "lag_1", "lag_3", "roll_mean_3", "roll_std_3",
                "category_enc"]


# ── DATA ──────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(ALL_TRENDS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["category"] = df["trend_class"]
    print(f"  {df['trend_name'].nunique()} trends loaded")
    print(f"  Categories: {df['category'].value_counts().to_dict()}")
    return df


def add_features(df):
    df = df.copy()
    df["value_norm"] = df["value_norm"].fillna(0).clip(0, 1)
    parts = []
    for _, grp in df.groupby("trend_name", sort=False):
        grp = grp.sort_values("date").copy()
        cat = grp["category"].iloc[0]

        # active window for t_rel
        active = grp[grp["value_norm"] >= THRESHOLD]["date"]
        if len(active) >= 2:
            start = active.iloc[0]
            dur   = max((active.iloc[-1] - start).days / 30.44, 1.0)
        else:
            start = grp["date"].iloc[0]
            dur   = 1.0

        t_rel = (((grp["date"] - start).dt.days / 30.44) / dur).clip(-0.5, 2.5)

        # category-specific t_rel columns — zero for other categories
        grp["t_rel_micro"] = t_rel if cat == "Micro" else 0.0
        grp["t_rel_macro"] = t_rel if cat == "Macro" else 0.0
        grp["t_rel_mega"]  = t_rel if cat == "Mega"  else 0.0

        m = grp["date"].dt.month
        grp["month_sin"]   = np.sin(2 * np.pi * m / 12)
        grp["month_cos"]   = np.cos(2 * np.pi * m / 12)
        grp["lag_1"]       = grp["value_norm"].shift(1).fillna(0)
        grp["lag_3"]       = grp["value_norm"].shift(3).fillna(0)
        grp["roll_mean_3"] = grp["value_norm"].shift(1).rolling(3, min_periods=1).mean().fillna(0)
        grp["roll_std_3"]  = grp["value_norm"].shift(1).rolling(3, min_periods=1).std().fillna(0)
        grp["category_enc"] = CAT_ENCODE[cat]
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def trim_to_active(df):
    parts = []
    for _, grp in df.groupby("trend_name", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True).copy()
        active_idx = grp[grp["value_norm"] >= THRESHOLD].index.tolist()
        if len(active_idx) == 0:
            continue
        lo = max(0, active_idx[0] - ACTIVE_BUFFER)
        hi = min(len(grp) - 1, active_idx[-1] + ACTIVE_BUFFER)
        parts.append(grp.iloc[lo:hi + 1])
    return pd.concat(parts, ignore_index=True)


# ── MODEL ─────────────────────────────────────────────────────────────────────

def fit_hierarchical_gam(X, y):
    """
    Single GAM trained on all trends.

    Feature indices:
      0  t_rel_micro   category-specific lifecycle shape (Micro)
      1  t_rel_macro   category-specific lifecycle shape (Macro)
      2  t_rel_mega    category-specific lifecycle shape (Mega)
      3  month_sin     shared seasonality
      4  month_cos     shared seasonality
      5  lag_1         shared momentum
      6  lag_3         shared momentum
      7  roll_mean_3   shared stability
      8  roll_std_3    shared volatility (linear)
      9  category_enc  category intercept shift (factor)
    """
    gam = LinearGAM(
        s(0, n_splines=20) +   # Micro lifecycle shape
        s(1, n_splines=20) +   # Macro lifecycle shape
        s(2, n_splines=20) +   # Mega lifecycle shape
        s(3, n_splines=8)  +   # shared seasonality
        s(4, n_splines=8)  +   # shared seasonality
        s(5, n_splines=12) +   # shared lag_1
        s(6, n_splines=10) +   # shared lag_3
        s(7, n_splines=8)  +   # shared roll_mean_3
        l(8)               +   # shared roll_std_3
        f(9)                   # category intercept
    )
    gam.gridsearch(X, y, progress=False)
    return gam


# ── EVALUATION ────────────────────────────────────────────────────────────────

def rollout(gam, trend_df):
    """Seed on first 40% of active window, forecast remaining 60%."""
    grp    = trend_df.sort_values("date").reset_index(drop=True).copy()
    actual = grp["value_norm"].values.copy()
    pred   = actual.copy()
    cat    = grp["category"].iloc[0]
    cat_enc = CAT_ENCODE[cat]

    active_idx = grp[grp["value_norm"] >= THRESHOLD].index.tolist()
    if len(active_idx) >= 2:
        first_pos, last_pos = active_idx[0], active_idx[-1]
    else:
        first_pos, last_pos = 0, len(grp) - 1

    seed_end = first_pos + max(2, int((last_pos - first_pos + 1) * ROLLOUT_FRAC))

    for i in range(seed_end, last_pos + 1):
        lag1 = pred[i-1] if i >= 1 else 0
        lag3 = pred[i-3] if i >= 3 else 0
        w    = pred[max(0, i-3):i]
        rm3  = float(np.mean(w)) if len(w) > 0 else 0
        rs3  = float(np.std(w))  if len(w) > 1 else 0

        t_rel_val = float(grp["t_rel_micro"].iloc[i] or
                          grp["t_rel_macro"].iloc[i] or
                          grp["t_rel_mega"].iloc[i])

        X = np.array([[
            t_rel_val if cat == "Micro" else 0.0,
            t_rel_val if cat == "Macro" else 0.0,
            t_rel_val if cat == "Mega"  else 0.0,
            grp["month_sin"].iloc[i], grp["month_cos"].iloc[i],
            lag1, lag3, rm3, rs3, cat_enc
        ]])
        pred[i] = np.clip(float(gam.predict(X)[0]), 0, 1)

    fc_actual = actual[seed_end:last_pos + 1]
    fc_pred   = pred[seed_end:last_pos + 1]
    return {
        "pred": pred, "actual": actual,
        "first_pos": first_pos, "seed_end": seed_end, "last_pos": last_pos,
        "rmse": float(np.sqrt(mean_squared_error(fc_actual, fc_pred))),
        "mae":  float(mean_absolute_error(fc_actual, fc_pred)),
    }


# ── FIGURES ───────────────────────────────────────────────────────────────────

def plot_fits(forecasts, test_df, results):
    names = sorted(forecasts.keys())
    cols = 3; rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        r    = forecasts[name]
        cat  = test_df[test_df["trend_name"] == name]["category"].iloc[0]
        col  = CAT_COLOURS[cat]
        t    = np.arange(len(r["actual"]))
        fp, se, lp = r["first_pos"], r["seed_end"], r["last_pos"]
        rmse = results[results["trend_name"] == name]["RMSE"].iloc[0]

        ax.axvspan(fp, se, alpha=0.07, color="steelblue")
        ax.axvline(se, color="steelblue", lw=1, linestyle=":", alpha=0.6)
        ax.plot(t,          r["actual"],         color=col,      lw=1.8, alpha=0.6, label="Observed")
        ax.plot(t[fp:se],   r["pred"][fp:se],    color="black",  lw=2,   linestyle="--", label="Seed fit")
        ax.plot(t[se:lp+1], r["pred"][se:lp+1], color="crimson", lw=2,  linestyle="--",
                label=f"Forecast RMSE={rmse:.3f}")
        ax.text(0.97, 0.97, cat, transform=ax.transAxes, fontsize=8,
                ha="right", va="top", color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.3)
        ax.set_xlabel("Months", fontsize=8)
        ax.set_ylabel("Normalised interest", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")

    fig.suptitle(
        "GAM 3 — Hierarchical model\n"
        "One model, category-specific lifecycle shapes + shared seasonality/momentum",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("gam_output/fig_gam3_fits.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam3_fits.png")


def plot_partial_effects(gam):
    labels = ["t_rel (Micro)", "t_rel (Macro)", "t_rel (Mega)",
              "month_sin", "month_cos", "lag_1", "lag_3",
              "roll_mean_3", "roll_std_3"]
    cols = ["#E24B4A", "#457b9d", "#2d6a4f",
            "#888", "#888", "#555", "#555", "#555", "#555"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (ax, label, col) in enumerate(zip(axes, labels, cols)):
        try:
            XX  = gam.generate_X_grid(term=i)
            pdp = gam.partial_dependence(term=i, X=XX)
            ax.plot(XX[:, i], pdp, color=col, lw=2)
            ax.axhline(0, color="#999", lw=0.8, linestyle="--")
            ax.set_title(label, fontsize=9, fontweight="bold", color=col)
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("Effect", fontsize=8)
        except Exception:
            ax.text(0.5, 0.5, f"n/a\n{label}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

    fig.suptitle(
        "GAM 3 — Partial effects\n"
        "Top row: category-specific lifecycle shapes   |   Remaining: shared terms",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("gam_output/fig_gam3_partial_effects.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam3_partial_effects.png")


def plot_three_way_comparison(gam3_results):
    """Compare GAM1, GAM2 (trimmed), and GAM3 side by side."""
    g1_path = "gam_output/gam1_results.csv"
    g2_path = "gam_output/gam2_trimmed_results.csv"
    if not os.path.exists(g1_path) or not os.path.exists(g2_path):
        print("  Skipping comparison — run GAM1 and GAM2_activetrimmed first")
        return

    g1 = pd.read_csv(g1_path)[["trend_name", "category", "RMSE"]].rename(columns={"RMSE": "GAM1"})
    g2 = pd.read_csv(g2_path)[["trend_name", "RMSE"]].rename(columns={"RMSE": "GAM2"})
    g3 = gam3_results[["trend_name", "RMSE"]].rename(columns={"RMSE": "GAM3"})

    m = g1.merge(g2, on="trend_name").merge(g3, on="trend_name")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(m)); bw = 0.25

    for offset, col_name, colour, label in [
        (-bw, "GAM1", "#534AB7", "GAM 1 — generalist"),
        (  0, "GAM2", "#457b9d", "GAM 2 — specialist"),
        ( bw, "GAM3", "#E24B4A", "GAM 3 — hierarchical"),
    ]:
        ax1.bar(x + offset, m[col_name], bw, label=label,
                color=colour, alpha=0.85, edgecolor="black", lw=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(m["trend_name"], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Forecast RMSE (lower = better)")
    ax1.set_title("GAM 1 vs GAM 2 vs GAM 3 — per trend", fontweight="bold")
    ax1.legend(fontsize=9)

    cat_avg = m.groupby("category")[["GAM1", "GAM2", "GAM3"]].mean().reset_index()
    x2 = np.arange(len(cat_avg))
    for offset, col_name, colour, label in [
        (-bw, "GAM1", "#534AB7", "GAM 1"),
        (  0, "GAM2", "#457b9d", "GAM 2"),
        ( bw, "GAM3", "#E24B4A", "GAM 3"),
    ]:
        bars = ax2.bar(x2 + offset, cat_avg[col_name], bw, label=label,
                       color=colour, alpha=0.85, edgecolor="black", lw=0.5)
        for bar, v in zip(bars, cat_avg[col_name]):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.003, f"{v:.3f}",
                     ha="center", fontsize=7, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cat_avg["category"], fontsize=11)
    ax2.set_ylabel("Average RMSE")
    ax2.set_title("GAM 1 vs GAM 2 vs GAM 3 — by category", fontweight="bold")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("gam_output/fig_gam1_gam2_gam3_comparison.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam1_gam2_gam3_comparison.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GAM 3 — Hierarchical model")
    print("=" * 60)

    print("\nLoading data...")
    df = add_features(load_data())

    train_df = df[~df["trend_name"].isin(ALL_TEST)].copy()
    test_df  = df[ df["trend_name"].isin(ALL_TEST)].copy()

    train_trimmed = trim_to_active(train_df)
    print(f"\nTrain: {train_df['trend_name'].nunique()} trends — "
          f"{len(train_df)} rows → {len(train_trimmed)} after trimming")
    print(f"Test:  {sorted(test_df['trend_name'].unique())}")

    print("\nFitting hierarchical GAM (one model, all 36 trends)...")
    gam = fit_hierarchical_gam(train_trimmed[FEATURE_COLS].values,
                               train_trimmed["value_norm"].values)
    print(f"  Pseudo-R²: {gam.statistics_['pseudo_r2']['McFadden']:.3f}")

    print("\nRollout forecast (seed 40% of active window → forecast 60%)...")
    records = {}; forecasts = {}
    for name, grp in test_df.groupby("trend_name"):
        r = rollout(gam, grp)
        forecasts[name] = r
        true_cat = TRUE_CATS[name]
        records[name] = {
            "trend_name":    name,
            "category":      true_cat,
            "active_months": r["last_pos"] - r["first_pos"] + 1,
            "n_seed":        r["seed_end"] - r["first_pos"],
            "RMSE":          round(r["rmse"], 4),
            "MAE":           round(r["mae"],  4),
            "model":         "GAM3_hierarchical",
        }

    results = pd.DataFrame(records.values())
    results.to_csv("gam_output/gam3_results.csv", index=False)

    print("\nResults:")
    print(results[["trend_name", "category", "active_months", "n_seed", "RMSE", "MAE"]].to_string(index=False))
    print("\nPer-category average:")
    print(results.groupby("category")[["RMSE", "MAE"]].mean().round(3).to_string())

    print("\nGenerating figures...")
    plot_fits(forecasts, test_df, results)
    plot_partial_effects(gam)
    plot_three_way_comparison(results)

    print("\n" + "=" * 60)
    print("GAM 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
