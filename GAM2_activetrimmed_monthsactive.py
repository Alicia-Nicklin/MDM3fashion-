"""
GAM2_activetrimmed.py
=====================
GAM 2 — Three specialist models, each tested blindly on all 9 test trends.

Three separate GAMs are trained:
  Micro specialist — trained only on Micro training trends
  Macro specialist — trained only on Macro training trends
  Mega specialist  — trained only on Mega training trends

Each specialist is then run on ALL 9 test trends without knowing the category.
The proof is that each specialist achieves its lowest RMSE on its own category's
3 test trends — showing that category-specific learning has captured real
differences in lifecycle shape.

Active window trimming (0.35 threshold) applied consistently with
myValadation.py and trend_classification_final.py.

Required:
    data/all_trends_with_classes.csv
    gam_output/gam1_results.csv  (run GAM1_generalised.py first)

Outputs (gam_output/):
    fig_micro_specialist_all9.png   — Micro GAM forecast on all 9 test trends
    fig_macro_specialist_all9.png   — Macro GAM forecast on all 9 test trends
    fig_mega_specialist_all9.png    — Mega GAM forecast on all 9 test trends
    fig_gam2_heatmap.png            — RMSE heatmap proving each specialist wins
    fig_gam1_vs_gam2_trimmed.png    — comparison with GAM 1
    gam2_monthsactive_results.csv
    gam2_monthsactive_all_specialists_results.csv
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

ALL_TRENDS_CSV = os.path.join("data", "all_trends_with_classes.csv")

TEST_TRENDS = {
    "Micro": ["teddycoat", "vsco", "coastalgrandmother"],
    "Macro": ["Darkacademia", "chokers", "Normcore"],
    "Mega":  ["hipster fashion", "TUMBLR", "Pastelgrunge"],
}

ROLLOUT_FRAC  = 0.4
ACTIVE_BUFFER = 3
THRESHOLD     = 0.35  # consistent with myValadation.py

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CAT_COLOURS  = {"Micro": "#E24B4A", "Macro": "#457b9d", "Mega": "#2d6a4f"}
LS_MAP       = {"Micro": "--", "Macro": "-.", "Mega": ":"}
ALL_TEST     = [t for names in TEST_TRENDS.values() for t in names]
# True category for each test trend — only used for plot labels, never passed to model
TRUE_CATS    = {t: cat for cat, trends in TEST_TRENDS.items() for t in trends}
FEATURE_COLS = ["months_active", "month_sin", "month_cos",
                "lag_1", "lag_3", "roll_mean_3", "roll_std_3"]


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
        active = grp[grp["value_norm"] >= THRESHOLD]["date"]
        if len(active) >= 1:
            first_active = active.iloc[0]
        else:
            first_active = grp["date"].iloc[0]
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

def fit_specialist(X, y, category):
    n_splines_t = 15 if category == "Micro" else 20
    gam = LinearGAM(
        s(0, n_splines=n_splines_t) +
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
    """Seed on first 40% of active window (>= 0.35), forecast remaining 60%."""
    grp    = trend_df.sort_values("date").reset_index(drop=True).copy()
    actual = grp["value_norm"].values.copy()
    pred   = actual.copy()

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
        X    = np.array([[grp["months_active"].iloc[i], grp["month_sin"].iloc[i],
                          grp["month_cos"].iloc[i], lag1, lag3, rm3, rs3]])
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

def plot_specialist_on_all9(spec_cat, results_by_trend, test_df):
    """
    One figure per specialist — shows that specialist's forecast on all 9 test trends.
    Each subplot labelled with the trend's TRUE category so you can see which
    trends the specialist fits best.
    """
    names = sorted(results_by_trend.keys())
    cols = 3; rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    spec_col = CAT_COLOURS[spec_cat]

    for ax, name in zip(axes, names):
        r        = results_by_trend[name]
        true_cat = TRUE_CATS[name]
        true_col = CAT_COLOURS[true_cat]
        t        = np.arange(len(r["actual"]))
        fp, se, lp = r["first_pos"], r["seed_end"], r["last_pos"]

        # shade seed window
        ax.axvspan(fp, se, alpha=0.07, color="steelblue")
        ax.axvline(se, color="steelblue", lw=1, linestyle=":", alpha=0.6)

        ax.plot(t,          r["actual"],          color=true_col, lw=1.8, alpha=0.6, label="Observed")
        ax.plot(t[fp:se],   r["pred"][fp:se],     color="black",  lw=2,   linestyle="--", label="Seed fit")
        ax.plot(t[se:lp+1], r["pred"][se:lp+1],  color=spec_col, lw=2,   linestyle="--",
                label=f"Forecast RMSE={r['rmse']:.3f}")

        # label true category — highlights whether this is the specialist's own type
        match = "✓ own type" if true_cat == spec_cat else f"✗ {true_cat}"
        match_col = "#1D9E75" if true_cat == spec_cat else "#888"
        ax.text(0.97, 0.97, match, transform=ax.transAxes, fontsize=8,
                ha="right", va="top", color=match_col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.3)
        ax.set_xlabel("Months", fontsize=8)
        ax.set_ylabel("Normalised interest", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")

    fig.suptitle(
        f"{spec_cat} specialist — blind forecast on all 9 test trends\n"
        f"✓ = own category   ✗ = different category (should score higher RMSE)",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"gam_output/fig_{spec_cat.lower()}_specialist_all9_monthsactive.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_heatmap(all_results_df):
    """
    RMSE heatmap — rows=test trends, cols=specialist used.
    Gold border = true category. Proves correct specialist wins on its own type.
    """
    pivot = all_results_df.pivot(index="trend_name", columns="specialist", values="RMSE")
    true_cats = all_results_df.groupby("trend_name")["true_category"].first()
    pivot["true_cat"]        = true_cats
    pivot["best_specialist"] = pivot[["Micro", "Macro", "Mega"]].idxmin(axis=1)
    pivot["correct"]         = pivot["true_cat"] == pivot["best_specialist"]
    n_correct = pivot["correct"].sum()

    # sort rows: Micro test trends first, then Macro, then Mega
    order = TEST_TRENDS["Micro"] + TEST_TRENDS["Macro"] + TEST_TRENDS["Mega"]
    pivot = pivot.reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── heatmap ──
    heat = pivot[["Micro", "Macro", "Mega"]].values.astype(float)
    im = axes[0].imshow(heat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5)
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["Micro", "Macro", "Mega"], fontsize=11)
    axes[0].set_yticks(range(len(pivot)))
    axes[0].set_yticklabels(pivot.index, fontsize=8)
    axes[0].set_xlabel("Specialist model used", fontsize=10)
    axes[0].set_ylabel("Test trend (true category)", fontsize=10)
    axes[0].set_title(
        "RMSE: each specialist on all 9 test trends\n"
        "Gold border = true category   Bold = best specialist",
        fontweight="bold")
    plt.colorbar(im, ax=axes[0], label="RMSE", shrink=0.8)

    # dividers between category groups
    for y in [2.5, 5.5]:
        axes[0].axhline(y, color="white", lw=2)

    for i, trend in enumerate(pivot.index):
        true_cat = pivot.loc[trend, "true_cat"]
        for j, spec in enumerate(["Micro", "Macro", "Mega"]):
            v       = pivot.loc[trend, spec]
            is_best = spec == pivot.loc[trend, "best_specialist"]
            axes[0].text(j, i, f"{v:.3f}", ha="center", va="center",
                         fontsize=8, fontweight="bold" if is_best else "normal",
                         color="white" if v > 0.35 else "black")
            if spec == true_cat:
                axes[0].add_patch(plt.Rectangle(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    fill=False, edgecolor="gold", lw=2.5))

    # category labels on y axis
    for i, trend in enumerate(pivot.index):
        true_cat = pivot.loc[trend, "true_cat"]
        axes[0].text(-0.6, i, true_cat[0], ha="center", va="center",
                     fontsize=7, color=CAT_COLOURS[true_cat], fontweight="bold")

    # ── correct vs best-wrong bar chart ──
    correct_rmse, best_wrong_rmse, labels, bar_cols = [], [], [], []
    for trend in pivot.index:
        true_cat = pivot.loc[trend, "true_cat"]
        cr  = pivot.loc[trend, true_cat]
        bw  = min(pivot.loc[trend, c] for c in ["Micro", "Macro", "Mega"] if c != true_cat)
        correct_rmse.append(cr)
        best_wrong_rmse.append(bw)
        labels.append(trend)
        bar_cols.append(CAT_COLOURS[true_cat])

    x = np.arange(len(labels)); bw = 0.35
    axes[1].bar(x - bw/2, correct_rmse,    bw, label="Correct specialist",
                color="#1D9E75", alpha=0.85, edgecolor="black", lw=0.5)
    axes[1].bar(x + bw/2, best_wrong_rmse, bw, label="Best wrong specialist",
                color="#E24B4A", alpha=0.85, edgecolor="black", lw=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("RMSE (lower = better)")
    axes[1].set_title(
        "Correct specialist vs best wrong specialist\n"
        "✓ = correct specialist wins   ✗ = wrong specialist wins",
        fontweight="bold")
    axes[1].legend(fontsize=9)

    n_wins = 0
    for xi, (cr, bw_val) in enumerate(zip(correct_rmse, best_wrong_rmse)):
        wins = cr < bw_val
        n_wins += int(wins)
        axes[1].text(xi, max(cr, bw_val) + 0.005, "✓" if wins else "✗",
                     ha="center", fontsize=13,
                     color="#1D9E75" if wins else "#E24B4A", fontweight="bold")

    fig.suptitle(
        f"GAM 2 — Specialist model proof\n"
        f"Correct specialist wins on {n_wins}/9 test trends",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("gam_output/fig_gam2_heatmap_monthsactive.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam2_heatmap.png")
    return n_correct


def plot_comparison_with_gam1(gam2_results):
    g1_path = os.path.join("gam_output", "gam1_results.csv")
    if not os.path.exists(g1_path):
        print("  Skipping comparison — run GAM1_generalised.py first")
        return

    g1 = pd.read_csv(g1_path)
    m  = g1[["trend_name", "category", "RMSE"]].merge(
        gam2_results[["trend_name", "RMSE"]], on="trend_name", suffixes=("_g1", "_g2"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(m)); bw = 0.35

    ax1.bar(x - bw/2, m["RMSE_g1"], bw, label="GAM 1 — generalist",
            color="#534AB7", alpha=0.85, edgecolor="black", lw=0.5)
    ax1.bar(x + bw/2, m["RMSE_g2"], bw, label="GAM 2 — specialist",
            color="#1D9E75", alpha=0.85, edgecolor="black", lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(m["trend_name"], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Forecast RMSE (lower = better)")
    ax1.set_title("GAM 1 vs GAM 2 — per trend", fontweight="bold")
    ax1.legend(fontsize=9)

    for xi, row in zip(x, m.itertuples()):
        imp = row.RMSE_g1 - row.RMSE_g2
        col = "#1D9E75" if imp > 0 else "#E24B4A"
        ax1.text(xi, max(row.RMSE_g1, row.RMSE_g2) + 0.004,
                 f"{'▼' if imp > 0 else '▲'}{abs(imp):.3f}",
                 ha="center", fontsize=7.5, color=col, fontweight="bold")

    g1_cat = g1.groupby("category")["RMSE"].mean().reset_index()
    g2_cat = gam2_results.groupby("category")["RMSE"].mean().reset_index()
    mc = g1_cat.merge(g2_cat, on="category", suffixes=("_g1", "_g2"))
    x2 = np.arange(len(mc))
    ax2.bar(x2 - bw/2, mc["RMSE_g1"], bw, label="GAM 1 — generalist",
            color="#534AB7", alpha=0.85, edgecolor="black", lw=0.5)
    ax2.bar(x2 + bw/2, mc["RMSE_g2"], bw, label="GAM 2 — specialist",
            color="#1D9E75", alpha=0.85, edgecolor="black", lw=0.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(mc["category"], fontsize=11)
    ax2.set_ylabel("Average RMSE")
    ax2.set_title("GAM 1 vs GAM 2 — by category", fontweight="bold")
    ax2.legend(fontsize=9)

    for xi, row in zip(x2, mc.itertuples()):
        imp = row.RMSE_g1 - row.RMSE_g2
        col = "#1D9E75" if imp > 0 else "#E24B4A"
        ax2.text(xi, max(row.RMSE_g1, row.RMSE_g2) + 0.004,
                 f"{'▼' if imp > 0 else '▲'}{abs(imp):.3f}",
                 ha="center", fontsize=10, color=col, fontweight="bold")

    plt.tight_layout()
    plt.savefig("gam_output/fig_gam1_vs_gam2_trimmed_monthsactive.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam1_vs_gam2_trimmed.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GAM 2 — Three specialists, each tested on all 9 trends")
    print("=" * 60)

    print("\nLoading data...")
    df = add_features(load_data())

    train_df = df[~df["trend_name"].isin(ALL_TEST)].copy()
    test_df  = df[ df["trend_name"].isin(ALL_TEST)].copy()

    # ── train one specialist per category ──
    print()
    specialists = {}
    for cat in ["Micro", "Macro", "Mega"]:
        cat_train   = train_df[train_df["category"] == cat]
        cat_trimmed = trim_to_active(cat_train)
        print(f"Fitting {cat} specialist on {cat_train['trend_name'].nunique()} trends — "
              f"{len(cat_train)} rows → {len(cat_trimmed)} after trimming...")
        gam = fit_specialist(cat_trimmed[FEATURE_COLS].values,
                             cat_trimmed["value_norm"].values, cat)
        print(f"  Pseudo-R²: {gam.statistics_['pseudo_r2']['McFadden']:.3f}")
        specialists[cat] = gam

    # ── run every specialist on every test trend (blindly) ──
    print("\nRunning all specialists on all 9 test trends (blind)...")

    # all_preds[spec_cat][trend_name] = rollout result
    all_preds = {cat: {} for cat in ["Micro", "Macro", "Mega"]}
    rows = []

    for spec_cat, gam in specialists.items():
        for name, grp in test_df.groupby("trend_name"):
            r = rollout(gam, grp)
            all_preds[spec_cat][name] = r
            rows.append({
                "trend_name":         name,
                "true_category":      TRUE_CATS[name],
                "specialist":         spec_cat,
                "RMSE":               round(r["rmse"], 4),
                "MAE":                round(r["mae"],  4),
                "correct_specialist": spec_cat == TRUE_CATS[name],
            })

    all_results = pd.DataFrame(rows)

    # matched specialist results (correct specialist on its own trends)
    matched_records = []
    for name in ALL_TEST:
        true_cat  = TRUE_CATS[name]
        matched_r = all_preds[true_cat][name]
        matched_records.append({
            "trend_name":    name,
            "category":      true_cat,
            "active_months": matched_r["last_pos"] - matched_r["first_pos"] + 1,
            "n_seed":        matched_r["seed_end"] - matched_r["first_pos"],
            "RMSE":          round(matched_r["rmse"], 4),
            "MAE":           round(matched_r["mae"],  4),
            "model":         "GAM2_activetrimmed",
        })
    gam2_results = pd.DataFrame(matched_records)

    all_results.to_csv("gam_output/gam2_monthsactive_all_specialists_results.csv", index=False)
    gam2_results.to_csv("gam_output/gam2_monthsactive_results.csv", index=False)

    # ── print summary ──
    pivot = all_results.pivot(index="trend_name", columns="specialist", values="RMSE")
    true_cats = all_results.groupby("trend_name")["true_category"].first()
    pivot["true_cat"]        = true_cats
    pivot["best_specialist"] = pivot[["Micro", "Macro", "Mega"]].idxmin(axis=1)
    pivot["correct"]         = pivot["true_cat"] == pivot["best_specialist"]

    print(f"\nAll specialists on all 9 test trends:")
    print(pivot[["Micro", "Macro", "Mega", "true_cat", "best_specialist", "correct"]].to_string())
    print(f"\nCorrect specialist wins: {pivot['correct'].sum()}/9")

    print("\nMatched specialist results (correct specialist on own category):")
    print(gam2_results[["trend_name", "category", "active_months", "n_seed", "RMSE", "MAE"]].to_string(index=False))
    print("\nPer-category average RMSE:")
    print(gam2_results.groupby("category")[["RMSE", "MAE"]].mean().round(3).to_string())

    # ── figures ──
    print("\nGenerating figures...")
    for spec_cat in ["Micro", "Macro", "Mega"]:
        plot_specialist_on_all9(spec_cat, all_preds[spec_cat], test_df)

    plot_heatmap(all_results)
    plot_comparison_with_gam1(gam2_results)

    print("\n" + "=" * 60)
    print(f"GAM 2 complete. Correct specialist wins: {pivot['correct'].sum()}/9")
    print("=" * 60)


if __name__ == "__main__":
    main()
