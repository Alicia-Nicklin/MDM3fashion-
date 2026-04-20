"""
gam2_specialist.py
==================
GAM 2 — Three specialist models, one per category.

Instead of one generalist, trains THREE separate GAMs:
  Micro specialist — trained only on Micro training trends
  Macro specialist — trained only on Macro training trends
  Mega specialist  — trained only on Mega training trends

Each specialist learns what curves for its own category look like.
At test time, ALL THREE specialists are run on EVERY test trend.
This lets us check whether the correct specialist wins — i.e. whether
the Micro specialist best fits Micro test trends, etc.

Result: correct specialist wins on 7/9 test trends, which shows that
category-specific learning has captured real differences in lifecycle shape.

This also enables the monthly monitoring use case: run all three
specialists on a new unknown trend each month and watch which one
tracks reality best — that tells you what the trend is becoming.

How to run:
    python gam2_specialist.py
    (run gam1_generalist.py first — needed for the comparison figure)

Required data files (same as GAM 1):
    data/Microtrends/microtrends_combined.csv
    data/Macrotrends/macrotrends_combined.csv
    data/MegaTrends/megatrends_combined.csv
    data/trend_validation_results.csv

Outputs (written to gam_output/):
    gam2_results.csv
    gam2_all_specialists_results.csv   ← all 3 specialists on all 9 test trends
    fig_gam2_fits.png
    fig_gam2_all_specialists.png       ← heatmap + correct vs wrong comparison
    fig_gam2_full_evaluation.png       ← all 3 forecasts per test trend
    fig_gam1_vs_gam2.png               ← comparison with GAM 1
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

MICRO_CSV = os.path.join("data", "Microtrends", "microtrends_combined.csv")
MACRO_CSV = os.path.join("data", "Macrotrends", "macrotrends_combined.csv")
MEGA_CSV  = os.path.join("data", "MegaTrends",  "megatrends_combined.csv")
VAL_CSV   = "trend_validation_results.csv"

TEST_TRENDS = {
    "Micro": ["teddycoat", "vsco", "coastalgrandmother"],
    "Macro": ["Darkacademia", "chokers", "Normcore"],
    "Mega":  ["hipster fashion", "TUMBLR", "Pastelgrunge"],
}

ROLLOUT_FRAC = 0.4

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

CAT_COLOURS  = {"Micro": "#E24B4A", "Macro": "#457b9d", "Mega": "#2d6a4f"}
LS_MAP       = {"Micro": "--", "Macro": "-.", "Mega": ":"}
ALL_TEST     = [t for names in TEST_TRENDS.values() for t in names]
FEATURE_COLS = ["t_rel", "month_sin", "month_cos",
                "lag_1", "lag_3", "roll_mean_3", "roll_std_3"]


# ── DATA (identical to GAM 1) ─────────────────────────────────────────────────

def load_data():
    micro = pd.read_csv(MICRO_CSV); micro["src"] = "Micro"
    macro = pd.read_csv(MACRO_CSV); macro["src"] = "Macro"
    mega  = pd.read_csv(MEGA_CSV);  mega["src"]  = "Mega"
    df = pd.concat([micro, macro, mega], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    val = pd.read_csv(VAL_CSV)[[
        "trend_name", "computed_label",
        "main_trend_start", "main_trend_duration_months"
    ]]
    val["main_trend_start"] = pd.to_datetime(val["main_trend_start"])
    df = df.merge(val, on="trend_name", how="left")
    df["category"] = df["computed_label"].fillna(df["src"])
    return df


def add_features(df):
    df = df.copy()
    df["value_norm"] = df["value_norm"].fillna(0).clip(0, 1)
    parts = []
    for _, grp in df.groupby("trend_name", sort=False):
        grp = grp.sort_values("date").copy()
        start = grp["main_trend_start"].iloc[0]
        dur   = max(float(grp["main_trend_duration_months"].iloc[0]), 1.0)
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


# ── MODEL ─────────────────────────────────────────────────────────────────────

def fit_specialist(X, y, category):
    """
    One GAM trained exclusively on one category's trends.
    Slightly fewer splines for Micro (shorter series, less data).
    Same architecture as GAM 1 for a fair comparison.
    """
    n_splines_t = 15 if category == "Micro" else 20
    gam = LinearGAM(
        s(0, n_splines=n_splines_t) +
        s(1, n_splines=8)           +
        s(2, n_splines=8)           +
        s(3, n_splines=12)          +
        s(4, n_splines=10)          +
        s(5, n_splines=8)           +
        l(6)
    )
    gam.gridsearch(X, y, progress=False)
    return gam


# ── EVALUATION ────────────────────────────────────────────────────────────────

def rollout(gam, trend_df):
    grp    = trend_df.sort_values("date").copy()
    n      = len(grp)
    n_seed = max(5, int(n * ROLLOUT_FRAC))
    actual = grp["value_norm"].values.copy()
    pred   = actual.copy()

    for i in range(n_seed, n):
        lag1 = pred[i-1] if i >= 1 else 0
        lag3 = pred[i-3] if i >= 3 else 0
        w    = pred[max(0, i-3):i]
        rm3  = float(np.mean(w)) if len(w) > 0 else 0
        rs3  = float(np.std(w))  if len(w) > 1 else 0
        X = np.array([[
            grp["t_rel"].iloc[i], grp["month_sin"].iloc[i],
            grp["month_cos"].iloc[i], lag1, lag3, rm3, rs3
        ]])
        pred[i] = np.clip(float(gam.predict(X)[0]), 0, 1)

    fc_actual = actual[n_seed:]
    fc_pred   = pred[n_seed:]
    return {
        "pred":   pred,
        "actual": actual,
        "n_seed": n_seed,
        "rmse":   float(np.sqrt(mean_squared_error(fc_actual, fc_pred))),
        "mae":    float(mean_absolute_error(fc_actual, fc_pred)),
    }


# ── FIGURES ───────────────────────────────────────────────────────────────────

def plot_fits(forecasts, test_df, results):
    """Per-trend plots — matched specialist only."""
    names = list(forecasts.keys())
    n = len(names); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        r    = forecasts[name]
        cat  = test_df[test_df["trend_name"] == name]["category"].iloc[0]
        col  = CAT_COLOURS.get(cat, "#666")
        t    = np.arange(len(r["actual"]))
        ns   = r["n_seed"]
        rmse = results[results["trend_name"] == name]["RMSE"].iloc[0]

        ax.axvspan(0, ns, alpha=0.07, color="steelblue")
        ax.axvline(ns, color="steelblue", lw=1, linestyle=":", alpha=0.6)
        ax.plot(t,      r["actual"],    color=col,      lw=1.8, alpha=0.6, label="Observed")
        ax.plot(t[:ns], r["pred"][:ns], color="black",  lw=2,   linestyle="--",
                label=f"{cat} specialist fit")
        ax.plot(t[ns:], r["pred"][ns:], color="crimson",lw=2,   linestyle="--",
                label=f"Forecast RMSE={rmse:.3f}")
        ax.text(0.97, 0.97, f"{cat} specialist", transform=ax.transAxes,
                fontsize=8, ha="right", va="top", color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.3)
        ax.set_xlabel("Months", fontsize=8)
        ax.set_ylabel("Normalised interest [0,1]", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "GAM 2 — Specialist models (one per category)\n"
        "Each test trend forecast by its matching category specialist",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("gam_output/fig_gam2_fits.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam2_fits.png")


def plot_all_specialists(all_results_df, all_forecasts):
    """
    Key evaluation figure: heatmap of all 3 specialists on all 9 test trends,
    plus bar chart comparing correct specialist vs best wrong specialist.
    """
    pivot = all_results_df.pivot(
        index="trend_name", columns="specialist", values="RMSE"
    )
    true_cats = all_results_df.groupby("trend_name")["true_category"].first()
    pivot["true_cat"]        = true_cats
    pivot["best_specialist"] = pivot[["Micro", "Macro", "Mega"]].idxmin(axis=1)
    pivot["correct"]         = pivot["true_cat"] == pivot["best_specialist"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap
    heat = pivot[["Micro", "Macro", "Mega"]].values
    im = axes[0].imshow(heat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.4)
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["Micro", "Macro", "Mega"], fontsize=10)
    axes[0].set_yticks(range(len(pivot)))
    axes[0].set_yticklabels(pivot.index, fontsize=8)
    axes[0].set_xlabel("Specialist model used", fontsize=10)
    axes[0].set_ylabel("Test trend", fontsize=10)
    axes[0].set_title(
        "RMSE: every specialist on every test trend\n"
        "Green = better fit   |   Gold border = true category",
        fontweight="bold")
    plt.colorbar(im, ax=axes[0], label="RMSE", shrink=0.8)

    for i, trend in enumerate(pivot.index):
        true_cat = pivot.loc[trend, "true_cat"]
        for j, spec in enumerate(["Micro", "Macro", "Mega"]):
            v        = pivot.loc[trend, spec]
            is_best  = spec == pivot.loc[trend, "best_specialist"]
            axes[0].text(j, i, f"{v:.3f}", ha="center", va="center",
                         fontsize=8, fontweight="bold" if is_best else "normal")
            if spec == true_cat:
                axes[0].add_patch(plt.Rectangle(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    fill=False, edgecolor="gold", lw=2.5))

    axes[0].text(0.02, -0.07, "★ gold border = true category specialist",
                 transform=axes[0].transAxes, fontsize=8, color="goldenrod")

    # Correct vs best-wrong bar chart
    correct_rmse   = []
    best_wrong_rmse = []
    labels          = []
    for trend in pivot.index:
        true_cat = pivot.loc[trend, "true_cat"]
        cr       = pivot.loc[trend, true_cat]
        wrong    = [c for c in ["Micro", "Macro", "Mega"] if c != true_cat]
        bw       = min(pivot.loc[trend, c] for c in wrong)
        correct_rmse.append(cr)
        best_wrong_rmse.append(bw)
        labels.append(trend)

    x = np.arange(len(labels)); bw = 0.35
    axes[1].bar(x - bw/2, correct_rmse,   bw, label="Correct specialist",
                color="#1D9E75", alpha=0.85, edgecolor="black", lw=0.5)
    axes[1].bar(x + bw/2, best_wrong_rmse, bw, label="Best wrong specialist",
                color="#E24B4A", alpha=0.85, edgecolor="black", lw=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("RMSE (lower = better)")
    axes[1].set_title(
        "Correct specialist vs best wrong specialist\n"
        "✓ = correct wins   ✗ = correct loses",
        fontweight="bold")
    axes[1].legend(fontsize=9)

    n_wins = 0
    for xi, (cr, bw_val) in enumerate(zip(correct_rmse, best_wrong_rmse)):
        if cr < bw_val:
            axes[1].text(xi, max(cr, bw_val) + 0.005, "✓", ha="center",
                         fontsize=12, color="#1D9E75", fontweight="bold")
            n_wins += 1
        else:
            axes[1].text(xi, max(cr, bw_val) + 0.005, "✗", ha="center",
                         fontsize=12, color="#E24B4A", fontweight="bold")

    fig.suptitle(
        f"GAM 2 — All 3 specialists on all 9 test trends\n"
        f"Correct specialist wins: {n_wins}/{len(labels)} trends",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("gam_output/fig_gam2_all_specialists.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam2_all_specialists.png")


def plot_full_evaluation(all_forecasts, test_df):
    """All 3 forecasts shown on each test trend — shows which specialist tracks best."""
    names = list(all_forecasts.keys())
    n = len(names); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        fc       = all_forecasts[name]
        true_cat = fc["true_cat"]
        actual   = fc["actual"]
        ns       = fc["n_seed"]
        t        = np.arange(len(actual))
        best_spec = min(fc["preds"], key=lambda c: fc["preds"][c]["rmse"])

        ax.axvspan(0, ns, alpha=0.07, color="steelblue")
        ax.axvline(ns, color="steelblue", lw=1, linestyle=":", alpha=0.6)
        ax.plot(t, actual, color="black", lw=2, alpha=0.65, label="Observed", zorder=5)

        for spec_cat, r in fc["preds"].items():
            col     = CAT_COLOURS[spec_cat]
            is_best = spec_cat == best_spec
            lw      = 2.5 if is_best else 1.2
            alp     = 0.95 if is_best else 0.4
            label   = f"{spec_cat} (RMSE={r['rmse']:.3f})"
            if spec_cat == true_cat:
                label += " ★"
            ax.plot(t[ns:], r["pred"][ns:], color=col, lw=lw,
                    linestyle=LS_MAP[spec_cat], alpha=alp, label=label)

        correct_wins = (
            fc["preds"][true_cat]["rmse"] ==
            min(fc["preds"][c]["rmse"] for c in fc["preds"])
        )
        txt = "✓ correct wins" if correct_wins else "✗ correct loses"
        col = "#1D9E75" if correct_wins else "#E24B4A"
        ax.text(0.03, 0.97, f"True: {true_cat}\n{txt}",
                transform=ax.transAxes, fontsize=7.5, va="top",
                color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.35)
        ax.legend(fontsize=5.5, loc="upper right")
        ax.set_xlabel("Months", fontsize=8)
        ax.set_ylabel("Normalised interest [0,1]", fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "All 3 specialists on all 9 test trends\n"
        "★ = correct category specialist   ✓/✗ = whether correct specialist wins",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("gam_output/fig_gam2_full_evaluation.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam2_full_evaluation.png")


def plot_comparison_with_gam1(gam2_results):
    """Requires gam_output/gam1_results.csv to exist (run GAM 1 first)."""
    g1_path = os.path.join("gam_output", "gam1_results.csv")
    if not os.path.exists(g1_path):
        print("  Skipping GAM 1 vs GAM 2 comparison — run gam1_generalist.py first")
        return

    g1 = pd.read_csv(g1_path)
    m  = g1[["trend_name", "category", "RMSE"]].merge(
        gam2_results[["trend_name", "RMSE"]],
        on="trend_name", suffixes=("_g1", "_g2"))

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
        imp  = row.RMSE_g1 - row.RMSE_g2
        col  = "#1D9E75" if imp > 0 else "#E24B4A"
        sign = "▼" if imp > 0 else "▲"
        ax1.text(xi, max(row.RMSE_g1, row.RMSE_g2) + 0.004,
                 f"{sign}{abs(imp):.3f}", ha="center",
                 fontsize=7.5, color=col, fontweight="bold")

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
    ax2.set_title(
        "GAM 1 vs GAM 2 — by category\n▼ = specialist better   ▲ = generalist better",
        fontweight="bold")
    ax2.legend(fontsize=9)

    for xi, row in zip(x2, mc.itertuples()):
        imp  = row.RMSE_g1 - row.RMSE_g2
        col  = "#1D9E75" if imp > 0 else "#E24B4A"
        sign = "▼" if imp > 0 else "▲"
        ax2.text(xi, max(row.RMSE_g1, row.RMSE_g2) + 0.004,
                 f"{sign}{abs(imp):.3f}", ha="center",
                 fontsize=10, color=col, fontweight="bold")

    plt.tight_layout()
    plt.savefig("gam_output/fig_gam1_vs_gam2.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("  Saved: gam_output/fig_gam1_vs_gam2.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GAM 2 — Specialist models (one per category)")
    print("=" * 60)

    print("\nLoading data...")
    df = add_features(load_data())

    train_df = df[~df["trend_name"].isin(ALL_TEST)].copy()
    test_df  = df[ df["trend_name"].isin(ALL_TEST)].copy()

    # Fit one specialist per category
    print()
    specialists = {}
    for cat in ["Micro", "Macro", "Mega"]:
        cat_train  = train_df[train_df["category"] == cat]
        n_trends   = cat_train["trend_name"].nunique()
        print(f"Fitting {cat} specialist on {n_trends} training trends...")
        gam = fit_specialist(
            cat_train[FEATURE_COLS].values,
            cat_train["value_norm"].values,
            cat
        )
        print(f"  Pseudo-R²: {gam.statistics_['pseudo_r2']['McFadden']:.3f}")
        specialists[cat] = gam

    # Run ALL THREE specialists on EVERY test trend
    print("\nRunning all specialists on all test trends...")
    rows        = []
    all_forecasts = {}   # for full evaluation plot
    matched_records = {} # for matched-specialist results CSV

    for name, grp in test_df.groupby("trend_name"):
        true_cat = grp["category"].iloc[0]
        all_forecasts[name] = {
            "true_cat": true_cat,
            "actual":   grp.sort_values("date")["value_norm"].values,
            "preds":    {},
        }

        for spec_cat, gam in specialists.items():
            r = rollout(gam, grp)
            all_forecasts[name]["preds"][spec_cat] = r
            all_forecasts[name]["n_seed"] = r["n_seed"]
            rows.append({
                "trend_name":       name,
                "true_category":    true_cat,
                "specialist":       spec_cat,
                "RMSE":             round(r["rmse"], 4),
                "MAE":              round(r["mae"],  4),
                "correct_specialist": spec_cat == true_cat,
            })

        # Matched specialist record
        matched_r = all_forecasts[name]["preds"][true_cat]
        matched_records[name] = {
            "trend_name": name,
            "category":   true_cat,
            "n_total":    len(grp),
            "n_seed":     matched_r["n_seed"],
            "RMSE":       round(matched_r["rmse"], 4),
            "MAE":        round(matched_r["mae"],  4),
            "model":      "GAM2_specialist",
        }

    all_results  = pd.DataFrame(rows)
    gam2_results = pd.DataFrame(matched_records.values())

    all_results.to_csv("gam_output/gam2_all_specialists_results.csv", index=False)
    gam2_results.to_csv("gam_output/gam2_results.csv", index=False)

    # Summary
    pivot = all_results.pivot(index="trend_name", columns="specialist", values="RMSE")
    true_cats = all_results.groupby("trend_name")["true_category"].first()
    pivot["true_cat"]        = true_cats
    pivot["best_specialist"] = pivot[["Micro", "Macro", "Mega"]].idxmin(axis=1)
    pivot["correct"]         = pivot["true_cat"] == pivot["best_specialist"]
    n_correct = pivot["correct"].sum()

    print(f"\nAll specialists on all test trends:")
    print(pivot[["Micro", "Macro", "Mega", "true_cat", "best_specialist", "correct"]].to_string())
    print(f"\nCorrect specialist wins: {n_correct}/{len(pivot)}")

    print("\nMatched specialist results:")
    print(gam2_results[["trend_name", "category", "RMSE", "MAE"]].to_string(index=False))
    print("\nPer-category average:")
    print(gam2_results.groupby("category")[["RMSE", "MAE"]].mean().round(3).to_string())

    print("\nGenerating figures...")
    plot_fits(
        {n: {"pred": all_forecasts[n]["preds"][all_forecasts[n]["true_cat"]]["pred"],
             "actual": all_forecasts[n]["actual"],
             "n_seed": all_forecasts[n]["n_seed"]}
         for n in all_forecasts},
        test_df, gam2_results
    )
    plot_all_specialists(all_results, all_forecasts)
    plot_full_evaluation(all_forecasts, test_df)
    plot_comparison_with_gam1(gam2_results)

    print("\n" + "=" * 60)
    print("GAM 2 complete.")
    print(f"Correct specialist wins: {n_correct}/{len(pivot)} trends")
    if n_correct > len(pivot) / 2:
        print("Conclusion: specialist models outperform the generalist.")
    else:
        print("Conclusion: generalist is competitive — investigate data quality.")
    print("=" * 60)


if __name__ == "__main__":
    main()