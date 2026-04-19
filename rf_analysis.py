"""
Fashion Trend Random Forest Pipeline
=====================================
Phase 1: Unsupervised - engineer features from time series, cluster blindly,
         visualise with UMAP, compare clusters to true labels after.
Phase 2: Supervised RF - use Micro/Macro/Mega labels, cross-validate,
         plot feature importances and confusion matrix.

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
import matplotlib.gridspec as gridspec
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── try importing UMAP (optional but nice) ──────────────────────────────────
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[INFO] umap-learn not installed — will use PCA for 2-D projection instead.")
    print("       Install with:  pip install umap-learn")

# ── output folder ────────────────────────────────────────────────────────────
OUT = "rf_output"
os.makedirs(OUT, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
DATE_COLS   = ["Month", "Date", "date", "month", "Week", "Time"]
THRESHOLD   = 0.35
GAP_TOL     = 1
LABEL_ORDER = ["Micro", "Macro", "Mega"]
COLOURS     = {"Micro": "#e63946", "Macro": "#457b9d", "Mega": "#2d6a4f"}
SEED        = 42


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
    """Returns DataFrame with columns [date, value] sorted by date."""
    for skip in (0, 1):
        try:
            df = pd.read_csv(path, skiprows=skip)
            dc = find_date_col(df)
            if dc is None:
                continue
            vc = find_value_col(df, dc)
            if vc is None:
                continue
            df = df[[dc, vc]].copy()
            df.columns = ["date", "value"]
            df["date"]  = pd.to_datetime(df["date"],  errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().sort_values("date").reset_index(drop=True)
            if len(df) > 5:
                return df
        except Exception:
            pass
    return None


def load_all_trends(data_folder):
    """
    Returns a dict: { trend_name: (df, true_label) }
    Tries individual CSVs first; falls back to combined CSV per folder.
    """
    folder_label = {
        "Macrotrends": "Macro",
        "MegaTrends":  "Mega",
        "Microtrends": "Micro",
    }
    trends = {}

    for folder, label in folder_label.items():
        folder_path = os.path.join(data_folder, folder)
        if not os.path.isdir(folder_path):
            print(f"[WARN] folder not found: {folder_path}")
            continue

        # collect individual CSVs (skip the combined one)
        csvs = [f for f in glob.glob(os.path.join(folder_path, "*.csv"))
                if "combined" not in os.path.basename(f).lower()
                and not os.path.basename(f).startswith(".")]

        for path in csvs:
            name = os.path.splitext(os.path.basename(path))[0]
            df = load_single_csv(path)
            if df is not None and len(df) > 10:
                trends[name] = (df, label)
            else:
                print(f"[WARN] could not load {name}")

    print(f"\n[INFO] Loaded {len(trends)} trends successfully.\n")
    return trends


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def normalise(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(np.zeros(len(s)), index=s.index)


def fill_gaps(mask, tol=1):
    mask = list(mask)
    i = 0
    while i < len(mask):
        if not mask[i]:
            start = i
            while i < len(mask) and not mask[i]:
                i += 1
            end = i - 1
            if (start > 0 and mask[start-1] and
                    end < len(mask)-1 and mask[end+1] and
                    end - start + 1 <= tol):
                for j in range(start, end+1):
                    mask[j] = True
        else:
            i += 1
    return mask


def longest_run(mask):
    best, cur, best_s = 0, 0, 0
    cur_s = 0
    for i, v in enumerate(mask):
        if v:
            if cur == 0:
                cur_s = i
            cur += 1
        else:
            if cur > best:
                best, best_s = cur, cur_s
            cur = 0
    if cur > best:
        best, best_s = cur, cur_s
    return best, best_s


def extract_features(df, name=""):
    """Extract a rich feature vector from a single trend time series."""
    v = df["value"].values.astype(float)
    n = normalise(pd.Series(v)).values

    # ── basic stats ─────────────────────────────────────────────────────
    feat = {}
    feat["n_points"]        = len(v)
    feat["mean_raw"]        = v.mean()
    feat["std_raw"]         = v.std()
    feat["max_raw"]         = v.max()
    feat["skewness"]        = float(skew(v))
    feat["kurt"]            = float(kurtosis(v))

    # ── peak analysis ────────────────────────────────────────────────────
    peaks, props = find_peaks(n, height=THRESHOLD, distance=3)
    feat["n_peaks"]         = len(peaks)
    feat["mean_peak_height"]= float(np.mean(props["peak_heights"])) if len(peaks) else 0.0
    feat["max_peak_height"] = float(np.max(props["peak_heights"]))  if len(peaks) else 0.0
    if len(peaks) >= 2:
        feat["peak_spacing_mean"] = float(np.diff(peaks).mean())
        feat["peak_spacing_std"]  = float(np.diff(peaks).std())
    else:
        feat["peak_spacing_mean"] = 0.0
        feat["peak_spacing_std"]  = 0.0

    # ── time above threshold ─────────────────────────────────────────────
    active = fill_gaps((n >= THRESHOLD).tolist(), GAP_TOL)
    feat["frac_above_thresh"] = sum(active) / len(active)

    lr, lr_start = longest_run(active)
    feat["longest_run_months"] = int(lr)
    feat["longest_run_frac"]   = lr / len(active)

    # ── rise / fall slopes around the main peak ──────────────────────────
    peak_idx = int(np.argmax(n))
    if peak_idx > 0:
        feat["rise_slope"] = float((n[peak_idx] - n[0]) / peak_idx)
    else:
        feat["rise_slope"] = 0.0
    tail = len(n) - peak_idx - 1
    if tail > 0:
        feat["fall_slope"] = float((n[-1] - n[peak_idx]) / tail)
    else:
        feat["fall_slope"] = 0.0

    # ── relative peak position (early vs late trend) ─────────────────────
    feat["peak_position_frac"] = peak_idx / max(len(n) - 1, 1)

    # ── autocorrelation (lag 1 and lag 12) ───────────────────────────────
    if len(v) > 12:
        feat["acf_lag1"]  = float(pd.Series(v).autocorr(lag=1))
        feat["acf_lag12"] = float(pd.Series(v).autocorr(lag=12))
    else:
        feat["acf_lag1"]  = 0.0
        feat["acf_lag12"] = 0.0

    # ── still active at end of series? ───────────────────────────────────
    feat["still_active"] = int(active[-1])

    # ── trend in second half vs first half ───────────────────────────────
    mid = len(n) // 2
    feat["second_half_mean"] = float(n[mid:].mean())
    feat["first_half_mean"]  = float(n[:mid].mean())
    feat["half_diff"]        = feat["second_half_mean"] - feat["first_half_mean"]

    return feat


def build_feature_matrix(trends):
    rows, labels, names = [], [], []
    for name, (df, label) in trends.items():
        feat = extract_features(df, name)
        rows.append(feat)
        labels.append(label)
        names.append(name)

    X = pd.DataFrame(rows, index=names)
    y = pd.Series(labels, index=names, name="label")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PHASE 1 — UNSUPERVISED CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

def phase1_unsupervised(X, y):
    print("=" * 60)
    print("PHASE 1 — UNSUPERVISED CLUSTERING (blind)")
    print("=" * 60)

    # KMeans with k=3 (same as number of true classes)
    km = KMeans(n_clusters=3, random_state=SEED, n_init=20)
    cluster_ids = km.fit_predict(X.values)

    # 2-D projection
    if HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=SEED, n_neighbors=min(10, len(X)-1))
        proj_label = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=SEED)
        proj_label = "PCA"

    coords = reducer.fit_transform(X.values)

    # ── figure: two panels ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")

    # Left: coloured by cluster
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        s=80, alpha=0.85, label=f"Cluster {cid}")
        for xi, yi, nm in zip(coords[mask, 0], coords[mask, 1],
                              np.array(X.index)[mask]):
            axes[0].annotate(nm, (xi, yi), fontsize=5.5, color="white",
                             alpha=0.6, ha="center", va="bottom")
    axes[0].set_title(f"{proj_label} — KMeans clusters (blind)", color="white", fontsize=12)
    axes[0].legend(labelcolor="white", facecolor="#2a2a2a", edgecolor="none")

    # Right: coloured by true label
    for lbl in LABEL_ORDER:
        mask = y.values == lbl
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        s=80, alpha=0.85, label=lbl, color=COLOURS[lbl])
        for xi, yi, nm in zip(coords[mask, 0], coords[mask, 1],
                              np.array(X.index)[mask]):
            axes[1].annotate(nm, (xi, yi), fontsize=5.5, color="white",
                             alpha=0.6, ha="center", va="bottom")
    axes[1].set_title(f"{proj_label} — True labels (revealed)", color="white", fontsize=12)
    axes[1].legend(labelcolor="white", facecolor="#2a2a2a", edgecolor="none")

    for ax in axes:
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    plt.suptitle("Phase 1 — Do clusters match trend classes?",
                 color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUT, "phase1_clustering.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")

    # ── alignment check: which cluster maps to which label? ─────────────
    cluster_df = pd.DataFrame({
        "trend": X.index,
        "cluster": cluster_ids,
        "true_label": y.values
    })
    print("\n  Cluster composition:")
    print(cluster_df.groupby(["cluster", "true_label"]).size().unstack(fill_value=0).to_string())
    cluster_df.to_csv(os.path.join(OUT, "phase1_cluster_assignments.csv"), index=False)
    print()

    return cluster_ids, coords


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PHASE 2 — SUPERVISED RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════

def phase2_supervised(X, y):
    print("=" * 60)
    print("PHASE 2 — SUPERVISED RANDOM FOREST")
    print("=" * 60)

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y_enc = le.transform(y)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )

    # Stratified K-Fold (use leave-one-out style for small dataset)
    n_splits = min(5, len(X) // 3)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_pred_enc = cross_val_predict(rf, X.values, y_enc, cv=cv)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y, y_pred)
    print(f"\n  Cross-validated accuracy: {acc:.1%}  ({n_splits}-fold stratified CV)")
    print(f"\n  Classification report:\n")
    print(classification_report(y, y_pred, target_names=LABEL_ORDER))

    # ── fit on full data for feature importances ─────────────────────────
    rf.fit(X.values, y_enc)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

    # ── FIGURE 1: confusion matrix ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#1a1a1a")
    cm = confusion_matrix(y, y_pred, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix  (CV acc = {acc:.1%})", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for text in ax.texts:
        text.set_color("black")
    plt.tight_layout()
    cm_path = os.path.join(OUT, "phase2_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {cm_path}")

    # ── FIGURE 2: feature importances ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#1a1a1a")
    colours_bar = ["#e63946" if importances.values[i] > importances.median()
                   else "#457b9d" for i in range(len(importances))]
    ax.barh(importances.index, importances.values, color=colours_bar, alpha=0.85)
    ax.set_xlabel("Mean decrease in impurity", color="white")
    ax.set_title("Feature Importances — Supervised RF", color="white", fontsize=12)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    fi_path = os.path.join(OUT, "phase2_feature_importances.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {fi_path}")

    # ── FIGURE 3: per-trend prediction vs truth ───────────────────────────
    results_df = pd.DataFrame({
        "trend":      X.index,
        "true_label": y.values,
        "pred_label": y_pred,
        "correct":    y.values == y_pred
    }).sort_values(["true_label", "trend"])
    results_df.to_csv(os.path.join(OUT, "phase2_predictions.csv"), index=False)

    wrong = results_df[~results_df["correct"]]
    if len(wrong):
        print(f"\n  Misclassified trends ({len(wrong)}):")
        for _, row in wrong.iterrows():
            print(f"    {row['trend']:25s}  true={row['true_label']}  pred={row['pred_label']}")
    else:
        print("\n  All trends correctly classified in CV! 🎉")

    return rf, importances, results_df


# ═══════════════════════════════════════════════════════════════════════════
# 5.  BONUS: TIME SERIES GALLERY
# ═══════════════════════════════════════════════════════════════════════════

def plot_time_series_gallery(trends, y, n_per_class=3):
    """Quick gallery showing representative trends from each class."""
    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#0d0d0d")
    gs = gridspec.GridSpec(3, n_per_class, hspace=0.5, wspace=0.35)

    for row, label in enumerate(LABEL_ORDER):
        names = [n for n, (_, lbl) in trends.items() if lbl == label][:n_per_class]
        for col, name in enumerate(names):
            df, _ = trends[name]
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#1a1a1a")
            norm = normalise(df["value"])
            ax.fill_between(df["date"], norm, alpha=0.25, color=COLOURS[label])
            ax.plot(df["date"], norm, color=COLOURS[label], linewidth=1.2)
            ax.axhline(THRESHOLD, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
            ax.set_title(name, color="white", fontsize=7.5)
            if col == 0:
                ax.set_ylabel(label, color=COLOURS[label], fontsize=9, fontweight="bold")
            ax.tick_params(colors="gray", labelsize=6)
            ax.set_ylim(-0.05, 1.05)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

    fig.suptitle("Representative trends by class", color="white", fontsize=13, y=1.01)
    out_path = os.path.join(OUT, "trend_gallery.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="data",
                        help="Path to the data folder (default: data)")
    args = parser.parse_args()

    # 1. load
    trends = load_all_trends(args.data_folder)
    if len(trends) < 6:
        raise RuntimeError("Too few trends loaded — check your data folder path.")

    # 2. features
    print("[INFO] Engineering features...")
    X, y = build_feature_matrix(trends)
    X = X.fillna(0)
    print(f"  Feature matrix: {X.shape[0]} trends × {X.shape[1]} features")
    X.to_csv(os.path.join(OUT, "feature_matrix.csv"))
    print(f"  Saved feature matrix to rf_output/feature_matrix.csv\n")

    # 3. gallery
    print("[INFO] Plotting time series gallery...")
    plot_time_series_gallery(trends, y)

    # 4. Phase 1
    cluster_ids, coords = phase1_unsupervised(X, y)

    # 5. Phase 2
    rf, importances, results_df = phase2_supervised(X, y)

    print("\n" + "=" * 60)
    print(f"All outputs saved to:  ./{OUT}/")
    print("=" * 60)


if __name__ == "__main__":
    main()