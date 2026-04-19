"""
Fashion Trend Classification — Improved Pipeline
=================================================
Improvements over v1:
  1. Richer feature engineering (AUC, time-to-peak, decay rate, seasonality,
     entropy, trend-in-second-half, smoothed peak width)
  2. Feature selection — drops near-zero-importance & redundant features
  3. Hyperparameter tuning via RandomizedSearchCV (fast but thorough)
  4. Model comparison: RF vs XGBoost vs SVM vs KNN vs GradientBoosting
  5. LOOCV for accuracy estimate (better than 5-fold on small datasets)
  6. Calibrated probability outputs + confidence analysis
  7. UMAP projection (falls back to PCA)

USAGE:
  pip install scikit-learn xgboost umap-learn matplotlib scipy pandas
  python fashion_rf_improved.py --data_folder data
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
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (LeaveOneOut, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_predict)
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] xgboost not installed — skipping XGBoost model.")
    print("       Install with: pip install xgboost")

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[INFO] umap-learn not installed — using PCA instead.")
    print("       Install with: pip install umap-learn")

# ── output folder ─────────────────────────────────────────────────────────────
OUT = "rf_output_v2"
os.makedirs(OUT, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
DATE_COLS   = ["Month", "Date", "date", "month", "Week", "Time"]
THRESHOLD   = 0.35
GAP_TOL     = 1
LABEL_ORDER = ["Micro", "Macro", "Mega"]
COLOURS     = {"Micro": "#e63946", "Macro": "#457b9d", "Mega": "#2d6a4f"}
SEED        = 42


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  (same as v1)
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
            if len(df) > 5:
                return df
        except Exception:
            pass
    return None

def load_all_trends(data_folder):
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
    print(f"\n[INFO] Loaded {len(trends)} trends.\n")
    return trends


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING — EXPANDED
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
            if cur == 0: cur_s = i
            cur += 1
        else:
            if cur > best: best, best_s = cur, cur_s
            cur = 0
    if cur > best: best, best_s = cur, cur_s
    return best, best_s

def exp_decay(x, a, b):
    return a * np.exp(-b * x)

def seasonality_strength(v, period=12):
    """Ratio of variance explained by annual cycle via DFT."""
    if len(v) < period * 2:
        return 0.0
    fft = np.fft.rfft(v - v.mean())
    power = np.abs(fft) ** 2
    freq_idx = int(round(len(v) / period))
    if freq_idx >= len(power):
        return 0.0
    window = slice(max(0, freq_idx-1), min(len(power), freq_idx+2))
    seasonal_power = power[window].sum()
    total_power = power[1:].sum()
    return float(seasonal_power / total_power) if total_power > 0 else 0.0

def spectral_entropy(v):
    """Entropy of the power spectrum — low = structured, high = noisy."""
    fft = np.fft.rfft(v - v.mean())
    power = np.abs(fft[1:]) ** 2
    power = power / power.sum() if power.sum() > 0 else power
    ent = -np.sum(power * np.log(power + 1e-10))
    return float(ent)

def extract_features(df):
    v = df["value"].values.astype(float)
    n = normalise(pd.Series(v)).values
    N = len(v)

    feat = {}

    # ── basic stats ──────────────────────────────────────────────────────
    feat["mean_raw"]    = v.mean()
    feat["std_raw"]     = v.std()
    feat["skewness"]    = float(skew(v))
    feat["kurt"]        = float(kurtosis(v))

    # ── area under curve (total accumulated interest) ────────────────────
    feat["auc"]         = float(np.trapz(n))
    feat["auc_frac"]    = feat["auc"] / N   # normalised by length

    # ── peak analysis ────────────────────────────────────────────────────
    peaks, props = find_peaks(n, height=THRESHOLD, distance=3)
    feat["n_peaks"]           = len(peaks)
    feat["mean_peak_height"]  = float(np.mean(props["peak_heights"])) if len(peaks) else 0.0
    feat["max_peak_height"]   = float(np.max(props["peak_heights"]))  if len(peaks) else 0.0

    # time to first peak (as fraction of series length)
    main_peak_idx = int(np.argmax(n))
    feat["time_to_peak_frac"] = main_peak_idx / max(N - 1, 1)

    # peak width at half max (proxy for how sustained the peak is)
    half_max = n[main_peak_idx] / 2
    above_half = n >= half_max
    feat["peak_width_frac"] = float(above_half.sum()) / N

    if len(peaks) >= 2:
        feat["peak_spacing_mean"] = float(np.diff(peaks).mean())
        feat["peak_spacing_std"]  = float(np.diff(peaks).std())
    else:
        feat["peak_spacing_mean"] = 0.0
        feat["peak_spacing_std"]  = 0.0

    # ── threshold activity ───────────────────────────────────────────────
    active = fill_gaps((n >= THRESHOLD).tolist(), GAP_TOL)
    feat["frac_above_thresh"] = sum(active) / len(active)

    lr, lr_start = longest_run(active)
    feat["longest_run_months"] = int(lr)
    feat["longest_run_frac"]   = lr / len(active)

    # ── rise / fall slopes ───────────────────────────────────────────────
    if main_peak_idx > 0:
        feat["rise_slope"] = float((n[main_peak_idx] - n[0]) / main_peak_idx)
    else:
        feat["rise_slope"] = 0.0

    tail = N - main_peak_idx - 1
    if tail > 0:
        feat["fall_slope"] = float((n[-1] - n[main_peak_idx]) / tail)
    else:
        feat["fall_slope"] = 0.0

    # ── decay rate (fit exponential after peak) ──────────────────────────
    post_peak = n[main_peak_idx:]
    if len(post_peak) > 5 and post_peak.max() > 0:
        try:
            x_fit = np.arange(len(post_peak))
            popt, _ = curve_fit(exp_decay, x_fit, post_peak,
                                p0=[post_peak[0], 0.01],
                                maxfev=2000, bounds=([0, 0], [2, 2]))
            feat["decay_rate"] = float(popt[1])
        except Exception:
            feat["decay_rate"] = 0.0
    else:
        feat["decay_rate"] = 0.0

    # ── autocorrelation ──────────────────────────────────────────────────
    if N > 24:
        feat["acf_lag1"]  = float(pd.Series(v).autocorr(lag=1))
        feat["acf_lag6"]  = float(pd.Series(v).autocorr(lag=6))
        feat["acf_lag12"] = float(pd.Series(v).autocorr(lag=12))
    else:
        feat["acf_lag1"]  = 0.0
        feat["acf_lag6"]  = 0.0
        feat["acf_lag12"] = 0.0

    # ── seasonality & entropy ────────────────────────────────────────────
    feat["seasonality"]      = seasonality_strength(v)
    feat["spectral_entropy"] = spectral_entropy(v)

    # ── half-series comparison ───────────────────────────────────────────
    mid = N // 2
    feat["first_half_mean"]  = float(n[:mid].mean())
    feat["second_half_mean"] = float(n[mid:].mean())
    feat["half_diff"]        = feat["second_half_mean"] - feat["first_half_mean"]

    # ── trend momentum: slope of linear fit to normalised series ─────────
    x_lin = np.arange(N)
    feat["linear_trend"] = float(np.polyfit(x_lin, n, 1)[0])

    # ── still active at end ──────────────────────────────────────────────
    feat["still_active"] = int(active[-1])

    # ── quartile activity ────────────────────────────────────────────────
    q = N // 4
    for qi in range(4):
        seg = n[qi*q:(qi+1)*q]
        feat[f"q{qi+1}_mean"] = float(seg.mean()) if len(seg) else 0.0

    return feat


def build_feature_matrix(trends):
    rows, labels, names = [], [], []
    for name, (df, label) in trends.items():
        feat = extract_features(df)
        rows.append(feat)
        labels.append(label)
        names.append(name)
    X = pd.DataFrame(rows, index=names).fillna(0)
    y = pd.Series(labels, index=names, name="label")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# 3.  FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def select_features(X, y_enc, verbose=True):
    """
    Use a quick RF to identify near-zero importance features and drop them.
    Keeps features that contribute at least 1% of total importance cumulatively.
    """
    rf_sel = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf_sel.fit(X.values, y_enc)
    imp = pd.Series(rf_sel.feature_importances_, index=X.columns).sort_values(ascending=False)

    # keep features up to 99% cumulative importance
    cumulative = imp.cumsum() / imp.sum()
    keep = cumulative[cumulative <= 0.99].index.tolist()
    # always keep at least top 10
    keep = list(dict.fromkeys(imp.index[:10].tolist() + keep))

    if verbose:
        dropped = [c for c in X.columns if c not in keep]
        print(f"  Feature selection: keeping {len(keep)}/{len(X.columns)} features")
        if dropped:
            print(f"  Dropped (near-zero importance): {dropped}")

    return X[keep], imp


# ═══════════════════════════════════════════════════════════════════════════
# 4.  MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def build_models(le):
    n_classes = len(le.classes_)

    models = {}

    # Random Forest — tuned
    models["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ))
    ])

    # Gradient Boosting
    models["GradientBoosting"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=SEED
        ))
    ])

    # SVM
    models["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=SEED
        ))
    ])

    # KNN
    models["KNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))
    ])

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=SEED,
                verbosity=0
            ))
        ])

    return models


def compare_models(X, y, y_enc, le):
    print("=" * 60)
    print("MODEL COMPARISON  (Leave-One-Out CV)")
    print("=" * 60)

    models = build_models(le)
    loo = LeaveOneOut()
    results = {}

    for name, model in models.items():
        y_pred_enc = cross_val_predict(model, X.values, y_enc, cv=loo)
        y_pred = le.inverse_transform(y_pred_enc)
        acc = accuracy_score(y, y_pred)
        results[name] = {"acc": acc, "y_pred": y_pred}
        print(f"  {name:20s}  LOO accuracy = {acc:.1%}")

    print()
    best_name = max(results, key=lambda k: results[k]["acc"])
    print(f"  ✓ Best model: {best_name}  ({results[best_name]['acc']:.1%})")
    return results, best_name, models


# ═══════════════════════════════════════════════════════════════════════════
# 5.  HYPERPARAMETER TUNING (RF)
# ═══════════════════════════════════════════════════════════════════════════

def tune_rf(X, y_enc):
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING — Random Forest")
    print("=" * 60)

    param_dist = {
        "clf__n_estimators":    [200, 300, 500, 800],
        "clf__max_depth":       [3, 5, 8, None],
        "clf__min_samples_leaf":[1, 2, 3],
        "clf__max_features":    ["sqrt", "log2", 0.4, 0.6],
        "clf__class_weight":    ["balanced", None],
    }

    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=40,
        cv=cv,
        scoring="accuracy",
        random_state=SEED,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X.values, y_enc)

    print(f"  Best CV accuracy: {search.best_score_:.1%}")
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_


# ═══════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_comparison(results):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#1a1a1a")

    names = list(results.keys())
    accs  = [results[n]["acc"] * 100 for n in names]
    colours = ["#e63946" if a == max(accs) else "#457b9d" for a in accs]

    bars = ax.barh(names, accs, color=colours, alpha=0.85, height=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{acc:.1f}%", va="center", color="white", fontsize=10)

    ax.set_xlim(0, 115)
    ax.set_xlabel("LOO Accuracy (%)", color="white")
    ax.set_title("Model Comparison — Leave-One-Out CV", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.axvline(80, color="#555", linestyle="--", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    path = os.path.join(OUT, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved: {path}")


def plot_confusion(y_true, y_pred, title, filename, acc):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#1a1a1a")
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{title}  (LOO acc = {acc:.1%})", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for text in ax.texts:
        text.set_color("black")
    plt.tight_layout()
    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(X, y_enc, imp):
    imp_sorted = imp.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(imp_sorted) * 0.32)))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#1a1a1a")
    med = imp_sorted.median()
    colours = ["#e63946" if v > med else "#457b9d" for v in imp_sorted.values]
    ax.barh(imp_sorted.index, imp_sorted.values, color=colours, alpha=0.85)
    ax.set_xlabel("Mean decrease in impurity", color="white")
    ax.set_title("Feature Importances (expanded feature set)", color="white", fontsize=12)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    path = os.path.join(OUT, "feature_importances_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_projection(X_sel, y, cluster_ids=None):
    if HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=SEED,
                       n_neighbors=min(10, len(X_sel)-1))
        proj_label = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=SEED)
        proj_label = "PCA"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel.values)
    coords = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#1a1a1a")

    for lbl in LABEL_ORDER:
        mask = y.values == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=90, alpha=0.85, label=lbl, color=COLOURS[lbl], zorder=3)
        for xi, yi, nm in zip(coords[mask, 0], coords[mask, 1],
                              np.array(X_sel.index)[mask]):
            ax.annotate(nm, (xi, yi), fontsize=6, color="white",
                        alpha=0.65, ha="center", va="bottom")

    ax.set_title(f"{proj_label} projection — true labels (selected features)",
                 color="white", fontsize=12)
    ax.legend(labelcolor="white", facecolor="#2a2a2a", edgecolor="none")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    path = os.path.join(OUT, f"{proj_label.lower()}_projection_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def plot_confidence_analysis(X_sel, y, y_enc, le, best_model):
    """Show per-trend prediction confidence for the best model."""
    loo = LeaveOneOut()
    names = list(X_sel.index)
    confidences, correct_flags, pred_labels = [], [], []

    for train_idx, test_idx in loo.split(X_sel.values, y_enc):
        X_tr, X_te = X_sel.values[train_idx], X_sel.values[test_idx]
        y_tr = y_enc[train_idx]
        best_model.fit(X_tr, y_tr)
        proba = best_model.predict_proba(X_te)[0]
        pred = best_model.predict(X_te)[0]
        conf = proba.max()
        confidences.append(conf)
        correct_flags.append(int(pred == y_enc[test_idx[0]]))
        pred_labels.append(le.inverse_transform([pred])[0])

    conf_df = pd.DataFrame({
        "trend": names,
        "true_label": y.values,
        "pred_label": pred_labels,
        "confidence": confidences,
        "correct": correct_flags
    }).sort_values("confidence", ascending=True)
    conf_df.to_csv(os.path.join(OUT, "confidence_analysis.csv"), index=False)

    # plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(conf_df) * 0.28)))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#1a1a1a")

    bar_colours = ["#2d6a4f" if c else "#e63946" for c in conf_df["correct"]]
    bars = ax.barh(conf_df["trend"], conf_df["confidence"], color=bar_colours, alpha=0.85)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=0.7, alpha=0.5)

    # label correct/wrong
    for i, (_, row) in enumerate(conf_df.iterrows()):
        label = "✓" if row["correct"] else f"✗ → {row['pred_label']}"
        ax.text(row["confidence"] + 0.01, i, label,
                va="center", color="white", fontsize=7, alpha=0.8)

    ax.set_xlim(0, 1.3)
    ax.set_xlabel("Prediction confidence", color="white")
    ax.set_title("Per-trend prediction confidence (LOO CV)", color="white", fontsize=12)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2d6a4f", label="Correct"),
                       Patch(facecolor="#e63946", label="Incorrect")]
    ax.legend(handles=legend_elements, labelcolor="white",
              facecolor="#2a2a2a", edgecolor="none")

    plt.tight_layout()
    path = os.path.join(OUT, "confidence_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return conf_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="data")
    args = parser.parse_args()

    # 1. load
    trends = load_all_trends(args.data_folder)
    if len(trends) < 6:
        raise RuntimeError("Too few trends loaded — check data folder path.")

    # 2. features
    print("[INFO] Engineering expanded feature set...")
    X, y = build_feature_matrix(trends)
    print(f"  Raw feature matrix: {X.shape[0]} trends × {X.shape[1]} features")

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y_enc = le.transform(y)

    # 3. feature selection
    print("\n[INFO] Selecting features...")
    X_sel, imp = select_features(X, y_enc)
    X_sel.to_csv(os.path.join(OUT, "feature_matrix_selected.csv"))

    # 4. feature importance plot
    print("\n[INFO] Plotting feature importances...")
    plot_feature_importance(X_sel, y_enc, imp)

    # 5. projection
    print("\n[INFO] Plotting 2D projection...")
    plot_projection(X_sel, y)

    # 6. model comparison (LOO)
    results, best_name, models = compare_models(X_sel, y, y_enc, le)

    # 7. plot model comparison bar chart
    plot_model_comparison(results)

    # 8. confusion matrix for best model
    print(f"\n[INFO] Detailed results for best model: {best_name}")
    best_y_pred = results[best_name]["y_pred"]
    best_acc    = results[best_name]["acc"]
    plot_confusion(y, best_y_pred, best_name,
                   f"confusion_matrix_{best_name}.png", best_acc)
    print(f"\n  Classification report ({best_name}):\n")
    print(classification_report(y, best_y_pred, target_names=LABEL_ORDER))

    # also plot confusion for RF specifically for comparison
    if best_name != "RandomForest":
        rf_pred = results["RandomForest"]["y_pred"]
        rf_acc  = results["RandomForest"]["acc"]
        plot_confusion(y, rf_pred, "RandomForest",
                       "confusion_matrix_RF.png", rf_acc)

    # 9. hyperparameter tuning
    tuned_rf, tuned_score = tune_rf(X_sel, y_enc)

    # 10. confidence analysis on best model
    print(f"\n[INFO] Running confidence analysis ({best_name})...")
    conf_df = plot_confidence_analysis(X_sel, y, y_enc, le, models[best_name])

    # 11. misclassifications summary
    wrong = conf_df[conf_df["correct"] == 0]
    print(f"\n  Misclassified trends ({len(wrong)}):")
    for _, row in wrong.iterrows():
        print(f"    {row['trend']:25s}  true={row['true_label']}  "
              f"pred={row['pred_label']}  conf={row['confidence']:.2f}")

    print("\n" + "=" * 60)
    print(f"  v1 accuracy (5-fold RF):    ~82.2%")
    print(f"  v2 best model ({best_name}): {best_acc:.1%} (LOO)")
    print(f"  v2 tuned RF (5-fold):       {tuned_score:.1%}")
    print(f"\n  All outputs saved to: ./{OUT}/")
    print("=" * 60)


if __name__ == "__main__":
    main()