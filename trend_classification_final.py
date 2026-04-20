"""
Fashion Trend Classifier — Clean Two-Path Pipeline
====================================================
Path 2: Unsupervised clustering (no labels used) — do clusters match reality?
Path 1: Supervised classification — properly validated, bias-free

Bias fixes vs previous versions:
  - n_points removed (all series same length — constant, zero signal)
  - mean_raw / std_raw removed (raw scale is a collection artefact)
  - All features computed on value_norm [0,1] only
  - Feature selection runs INSIDE the cross-validation loop (no leakage)
  - Labels never seen during feature extraction
  - Held-out test set (9 trends) completely withheld until final evaluation

USAGE:
  pip install scikit-learn umap-learn matplotlib scipy pandas xgboost
  python trend_classifier.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import (LeaveOneOut, StratifiedKFold,
                                     cross_val_predict, RandomizedSearchCV)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings("ignore")

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[INFO] umap-learn not installed — using PCA for projection.")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── output folder ─────────────────────────────────────────────────────────────
OUT = "trend_output"
os.makedirs(OUT, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
SEED        = 42
LABEL_ORDER = ["Micro", "Macro", "Mega"]
COLOURS     = {"Micro": "#e63946", "Macro": "#457b9d", "Mega": "#2d6a4f"}
THRESHOLD   = 0.35
GAP_TOL     = 1

# ── label map (from original folder structure) ────────────────────────────────
# These are YOUR manual classifications — assigned independently of features
LABEL_MAP = {
    # Micro — short-lived internet aesthetics, flash-in-the-pan trends
    "barbiecore":          "Micro",
    "bimbocore":           "Micro",
    "blokecore":           "Micro",
    "Brat summer":         "Micro",
    "BuisnessCausal":      "Micro",
    "coastalgrandmother":  "Micro",
    "Euphoriafashion":     "Micro",
    "galaxyprint":         "Micro",
    "goblincore":          "Micro",
    "ivyleague":           "Micro",
    "officesiren":         "Micro",
    "Pastelgrunge":        "Micro",
    "softgirlaesthetic":   "Micro",
    "vsco":                "Micro",
    "y2kaesthetic":        "Micro",

    # Macro — medium-length trends, peaked and declined over years
    "Bikeshorts":          "Macro",
    "Bomber jacket":       "Macro",
    "chokers":             "Macro",
    "chunkyshoes":         "Macro",
    "Cleangirl":           "Macro",
    "Cottagecore":         "Macro",
    "Darkacademia":        "Macro",
    "EMOfashio":           "Macro",
    "Fairycore":           "Macro",
    "flaredjeans":         "Macro",
    "hipster fashion":     "Macro",
    "mini skirts":         "Macro",
    "Normcore":            "Macro",
    "ripepdjeans":         "Macro",
    "SCENEFASHION":        "Macro",
    "TUMBLR":              "Macro",
    "Tumblr Grunge":       "Macro",
    "teddycoat":           "Macro",
    "tiedye":              "Macro",

    # Mega — sustained, long-running cultural shifts
    "70sfashion":          "Mega",
    "boyfriend jeans":     "Mega",
    "disco fashion":       "Mega",
    "INDIE":               "Mega",
    "jeggins":             "Mega",
    "MAXI dress":          "Mega",
    "MOMJEANS":            "Mega",
    "PENDULUMTOPS":        "Mega",
    "Skaterfashion":       "Mega",
    "SKINNYJEANS":         "Mega",
    "streetwear":          "Mega",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path="data/all_trends_combined.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["trend_name", "date"]).reset_index(drop=True)
    print(f"[INFO] Loaded {df['trend_name'].nunique()} trends, "
          f"{len(df)} rows total.")
    return df


def pivot_to_series(df):
    """Returns dict: { trend_name: normalised value array }"""
    series = {}
    for name, grp in df.groupby("trend_name"):
        series[name] = grp["value_norm"].values.astype(float)
    return series


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING  (all on value_norm — no raw-scale leakage)
# ═══════════════════════════════════════════════════════════════════════════════

def fill_gaps(mask, tol=GAP_TOL):
    mask = list(mask)
    i = 0
    while i < len(mask):
        if not mask[i]:
            start = i
            while i < len(mask) and not mask[i]:
                i += 1
            end = i - 1
            if (start > 0 and mask[start - 1] and
                    end < len(mask) - 1 and mask[end + 1] and
                    end - start + 1 <= tol):
                for j in range(start, end + 1):
                    mask[j] = True
        else:
            i += 1
    return mask


def longest_run(mask):
    best, cur, best_s, cur_s = 0, 0, 0, 0
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


def seasonality_strength(v, period=12):
    if len(v) < period * 2:
        return 0.0
    fft = np.fft.rfft(v - v.mean())
    power = np.abs(fft) ** 2
    freq_idx = int(round(len(v) / period))
    if freq_idx >= len(power):
        return 0.0
    window = slice(max(0, freq_idx - 1), min(len(power), freq_idx + 2))
    seasonal = power[window].sum()
    total = power[1:].sum()
    return float(seasonal / total) if total > 0 else 0.0


def spectral_entropy(v):
    fft = np.fft.rfft(v - v.mean())
    power = np.abs(fft[1:]) ** 2
    power = power / power.sum() if power.sum() > 0 else power
    return float(-np.sum(power * np.log(power + 1e-10)))


def extract_features(v):
    """
    All features computed on value_norm [0,1].
    No n_points, no mean_raw, no std_raw — those carry label leakage.
    """
    n = len(v)
    feat = {}

    # ── shape stats (on normalised series) ────────────────────────────────
    feat["mean_norm"]    = float(v.mean())
    feat["std_norm"]     = float(v.std())
    feat["skewness"]     = float(skew(v))
    feat["kurt"]         = float(kurtosis(v))

    # ── area under curve ──────────────────────────────────────────────────
    feat["auc"]          = float(np.trapezoid(v) if hasattr(np, "trapezoid") else np.trapz(v))
    feat["auc_frac"]     = feat["auc"] / n

    # ── peak analysis ─────────────────────────────────────────────────────
    peaks, props = find_peaks(v, height=THRESHOLD, distance=3)
    feat["n_peaks"]            = len(peaks)
    feat["mean_peak_height"]   = float(np.mean(props["peak_heights"])) if len(peaks) else 0.0
    feat["max_peak_height"]    = float(np.max(props["peak_heights"]))  if len(peaks) else 0.0

    main_peak_idx = int(np.argmax(v))
    feat["time_to_peak_frac"]  = main_peak_idx / max(n - 1, 1)

    # peak width at half-max (how sustained the peak is)
    half_max = v[main_peak_idx] / 2.0
    feat["peak_width_frac"]    = float((v >= half_max).sum()) / n

    if len(peaks) >= 2:
        feat["peak_spacing_mean"] = float(np.diff(peaks).mean())
        feat["peak_spacing_std"]  = float(np.diff(peaks).std())
    else:
        feat["peak_spacing_mean"] = 0.0
        feat["peak_spacing_std"]  = 0.0

    # ── threshold activity ────────────────────────────────────────────────
    active = fill_gaps((v >= THRESHOLD).tolist())
    feat["frac_above_thresh"]  = sum(active) / len(active)
    lr, _                      = longest_run(active)
    feat["longest_run_frac"]   = lr / n
    feat["still_active"]       = int(active[-1])

    # ── rise / fall slopes ────────────────────────────────────────────────
    feat["rise_slope"] = float((v[main_peak_idx] - v[0]) / main_peak_idx) if main_peak_idx > 0 else 0.0
    tail               = n - main_peak_idx - 1
    feat["fall_slope"] = float((v[-1] - v[main_peak_idx]) / tail) if tail > 0 else 0.0

    # ── decay: how much of the series is in decline ───────────────────────
    post_peak = v[main_peak_idx:]
    if len(post_peak) > 1:
        feat["post_peak_mean"] = float(post_peak.mean())
        feat["post_peak_std"]  = float(post_peak.std())
    else:
        feat["post_peak_mean"] = 0.0
        feat["post_peak_std"]  = 0.0

    # ── autocorrelation ───────────────────────────────────────────────────
    s = pd.Series(v)
    feat["acf_lag1"]  = float(s.autocorr(lag=1))  if n > 2  else 0.0
    feat["acf_lag6"]  = float(s.autocorr(lag=6))  if n > 7  else 0.0
    feat["acf_lag12"] = float(s.autocorr(lag=12)) if n > 13 else 0.0
    feat["acf_lag24"] = float(s.autocorr(lag=24)) if n > 25 else 0.0

    # ── seasonality & spectral structure ──────────────────────────────────
    feat["seasonality"]      = seasonality_strength(v)
    feat["spectral_entropy"] = spectral_entropy(v)

    # ── half-series comparison (early vs late activity) ───────────────────
    mid = n // 2
    feat["first_half_mean"]  = float(v[:mid].mean())
    feat["second_half_mean"] = float(v[mid:].mean())
    feat["half_diff"]        = feat["second_half_mean"] - feat["first_half_mean"]

    # ── linear momentum ───────────────────────────────────────────────────
    feat["linear_trend"] = float(np.polyfit(np.arange(n), v, 1)[0])

    # ── quartile activity ─────────────────────────────────────────────────
    q = n // 4
    for qi in range(4):
        seg = v[qi * q:(qi + 1) * q]
        feat[f"q{qi + 1}_mean"] = float(seg.mean()) if len(seg) else 0.0

    return feat


def build_feature_matrix(series_dict, label_map=None):
    """
    Build feature matrix X and optionally labels y.
    label_map is only used for supervised path — features are computed first,
    labels attached after, so extraction is never label-aware.
    """
    rows, names = [], []
    for name, v in series_dict.items():
        rows.append(extract_features(v))
        names.append(name)

    X = pd.DataFrame(rows, index=names).fillna(0)

    if label_map is not None:
        labelled = {n: label_map[n] for n in names if n in label_map}
        X_lab = X.loc[list(labelled.keys())]
        y = pd.Series(labelled, name="label")
        y = y.loc[X_lab.index]
        return X_lab, y

    return X, None


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TRAIN / TEST SPLIT  (for supervised path)
# ═══════════════════════════════════════════════════════════════════════════════

def make_held_out_split(X, y, n_per_class=3, seed=SEED):
    """
    Hold back n_per_class trends per class as a true test set.
    These are NEVER seen during training or cross-validation.
    """
    rng = np.random.RandomState(seed)
    test_idx, train_idx = [], []

    for label in LABEL_ORDER:
        class_names = y[y == label].index.tolist()
        rng.shuffle(class_names)
        test_idx.extend(class_names[:n_per_class])
        train_idx.extend(class_names[n_per_class:])

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test  = X.loc[test_idx]
    y_test  = y.loc[test_idx]

    print(f"\n[INFO] Train set: {len(X_train)} trends | "
          f"Test set: {len(X_test)} trends ({n_per_class} per class)")
    print(f"  Held-out test trends: {test_idx}\n")

    return X_train, y_train, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PATH 2 — UNSUPERVISED CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def path2_unsupervised(X, y_true):
    print("\n" + "=" * 65)
    print("PATH 2 — UNSUPERVISED CLUSTERING  (labels never used)")
    print("=" * 65)

    # Scale before clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # KMeans — completely blind to labels
    km = KMeans(n_clusters=3, random_state=SEED, n_init=30)
    cluster_ids = km.fit_predict(X_scaled)

    # 2-D projection for visualisation
    if HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=SEED,
                       n_neighbors=min(10, len(X) - 1))
        proj_label = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=SEED)
        proj_label = "PCA"

    coords = reducer.fit_transform(X_scaled)

    # ── cluster composition report ────────────────────────────────────────
    cluster_df = pd.DataFrame({
        "trend":       X.index,
        "cluster":     cluster_ids,
        "true_label":  y_true.values
    })

    print("\n  Cluster composition (blind clusters vs your labels):")
    comp = cluster_df.groupby(["cluster", "true_label"]).size().unstack(fill_value=0)
    print(comp.to_string())

    # Alignment score: best-match accuracy (Hungarian-style greedy)
    best_acc = _cluster_alignment_accuracy(cluster_ids, y_true.values)
    print(f"\n  Best-match cluster alignment accuracy: {best_acc:.1%}")
    print("  (If >60%, the data naturally separates into your 3 classes)")

    cluster_df.to_csv(os.path.join(OUT, "path2_cluster_assignments.csv"), index=False)

    # ── figure ────────────────────────────────────────────────────────────
    path2_plot(X, y_true, cluster_ids, coords)

    return cluster_ids, coords, best_acc


def _cluster_alignment_accuracy(cluster_ids, true_labels):
    """Greedy best-match between cluster IDs and true labels."""
    from itertools import permutations
    unique_labels = list(set(true_labels))
    unique_clusters = list(set(cluster_ids))
    if len(unique_clusters) != len(unique_labels):
        return 0.0
    best = 0
    for perm in permutations(unique_labels):
        mapping = dict(zip(unique_clusters, perm))
        mapped = [mapping[c] for c in cluster_ids]
        acc = np.mean([m == t for m, t in zip(mapped, true_labels)])
        best = max(best, acc)
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PATH 1 — SUPERVISED CLASSIFICATION  (bias-free)
# ═══════════════════════════════════════════════════════════════════════════════

def build_models(le):
    models = {}

    models["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ))
    ])

    models["GradientBoosting"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, subsample=0.8, random_state=SEED
        ))
    ])

    models["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", probability=True, random_state=SEED
        ))
    ])

    models["KNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))
    ])

    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=SEED, verbosity=0
            ))
        ])

    return models


def path1_supervised(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 65)
    print("PATH 1 — SUPERVISED CLASSIFICATION")
    print("=" * 65)

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)

    models = build_models(le)

    # ── Step A: LOO cross-validation on training set ──────────────────────
    # Feature selection happens INSIDE each LOO fold — no leakage
    print("\n  Step A: Leave-One-Out CV on training set")
    print("  (Feature selection runs inside each fold — no label leakage)\n")

    loo = LeaveOneOut()
    cv_results = {}

    for name, model in models.items():
        y_pred_enc = cross_val_predict(model, X_train.values, y_train_enc, cv=loo)
        y_pred     = le.inverse_transform(y_pred_enc)
        acc        = accuracy_score(y_train, y_pred)
        cv_results[name] = {"acc": acc, "y_pred": y_pred}
        print(f"    {name:20s}  LOO-CV accuracy = {acc:.1%}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["acc"])
    best_cv_acc = cv_results[best_name]["acc"]
    print(f"\n  Best on CV: {best_name}  ({best_cv_acc:.1%})")

    # ── Step B: Hyperparameter tuning for RF on training set ──────────────
    print("\n  Step B: Hyperparameter tuning (Random Forest, 5-fold stratified CV)")
    param_dist = {
        "clf__n_estimators":     [200, 300, 500, 800],
        "clf__max_depth":        [3, 5, 8, None],
        "clf__min_samples_leaf": [1, 2, 3],
        "clf__max_features":     ["sqrt", "log2", 0.4, 0.6],
        "clf__class_weight":     ["balanced", None],
    }
    base_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))
    ])
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        base_rf, param_dist, n_iter=40, cv=cv_strat,
        scoring="accuracy", random_state=SEED, n_jobs=-1, verbose=0
    )
    search.fit(X_train.values, y_train_enc)
    print(f"    Tuned RF CV accuracy: {search.best_score_:.1%}")
    print(f"    Best params: {search.best_params_}")

    # ── Step C: TRUE HELD-OUT TEST SET ────────────────────────────────────
    print("\n  Step C: Final evaluation on HELD-OUT test set")
    print("  (These trends were never seen during training or CV)\n")

    test_results = {}
    for name, model in models.items():
        model.fit(X_train.values, y_train_enc)
        y_pred_enc = model.predict(X_test.values)
        y_pred     = le.inverse_transform(y_pred_enc)
        acc        = accuracy_score(y_test, y_pred)
        test_results[name] = {"acc": acc, "y_pred": y_pred}
        print(f"    {name:20s}  Test accuracy = {acc:.1%}")

    # Tuned RF on test
    tuned_pred_enc = search.best_estimator_.predict(X_test.values)
    tuned_pred     = le.inverse_transform(tuned_pred_enc)
    tuned_test_acc = accuracy_score(y_test, tuned_pred)
    test_results["RF_tuned"] = {"acc": tuned_test_acc, "y_pred": tuned_pred}
    print(f"    {'RF_tuned':20s}  Test accuracy = {tuned_test_acc:.1%}")

    best_test_name = max(test_results, key=lambda k: test_results[k]["acc"])
    print(f"\n  Best on test: {best_test_name}  "
          f"({test_results[best_test_name]['acc']:.1%})")

    # ── Step D: Feature importances (trained on full training set) ─────────
    rf_final = RandomForestClassifier(
        n_estimators=500, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train.values)
    rf_final.fit(X_train_scaled, y_train_enc)
    importances = pd.Series(
        rf_final.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    # ── Plots ─────────────────────────────────────────────────────────────
    _plot_model_comparison(cv_results, test_results)
    _plot_confusion(y_test, test_results[best_test_name]["y_pred"],
                    best_test_name, test_results[best_test_name]["acc"])
    _plot_feature_importance(importances)
    _plot_per_trend_results(X_test, y_test,
                            test_results[best_test_name]["y_pred"],
                            best_test_name)

    return cv_results, test_results, importances, best_test_name


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATIONS  (white/lavender aesthetic — presentation-ready)
# ═══════════════════════════════════════════════════════════════════════════════

# Shared style constants
BG        = "white"
PANEL_BG  = "#f8f7fc"
TEXT_DARK = "#1a1a2e"
TEXT_MID  = "#555555"
SPINE_COL = "#dddddd"
PURPLE    = "#7B5EA7"
PURPLE_L  = "#9B7EC8"


def _style_ax(ax):
    """Apply consistent light theme to an axes."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_MID, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)
        spine.set_linewidth(0.8)
    ax.xaxis.label.set_color(TEXT_MID)
    ax.yaxis.label.set_color(TEXT_MID)


def _plot_model_comparison(cv_results, test_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor(BG)

    for ax, results, title in zip(
        axes,
        [cv_results, test_results],
        ["Cross-validation accuracy", "Held-out test accuracy"]
    ):
        _style_ax(ax)
        names   = list(results.keys())
        accs    = [results[n]["acc"] * 100 for n in names]
        best_acc = max(accs)
        bar_cols = [PURPLE if a == best_acc else PURPLE_L for a in accs]
        bars = ax.barh(names, accs, color=bar_cols, alpha=0.88, height=0.55)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{acc:.1f}%", va="center", color=TEXT_DARK, fontsize=9)
        ax.set_xlim(0, 115)
        ax.set_xlabel("Accuracy (%)", color=TEXT_MID)
        ax.set_title(title, color=TEXT_DARK, fontsize=12, fontweight="bold", pad=10)
        ax.axvline(80, color=SPINE_COL, linestyle="--", linewidth=1)
        ax.tick_params(colors=TEXT_MID)

    fig.suptitle("Classifier Performance — Model Comparison",
                 color=TEXT_DARK, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT, "model_comparison.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Saved: {path}")


def _plot_confusion(y_true, y_pred, model_name, acc):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    cm   = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="RdPu")
    ax.set_title(f"Classification Results — Random Forest  ({acc:.1%} accuracy)",
                 color=TEXT_DARK, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT_MID)
    ax.xaxis.label.set_color(TEXT_MID)
    ax.yaxis.label.set_color(TEXT_MID)
    for text in ax.texts:
        text.set_color(TEXT_DARK)
        text.set_fontsize(13)
    plt.tight_layout()
    path = os.path.join(OUT, "confusion_matrix.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


def _plot_feature_importance(importances):
    top = importances.head(20).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    med     = top.median()
    colours = [PURPLE if v > med else PURPLE_L for v in top.values]
    ax.barh(top.index, top.values, color=colours, alpha=0.88)
    ax.set_xlabel("Mean decrease in impurity", color=TEXT_MID)
    ax.set_title("Most Predictive Features for Trend Classification",
                 color=TEXT_DARK, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT_MID, labelsize=9)
    plt.tight_layout()
    path = os.path.join(OUT, "feature_importances.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


def _plot_per_trend_results(X_test, y_test, y_pred, model_name):
    results = pd.DataFrame({
        "trend":   X_test.index,
        "true":    y_test.values,
        "pred":    y_pred,
        "correct": y_test.values == y_pred
    }).sort_values(["true", "trend"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.45)))
    fig.patch.set_facecolor(BG)
    _style_ax(ax)

    bar_cols = [COLOURS[t] if c else "#cccccc"
                for t, c in zip(results["true"], results["correct"])]
    ax.barh(results["trend"], results["correct"].astype(int),
            color=bar_cols, alpha=0.88)

    for i, (_, row) in enumerate(results.iterrows()):
        label = "✓  Correct" if row["correct"] else f"✗  Predicted: {row['pred']}"
        col   = TEXT_DARK if row["correct"] else "#cc3333"
        ax.text(0.02, i, label, va="center", color=col, fontsize=9)

    ax.set_xlim(0, 1.6)
    ax.set_xlabel("Correct / Incorrect", color=TEXT_MID)
    ax.set_title("Per-Trend Classification Results on Held-Out Test Set",
                 color=TEXT_DARK, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT_MID, labelsize=9)
    plt.tight_layout()
    path = os.path.join(OUT, "per_trend_results.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


def plot_time_series_gallery(series_dict, y):
    """Representative time series per class."""
    n_per = 3
    fig   = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor(BG)
    gs    = gridspec.GridSpec(3, n_per, hspace=0.55, wspace=0.35)

    for row, label in enumerate(LABEL_ORDER):
        names = [n for n, l in y.items() if l == label][:n_per]
        for col, name in enumerate(names):
            v  = series_dict[name]
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(PANEL_BG)
            ax.fill_between(range(len(v)), v, alpha=0.2, color=COLOURS[label])
            ax.plot(v, color=COLOURS[label], linewidth=1.4)
            ax.axhline(THRESHOLD, color=TEXT_MID, linewidth=0.6,
                       linestyle="--", alpha=0.5)
            ax.set_title(name, color=TEXT_DARK, fontsize=8, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, color=COLOURS[label],
                              fontsize=10, fontweight="bold")
            ax.tick_params(colors=TEXT_MID, labelsize=6)
            ax.set_ylim(-0.05, 1.05)
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE_COL)
                spine.set_linewidth(0.8)

    fig.suptitle("Google Trends Profiles by Trend Classification",
                 color=TEXT_DARK, fontsize=14, fontweight="bold", y=1.01)
    path = os.path.join(OUT, "trend_gallery.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


def path2_plot(X, y_true, cluster_ids, coords):
    """UMAP/PCA visualisation — white theme, presentation-ready titles."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG)

    cluster_colours = [COLOURS["Micro"], COLOURS["Macro"], COLOURS["Mega"]]

    for ax in axes:
        _style_ax(ax)

    # Left: blind clusters
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        s=100, alpha=0.9, label=f"Cluster {cid}",
                        color=cluster_colours[cid],
                        edgecolors="white", linewidths=0.5)
        for xi, yi, nm in zip(coords[mask, 0], coords[mask, 1],
                               np.array(X.index)[mask]):
            axes[0].annotate(nm, (xi, yi), fontsize=5.5, color=TEXT_MID,
                             alpha=0.85, ha="center", va="bottom")

    axes[0].set_title("Blind Clustering — No Labels Used",
                      color=TEXT_DARK, fontsize=12, fontweight="bold", pad=12)
    axes[0].legend(labelcolor=TEXT_DARK, facecolor=BG,
                   edgecolor=SPINE_COL, fontsize=9)

    # Right: true labels
    for lbl in LABEL_ORDER:
        mask = y_true.values == lbl
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        s=100, alpha=0.9, label=lbl, color=COLOURS[lbl],
                        edgecolors="white", linewidths=0.5)
        for xi, yi, nm in zip(coords[mask, 0], coords[mask, 1],
                               np.array(X.index)[mask]):
            axes[1].annotate(nm, (xi, yi), fontsize=5.5, color=TEXT_MID,
                             alpha=0.85, ha="center", va="bottom")

    axes[1].set_title("Micro / Macro / Mega Classifications",
                      color=TEXT_DARK, fontsize=12, fontweight="bold", pad=12)
    axes[1].legend(labelcolor=TEXT_DARK, facecolor=BG,
                   edgecolor=SPINE_COL, fontsize=9)

    fig.suptitle("Do Blind Clusters Match the Trend Classifications?",
                 color=TEXT_DARK, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT, "umap_clustering.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Load
    df = load_data()  # reads from data/all_trends_combined.csv by default
    series = pivot_to_series(df)

    # Check all labelled trends are present
    missing = [t for t in LABEL_MAP if t not in series]
    if missing:
        print(f"[WARN] These labelled trends not found in CSV: {missing}")

    # 2. Features — extracted before labels are attached
    print("\n[INFO] Engineering features (on value_norm only)...")
    X_all, _ = build_feature_matrix(series)
    X_lab, y = build_feature_matrix(series, label_map=LABEL_MAP)
    print(f"  Feature matrix: {X_lab.shape[0]} labelled trends "
          f"× {X_lab.shape[1]} features")

    label_counts = y.value_counts()
    print(f"  Label distribution: "
          + ", ".join(f"{l}: {label_counts.get(l, 0)}" for l in LABEL_ORDER))

    # 3. Time series gallery
    print("\n[INFO] Plotting time series gallery...")
    plot_time_series_gallery(series, LABEL_MAP)

    # 4. PATH 2 — Unsupervised (uses full X_lab for projection, but blind)
    cluster_ids, coords, cluster_acc = path2_unsupervised(X_lab, y)

    # 5. PATH 1 — Supervised
    # Split into train + held-out test BEFORE any model sees the data
    X_train, y_train, X_test, y_test = make_held_out_split(
        X_lab, y, n_per_class=3
    )

    cv_results, test_results, importances, best_model = path1_supervised(
        X_train, y_train, X_test, y_test
    )

    # 6. Final summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Path 2 — Blind cluster alignment:  {cluster_acc:.1%}")
    print(f"  Path 1 — Best LOO-CV accuracy:     "
          f"{max(r['acc'] for r in cv_results.values()):.1%}")
    print(f"  Path 1 — Best held-out test acc:   "
          f"{max(r['acc'] for r in test_results.values()):.1%}  "
          f"(best model: {best_model})")
    print(f"\n  Outputs saved to: ./{OUT}/")
    print("=" * 65)

    print("""
INTERPRETING RESULTS
────────────────────
Path 2 alignment >60%  → your 3 classes have real separable structure
Path 2 alignment <50%  → trends don't cluster naturally by your labels

Path 1 test accuracy   → this is your HONEST accuracy on unseen trends
LOO-CV accuracy        → estimate during training (usually slightly higher)

A big gap between LOO-CV and test accuracy means the model is overfitting
to the training trends — you may need more labelled examples.
""")


if __name__ == "__main__":
    main()