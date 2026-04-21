"""
ABM Figures — presentation + appendix

Generates exactly 3 figures:
  1. abm_representative_fits.png       ← 3-panel combined (slide)
  2. abm_rmse_by_category.png          ← RMSE boxplot (slide)
  3. abm_ci_coverage_by_category.png   ← CI coverage bar chart (appendix)

Requires:
  - output/abm_summary.csv
  - output/sir_parameters.csv
  - output/abm_optuna_best_params.csv
  - data/Microtrends/officesiren.csv
  - data/Macrotrends/cottagecore.csv
  - data/MegaTrends/skaterfashion.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ── Palette ──────────────────────────────────────────────────────────
ACCENT_DARK  = "#2D1B3D"
VIOLET       = "#7B3FA0"
PINK         = "#E8609A"
PURPLE_LIGHT = "#D8C8EE"
GRID_COLOR   = "#EDE3F7"

CAT_COLORS = {
    "Microtrends": "#E8609A",
    "Macrotrends": "#7B3FA0",
    "MegaTrends":  "#B07FCC",
}

plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     PURPLE_LIGHT,
    "axes.labelcolor":    ACCENT_DARK,
    "axes.grid":          True,
    "grid.color":         GRID_COLOR,
    "grid.alpha":         0.6,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.color":        ACCENT_DARK,
    "ytick.color":        ACCENT_DARK,
    "text.color":         ACCENT_DARK,
    "font.family":        "serif",
    "font.size":          10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   PURPLE_LIGHT,
    "legend.facecolor":   "white",
})

CAT_ORDER  = ["Microtrends", "Macrotrends", "MegaTrends"]
CAT_LABELS = {"Microtrends": "Micro", "Macrotrends": "Macro", "MegaTrends": "Mega"}

ACTIVE_THRESHOLD = 0.35
GAP_TOLERANCE    = 1
BUFFER           = 5
N_PEOPLE         = 1000
N_ABM_RUNS       = 10

SEARCH_DIRS = {
    "Microtrends": os.path.join(DATA_DIR, "Microtrends"),
    "Macrotrends": os.path.join(DATA_DIR, "Macrotrends"),
    "MegaTrends":  os.path.join(DATA_DIR, "MegaTrends"),
}

# ── Load CSVs ────────────────────────────────────────────────────────
for p in [
    os.path.join(OUTPUT_DIR, "abm_summary.csv"),
    os.path.join(OUTPUT_DIR, "abm_optuna_best_params.csv"),
    os.path.join(OUTPUT_DIR, "sir_parameters.csv"),
]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

df        = pd.read_csv(os.path.join(OUTPUT_DIR, "abm_summary.csv"))
df["abm_corr"]    = pd.to_numeric(df["abm_corr"],    errors="coerce")
df["ci_coverage"] = pd.to_numeric(df["ci_coverage"], errors="coerce")

params_df   = pd.read_csv(os.path.join(OUTPUT_DIR, "abm_optuna_best_params.csv"))
BEST_PARAMS = dict(zip(params_df["param"], params_df["value"]))

sir_df = pd.read_csv(os.path.join(OUTPUT_DIR, "sir_parameters.csv"))

# ── Helpers ──────────────────────────────────────────────────────────

def safe_normalise(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

def normalise_name(s):
    return str(s).lower().replace("_","").replace("-","").replace(" ","").strip()

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

def find_trend_file(name):
    target = normalise_name(name)
    for folder, folder_path in SEARCH_DIRS.items():
        if not os.path.exists(folder_path):
            continue
        for path in glob.glob(os.path.join(folder_path, "*.csv")):
            stem = os.path.splitext(os.path.basename(path))[0]
            if "combined" in stem.lower():
                continue
            if normalise_name(stem) == target:
                return path
    return None

def solve_sir_I(beta, gamma, I0, n_periods, sub=4):
    dt = 1.0 / sub
    h6, h2 = dt / 6.0, dt / 2.0
    S, I = max(1.0 - I0, 1e-9), float(I0)
    I_out = np.empty(n_periods)
    I_out[0] = I
    out_idx = 1
    for step in range(1, (n_periods - 1) * sub + 1):
        bSI = beta * S * I
        k1s, k1i = -bSI, bSI - gamma * I
        S2, I2 = S + h2*k1s, I + h2*k1i
        bSI = beta * S2 * I2
        k2s, k2i = -bSI, bSI - gamma*I2
        S3, I3 = S + h2*k2s, I + h2*k2i
        bSI = beta * S3 * I3
        k3s, k3i = -bSI, bSI - gamma*I3
        S4, I4 = S + dt*k3s, I + dt*k3i
        bSI = beta * S4 * I4
        k4s, k4i = -bSI, bSI - gamma*I4
        S = max(S + h6*(k1s+2*k2s+2*k3s+k4s), 0.0)
        I = max(I + h6*(k1i+2*k2i+2*k3i+k4i), 0.0)
        if step % sub == 0 and out_idx < n_periods:
            I_out[out_idx] = I
            out_idx += 1
    return I_out

def simulate_trend(wave, sir_I, n_steps, params, start_delay=0, seed=None):
    rng       = np.random.default_rng(seed)
    peak_step = int(np.argmax(sir_I))
    state     = np.zeros(N_PEOPLE, dtype=int)
    state[:max(1, int(params["start_share"] * N_PEOPLE))] = 1
    rng.shuffle(state)
    uptake_w = rng.uniform(0.9, 1.1, N_PEOPLE)
    fade_w   = rng.uniform(0.9, 1.1, N_PEOPLE)
    series   = np.zeros(n_steps)
    for i in range(n_steps):
        if i < start_delay:
            continue
        share      = state.mean()
        join_rate  = (params["start_chance"] + params["social_weight"] * (share ** 1.1)) * wave[i]
        join_rate += rng.normal(0, params["noise"])
        if i > peak_step:
            join_rate *= params["post_peak_suppression"]
        extra_fade = (
            params["late_drop"] * (i - peak_step) / max(1, n_steps - peak_step)
            if i > peak_step else 0.0
        )
        leave_rate = params["fade_base"] + extra_fade + rng.normal(0, params["noise"] / 2)
        join_prob  = np.clip(join_rate  * uptake_w, 0, 0.95)
        leave_prob = np.clip(leave_rate * fade_w,   0, 0.95)
        rj, rl = rng.random(N_PEOPLE), rng.random(N_PEOPLE)
        state[(state == 0) & (rj < join_prob)]  = 1
        state[(state == 1) & (rl < leave_prob)] = 0
        series[i] = state.mean()
    if series.max() > 0:
        series /= series.max()
    return series

def build_trend_data(trend_name):
    sir_mask = sir_df["trend"].astype(str).apply(normalise_name) == normalise_name(trend_name)
    sir_row = sir_df[sir_mask]
    if sir_row.empty:
        raise ValueError(f"'{trend_name}' not in sir_parameters.csv")
    sir_row = sir_row.iloc[0]
    beta, gamma, I0 = float(sir_row["beta"]), float(sir_row["gamma"]), float(sir_row["I0"])

    csv_path = find_trend_file(trend_name)
    if csv_path is None:
        raise FileNotFoundError(f"Raw CSV not found for '{trend_name}'")

    raw  = pd.read_csv(csv_path)
    real = pd.to_numeric(raw.iloc[:, 1], errors="coerce").fillna(0).values.astype(float)
    real = safe_normalise(real)

    active_mask           = _fill_gaps(real >= ACTIVE_THRESHOLD)
    _, run_start, run_end = _longest_run(active_mask)
    start_idx   = max(0, run_start - BUFFER)
    end_idx     = min(len(real) - 1, run_end + BUFFER)
    n_steps     = end_idx - start_idx + 1
    start_delay = max(0, (run_start - start_idx) - 2)

    sir_I     = solve_sir_I(beta, gamma, I0, n_steps)
    wave      = safe_normalise(sir_I)
    real_clip = real[start_idx:end_idx + 1]

    runs = [
        simulate_trend(wave, sir_I, n_steps, BEST_PARAMS, start_delay, seed=42 + k)
        for k in range(N_ABM_RUNS)
    ]
    mean_sim = np.mean(runs, axis=0)

    return dict(real_clip=real_clip, mean_sim=mean_sim,
                n_steps=n_steps, start_idx=start_idx)

# ====================================================================
# FIGURE 1 — Representative ABM fits  (drawn from scratch, no PNG crop)
# ====================================================================

REPRESENTATIVES = [
    {"trend_name": "officesiren",   "title": "Office Siren",   "cat_key": "Microtrends"},
    {"trend_name": "cottagecore",   "title": "Cottagecore",    "cat_key": "Macrotrends"},
    {"trend_name": "skaterfashion", "title": "Skater Fashion", "cat_key": "MegaTrends"},
]

fig1, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")
fig1.subplots_adjust(left=0.06, right=0.97, top=0.78, bottom=0.18, wspace=0.30)

for ax, item in zip(axes, REPRESENTATIVES):
    data = build_trend_data(item["trend_name"])
    col  = CAT_COLORS[item["cat_key"]]
    x    = np.arange(data["start_idx"], data["start_idx"] + data["n_steps"])

    ax.plot(x, data["real_clip"], color=col,   lw=2.2, label="Real trend")
    ax.plot(x, data["mean_sim"],  color=VIOLET, lw=1.8, linestyle="--", label="ABM mean")
    ax.axhline(ACTIVE_THRESHOLD, color=PURPLE_LIGHT, linestyle=":", lw=1.2)

    ax.set_facecolor("white")
    ax.set_ylim(-0.04, 1.15)
    ax.set_xlabel("Month", labelpad=5, fontsize=9)
    ax.set_ylabel("Normalised interest", labelpad=5, fontsize=9)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PURPLE_LIGHT)
        ax.spines[spine].set_linewidth(1.4)

    mask  = df["trend"].str.lower().str.replace(" ", "", regex=False).str.contains(
        item["trend_name"].lower(), na=False)
    stats = df[mask].iloc[0] if mask.any() else None
    metric = (f"RMSE {stats['abm_rmse']:.3f}  ·  corr {stats['abm_corr']:.2f}"
              if stats is not None else "")

    ax.set_title(item["title"], fontsize=13, fontweight="bold",
                 color=ACCENT_DARK, pad=8)
    ax.text(0.5, -0.22, f"{CAT_LABELS[item['cat_key']]}  |  {metric}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=VIOLET)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

fig1.suptitle("Representative ABM fits by trend category",
              fontsize=16, fontweight="bold", color=ACCENT_DARK, y=0.96)

out1 = os.path.join(OUTPUT_DIR, "abm_representative_fits.png")
fig1.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print(f"Saved: {out1}")

# ====================================================================
# FIGURE 2 — RMSE by category
# ====================================================================

fig2, ax = plt.subplots(figsize=(7, 5), facecolor="white")
ax.set_facecolor("white")

rmse_groups = [df.loc[df["category"] == cat, "abm_rmse"].dropna().values for cat in CAT_ORDER]

bp = ax.boxplot(
    rmse_groups, patch_artist=True, widths=0.45,
    medianprops=dict(color=ACCENT_DARK, lw=2.2),
    whiskerprops=dict(color=ACCENT_DARK, lw=1.4),
    capprops=dict(color=ACCENT_DARK, lw=1.4),
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
)
for patch, cat in zip(bp["boxes"], CAT_ORDER):
    patch.set_facecolor(CAT_COLORS[cat]); patch.set_alpha(0.55)
    patch.set_edgecolor(ACCENT_DARK);     patch.set_linewidth(1.5)
for flier, cat in zip(bp["fliers"], CAT_ORDER):
    flier.set_markerfacecolor(CAT_COLORS[cat])
    flier.set_markeredgecolor(ACCENT_DARK)

jitter_rng = np.random.default_rng(0)
for j, (group, cat) in enumerate(zip(rmse_groups, CAT_ORDER), start=1):
    jitter = jitter_rng.uniform(-0.12, 0.12, len(group))
    ax.scatter(j + jitter, group, color=CAT_COLORS[cat], s=28, alpha=0.7,
               zorder=3, edgecolors=ACCENT_DARK, linewidths=0.4)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels([CAT_LABELS[c] for c in CAT_ORDER], fontsize=12)
ax.set_ylabel("RMSE (lower = better fit)", labelpad=8)
ax.set_title("ABM fit quality by trend category",
             fontsize=14, fontweight="bold", color=ACCENT_DARK, pad=12)
ax.text(0.5, -0.13,
        "Macrotrends fit best overall · Microtrends fit worst · Megatrends sit in between",
        transform=ax.transAxes, ha="center", fontsize=9.5, color=VIOLET, style="italic")
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(PURPLE_LIGHT); ax.spines[spine].set_linewidth(1.5)

fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "abm_rmse_by_category.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"Saved: {out2}")

# ====================================================================
# FIGURE 3 — CI coverage by category
# ====================================================================

fig3, ax = plt.subplots(figsize=(6.5, 4.5), facecolor="white")
ax.set_facecolor("white")

cov_means = [df.loc[df["category"] == cat, "ci_coverage"].dropna().mean() for cat in CAT_ORDER]

bars = ax.bar([CAT_LABELS[c] for c in CAT_ORDER], [v * 100 for v in cov_means],
              color=[CAT_COLORS[c] for c in CAT_ORDER],
              alpha=0.70, edgecolor=ACCENT_DARK, linewidth=1.4, width=0.48, zorder=3)

for bar, val in zip(bars, cov_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{val:.0%}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=ACCENT_DARK)

ax.axhline(95, color=PINK, linestyle="--", lw=1.4, alpha=0.6, label="95% ideal coverage")
ax.set_ylabel("Mean CI coverage (%)", labelpad=8)
ax.set_ylim(0, 110)
ax.set_title("95% simulation interval coverage by category",
             fontsize=13, fontweight="bold", color=ACCENT_DARK, pad=12)
ax.text(0.5, -0.13,
        "Coverage = fraction of real data points falling inside the 95% simulation band",
        transform=ax.transAxes, ha="center", fontsize=9, color=VIOLET, style="italic")
ax.legend(fontsize=9, loc="upper right")
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(PURPLE_LIGHT); ax.spines[spine].set_linewidth(1.5)

fig3.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "abm_ci_coverage_by_category.png")
fig3.savefig(out3, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print(f"Saved: {out3}")

print("\nAll done.")
