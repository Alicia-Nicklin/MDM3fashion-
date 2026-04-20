import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.style.use("seaborn-v0_8-whitegrid")

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(BASE_DIR, "data", "all_trends_with_classes.csv")

THRESHOLD = 0.35
GAP_TOLERANCE = 1
RESAMPLE_POINTS = 100

# Set to None to auto-pick a representative trend
EXAMPLE_TREND_NAME = None
EXAMPLE_CLASS = "Macro"   # "Micro", "Macro", or "Mega"

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# COLOURS
# =========================================================
PURPLE = "#7B2CBF"
LIGHT_PURPLE = "#CDB4F8"
PINK = "#FF4D6D"
DARK_PURPLE = "#5A189A"

# =========================================================
# HELPERS
# =========================================================
def fill_small_gaps(active_mask, gap_tolerance=1):
    active_mask = list(active_mask)
    n = len(active_mask)
    i = 0

    while i < n:
        if not active_mask[i]:
            start = i
            while i < n and not active_mask[i]:
                i += 1
            end = i - 1
            gap_len = end - start + 1

            left_active = start > 0 and active_mask[start - 1]
            right_active = end < n - 1 and active_mask[end + 1]

            if left_active and right_active and gap_len <= gap_tolerance:
                for j in range(start, end + 1):
                    active_mask[j] = True
        else:
            i += 1

    return active_mask


def longest_true_run(mask):
    max_len = 0
    max_start = None
    max_end = None

    current_len = 0
    current_start = None

    for i, val in enumerate(mask):
        if val:
            if current_len == 0:
                current_start = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
                max_end = i - 1
            current_len = 0
            current_start = None

    if current_len > max_len:
        max_len = current_len
        max_start = current_start
        max_end = len(mask) - 1

    return max_len, max_start, max_end


def prepare_df(df):
    df = df.copy()

    cols_lower = {c.lower(): c for c in df.columns}

    if "date" not in df.columns and "date" in cols_lower:
        df = df.rename(columns={cols_lower["date"]: "date"})
    if "trend_name" not in df.columns and "trend_name" in cols_lower:
        df = df.rename(columns={cols_lower["trend_name"]: "trend_name"})
    if "value_norm" not in df.columns and "value_norm" in cols_lower:
        df = df.rename(columns={cols_lower["value_norm"]: "value_norm"})

    if "class" not in df.columns:
        if "trend_class" in df.columns:
            df = df.rename(columns={"trend_class": "class"})
        elif "computed_label" in df.columns:
            df = df.rename(columns={"computed_label": "class"})
        elif "label" in df.columns:
            df = df.rename(columns={"label": "class"})

    required = ["date", "trend_name", "value_norm", "class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value_norm"] = pd.to_numeric(df["value_norm"], errors="coerce")
    df["trend_name"] = df["trend_name"].astype(str)
    df["class"] = df["class"].astype(str).str.strip().str.title()

    df = df.dropna(subset=["date", "value_norm", "trend_name", "class"])
    df = df.sort_values(["class", "trend_name", "date"]).reset_index(drop=True)

    return df


def get_active_window(df_trend):
    values = df_trend["value_norm"].values
    active_mask = (values >= THRESHOLD).tolist()
    active_mask_filled = fill_small_gaps(active_mask, gap_tolerance=GAP_TOLERANCE)

    longest_points, start_idx, end_idx = longest_true_run(active_mask_filled)

    if start_idx is None or end_idx is None or longest_points < 2:
        return None, None, None, None

    clipped = df_trend.iloc[start_idx:end_idx + 1].copy()
    return clipped, start_idx, end_idx, longest_points


def resample_curve(values, n_points=100):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return None

    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, n_points)

    f = interp1d(x_old, values, kind="linear")
    return f(x_new)


def build_class_curves(df_class):
    curves = []
    kept_names = []
    durations = {}

    for trend_name, group in df_class.groupby("trend_name"):
        group = group.sort_values("date").reset_index(drop=True)
        clipped, start_idx, end_idx, longest_points = get_active_window(group)

        if clipped is None:
            continue

        curve = resample_curve(clipped["value_norm"].values, n_points=RESAMPLE_POINTS)
        if curve is None:
            continue

        start_date = clipped["date"].iloc[0]
        end_date = clipped["date"].iloc[-1]
        duration_days = (end_date - start_date).days
        duration_months = max(1, int(round(duration_days / 30.44)) + 1)

        curves.append(curve)
        kept_names.append(trend_name)
        durations[trend_name] = duration_months

    if len(curves) == 0:
        return None, [], {}

    return np.vstack(curves), kept_names, durations


def auto_pick_example(df_class):
    curves, names, durations = build_class_curves(df_class)
    if not names:
        return None

    duration_series = pd.Series(durations)
    median_duration = duration_series.median()
    best_name = (duration_series - median_duration).abs().sort_values().index[0]
    return best_name


def get_class_df(class_name, all_df):
    class_name = class_name.strip().title()
    return all_df[all_df["class"] == class_name].copy()


# =========================================================
# LOAD DATA
# =========================================================
all_df = pd.read_csv(FILE)
all_df = prepare_df(all_df)

print("Columns found:", list(all_df.columns))
print("Classes found:", sorted(all_df["class"].unique()))
print("Number of unique trends:", all_df["trend_name"].nunique())

micro_df = get_class_df("Micro", all_df)
macro_df = get_class_df("Macro", all_df)
mega_df = get_class_df("Mega", all_df)

# =========================================================
# BUILD CURVES
# =========================================================
micro_curves, micro_names, micro_durations = build_class_curves(micro_df)
macro_curves, macro_names, macro_durations = build_class_curves(macro_df)
mega_curves, mega_names, mega_durations = build_class_curves(mega_df)

# =========================================================
# PLOT 1: CLASS SHAPE COMPARISON
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

class_data = [
    ("Micro Trends", micro_curves),
    ("Macro Trends", macro_curves),
    ("Mega Trends", mega_curves),
]

x = np.linspace(0, 100, RESAMPLE_POINTS)

for ax, (title, curves) in zip(axes, class_data):
    if curves is None or len(curves) == 0:
        ax.set_title(f"{title}\n(no valid trends)")
        ax.grid(alpha=0.25)
        continue

    for curve in curves:
        ax.plot(x, curve, color=LIGHT_PURPLE, alpha=0.08, linewidth=1)

    mean_curve = curves.mean(axis=0)
    ax.plot(x, mean_curve, color=PURPLE, linewidth=4)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("% of Active Duration")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

axes[0].set_ylabel("Normalised Interest")

plt.suptitle(
    "Trend Class Shape Comparison\n(Clipped to Active Window and Rescaled)",
    fontsize=15,
    fontweight="bold"
)
plt.tight_layout()

class_plot_path = os.path.join(OUTPUT_DIR, "class_shape_comparison.png")
plt.savefig(class_plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {class_plot_path}")

# =========================================================
# CHOOSE EXAMPLE TREND
# =========================================================
example_df_class = get_class_df(EXAMPLE_CLASS, all_df)

if EXAMPLE_TREND_NAME is None:
    EXAMPLE_TREND_NAME = auto_pick_example(example_df_class)

if EXAMPLE_TREND_NAME is None:
    raise ValueError(f"No valid example trend found for class: {EXAMPLE_CLASS}")

print(f"Example trend selected: {EXAMPLE_TREND_NAME} ({EXAMPLE_CLASS})")

example_trend_df = example_df_class[example_df_class["trend_name"] == EXAMPLE_TREND_NAME].copy()
example_trend_df = example_trend_df.sort_values("date").reset_index(drop=True)

clipped_example, start_idx, end_idx, longest_points = get_active_window(example_trend_df)

if clipped_example is None:
    raise ValueError(f"Could not find valid active window for trend: {EXAMPLE_TREND_NAME}")

start_date = clipped_example["date"].iloc[0]
end_date = clipped_example["date"].iloc[-1]
duration_days = (end_date - start_date).days
duration_months = max(1, int(round(duration_days / 30.44)) + 1)

# =========================================================
# PLOT 2: EXAMPLE THRESHOLD PLOT
# =========================================================
plt.figure(figsize=(10, 5.5))

plt.plot(
    example_trend_df["date"],
    example_trend_df["value_norm"],
    color="black",
    linewidth=2.5,
    label="Normalised Interest"
)

plt.axhline(
    THRESHOLD,
    color=PINK,
    linestyle="--",
    linewidth=2,
    label="35% of Peak (Activity Threshold)"
)

mask = np.zeros(len(example_trend_df), dtype=bool)
mask[start_idx:end_idx + 1] = True

plt.fill_between(
    example_trend_df["date"],
    example_trend_df["value_norm"],
    where=mask,
    alpha=0.35,
    color=LIGHT_PURPLE,
    label="Active Window"
)

plt.axvline(start_date, color=PINK, linestyle="--", linewidth=1.5)
plt.axvline(end_date, color=PINK, linestyle="--", linewidth=1.5)

plt.text(start_date, 0.9, "Start", color=PINK, ha="left", va="top", fontsize=10)
plt.text(end_date, 0.9, "End", color=PINK, ha="right", va="top", fontsize=10)

plt.title(
    f"Example Threshold Plot: {EXAMPLE_TREND_NAME}\n"
    f"Active Duration = {duration_months} months",
    fontsize=16,
    fontweight="bold"
)
plt.xlabel("Date")
plt.ylabel("Normalised Interest")
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

threshold_plot_path = os.path.join(OUTPUT_DIR, "example_threshold_plot.png")
plt.savefig(threshold_plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {threshold_plot_path}")