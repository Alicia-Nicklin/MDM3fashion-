import os
import glob
import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================
DATA_FOLDER = r"data"
GOOGLE_TRENDS_FOLDER = "google_trends_data"
OUTPUT_FILE = "trend_validation_results.csv"

THRESHOLD = 0.35       # lowered from 0.50 — captures sustained interest, not just the sharp peak
# Classification boundaries: Micro < 6m, Macro 6–36m, Mega > 36m
# Adjusted from original (9/24) to match where the data actually clusters
GAP_TOLERANCE = 1

DATE_COL_CANDIDATES = ["Month", "Date", "date", "month", "Week", "Time"]


# =========================================================
# CLASSIFICATION RULES
# =========================================================
def classify_trend(months):
    if months < 6:
        return "Micro"
    elif months <= 36:
        return "Macro"
    else:
        return "Mega"


# =========================================================
# HELPERS
# =========================================================
def find_date_column(df):
    for col in DATE_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def find_value_column(df, date_col):
    """
    Returns the first column that is numeric (or can be coerced to numeric),
    excluding the date column and any boolean/flag columns like 'isPartial'.
    Previously this just grabbed the first non-date column, which could
    silently pick up metadata columns and produce wrong results.
    """
    for col in df.columns:
        if col == date_col:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        # Require that at least half the rows are valid numbers
        if coerced.notna().sum() >= len(df) * 0.5:
            return col
    return None


def load_google_trends_csv(file_path):
    """
    Loads a Google Trends CSV and returns dataframe with:
    columns = ['date', 'value']
    """
    try:
        df = pd.read_csv(file_path)
        date_col = find_date_column(df)

        # Some Google Trends exports need skiprows=1
        if date_col is None:
            df = pd.read_csv(file_path, skiprows=1)
            date_col = find_date_column(df)

        if date_col is None:
            raise ValueError("No date column found")

        value_col = find_value_column(df, date_col)
        if value_col is None:
            raise ValueError("No numeric value column found")

        df = df[[date_col, value_col]].copy()
        df.columns = ["date", "value"]

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid rows after cleaning")

        return df

    except Exception as e:
        raise ValueError(f"Error reading {os.path.basename(file_path)}: {e}")


def normalise_0_1(series):
    s = series.astype(float)
    s_min = s.min()
    s_max = s.max()

    if s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - s_min) / (s_max - s_min)


def fill_small_gaps(active_mask, gap_tolerance=1):
    """
    Fill small False gaps inside True runs.
    Example:
    [True, True, False, True] -> [True, True, True, True]
    if gap_tolerance >= 1
    """
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
    """
    Returns:
    longest_length, start_index, end_index
    """
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


# =========================================================
# ANALYSIS PER TREND
# =========================================================
def analyse_trend(file_path):
    df = load_google_trends_csv(file_path)

    df["value_norm"] = normalise_0_1(df["value"])

    active_mask = (df["value_norm"] >= THRESHOLD).tolist()
    active_mask_filled = fill_small_gaps(active_mask, gap_tolerance=GAP_TOLERANCE)

    longest_points, start_idx, end_idx = longest_true_run(active_mask_filled)

    if start_idx is not None and end_idx is not None:
        start_date = df.loc[start_idx, "date"]
        end_date = df.loc[end_idx, "date"]

        duration_days = (end_date - start_date).days
        duration_months = max(1, int(round(duration_days / 30.44)) + 1)
    else:
        start_date = pd.NaT
        end_date = pd.NaT
        duration_months = 0

    computed_label = classify_trend(duration_months)

    return {
        "trend_name": os.path.splitext(os.path.basename(file_path))[0],
        "num_points": len(df),
        "max_raw_value": float(df["value"].max()),
        "active_points_total": int(sum(active_mask_filled)),
        "main_trend_duration_months": int(duration_months),
        "main_trend_start": start_date,
        "main_trend_end": end_date,
        "computed_label": computed_label,
        "file_path": file_path
    }


# =========================================================
# MAIN
# =========================================================
def main():
    files = (
            glob.glob(os.path.join(DATA_FOLDER, "Macrotrends", "*.csv")) +
            glob.glob(os.path.join(DATA_FOLDER, "MegaTrends", "*.csv")) +
            glob.glob(os.path.join(DATA_FOLDER, "Microtrends", "*.csv"))
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: {os.path.join(DATA_FOLDER, GOOGLE_TRENDS_FOLDER)}"
        )

    results = []
    failed = []

    print(f"Found {len(files)} CSV files.\n")

    for file_path in files:
        try:
            result = analyse_trend(file_path)
            results.append(result)

            print(
                f"{result['trend_name']}: "
                f"computed={result['computed_label']}, "
                f"duration={result['main_trend_duration_months']} months"
            )

        except Exception as e:
            failed.append({
                "file": os.path.basename(file_path),
                "error": str(e)
            })
            print(f"FAILED: {os.path.basename(file_path)} -> {e}")

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        category_order = pd.CategoricalDtype(
            categories=["Micro", "Macro", "Mega"],
            ordered=True
        )
        results_df["computed_label"] = results_df["computed_label"].astype(category_order)

        results_df = results_df.sort_values(
            by=["computed_label", "main_trend_duration_months"],
            ascending=[True, False]
        )

    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to: {OUTPUT_FILE}")

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv("trend_validation_failed.csv", index=False)
        print("Some files failed. Saved errors to: trend_validation_failed.csv")


if __name__ == "__main__":
    main()