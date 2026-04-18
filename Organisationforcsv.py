import os
import glob
import pandas as pd
import numpy as np

# =========================================================
# CONFIG
# =========================================================
DATA_FOLDER = "data"
TARGET_FOLDERS = ["Microtrends", "Macrotrends", "MegaTrends"]

DATE_COL_CANDIDATES = ["Month", "Date", "date", "month", "Week", "Time"]


# =========================================================
# HELPERS
# =========================================================
def find_date_column(df):
    for col in DATE_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def find_value_column(df, date_col):
    for col in df.columns:
        if col == date_col:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() >= len(df) * 0.5:
            return col
    return None


def normalise_0_1(series):
    s = series.astype(float)
    s_min = s.min()
    s_max = s.max()

    if s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - s_min) / (s_max - s_min)


def load_trend(file_path):
    df = pd.read_csv(file_path)

    date_col = find_date_column(df)
    if date_col is None:
        df = pd.read_csv(file_path, skiprows=1)
        date_col = find_date_column(df)

    if date_col is None:
        raise ValueError(f"No date column found in {file_path}")

    value_col = find_value_column(df, date_col)
    if value_col is None:
        raise ValueError(f"No numeric value column found in {file_path}")

    df = df[[date_col, value_col]].copy()
    df.columns = ["date", "value_raw"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value_raw"] = pd.to_numeric(df["value_raw"], errors="coerce")

    df = df.dropna(subset=["date", "value_raw"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No valid rows after cleaning in {file_path}")

    df["value_norm"] = normalise_0_1(df["value_raw"])
    df["trend_name"] = os.path.splitext(os.path.basename(file_path))[0]

    return df


# =========================================================
# COMBINE ONE FOLDER
# =========================================================
def combine_folder(folder_name):
    folder_path = os.path.join(DATA_FOLDER, folder_name)

    if not os.path.exists(folder_path):
        print(f"SKIPPED: folder does not exist -> {folder_path}")
        return

    files = [
        f for f in glob.glob(os.path.join(folder_path, "*.csv"))
        if "combined" not in os.path.basename(f).lower()
    ]

    all_data = []

    print(f"\nProcessing {folder_name}... ({len(files)} raw CSV files found)")

    for f in files:
        try:
            df = load_trend(f)
            all_data.append(df)
            print(f"  loaded: {os.path.basename(f)}")
        except Exception as e:
            print(f"  FAILED: {os.path.basename(f)} -> {e}")

    if not all_data:
        print(f"No valid trend files found in {folder_name}")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    output_name = f"{folder_name.lower()}_combined.csv"
    output_path = os.path.join(folder_path, output_name)

    combined_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Unique trends: {combined_df['trend_name'].nunique()}")

    print("\nMax value_norm by trend:")
    print(combined_df.groupby("trend_name")["value_norm"].max().sort_values())


# =========================================================
# MAIN
# =========================================================
def main():
    for folder_name in TARGET_FOLDERS:
        combine_folder(folder_name)


if __name__ == "__main__":
    main()