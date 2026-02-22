import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path("data")

# ---- helper: load a single Google Trends CSV robustly ----
def load_trends_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardise column names
    df.columns = df.columns.str.strip()

    # Parse time
    if "Time" not in df.columns:
        raise ValueError(f"'Time' column not found in {path.name}. Columns: {df.columns.tolist()}")
    df["Time"] = pd.to_datetime(df["Time"])

    # Find the value column (anything that's not 'Time')
    value_cols = [c for c in df.columns if c != "Time"]
    if len(value_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 value column in {path.name}, got {value_cols}"
        )

    value_col = value_cols[0]
    trend_name = value_col.strip()

    # Rename value column to a clean trend name (title case, no weird spaces)
    trend_name_clean = " ".join(trend_name.split()).title()
    df = df.rename(columns={value_col: trend_name_clean})

    # Ensure numeric
    df[trend_name_clean] = pd.to_numeric(df[trend_name_clean], errors="coerce")

    # Basic checks
    df = df.dropna(subset=["Time"])  # should be none anyway
    df = df.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    return df


# ---- 1) pick your four files (works even if filenames are long) ----
# If you renamed them to short names, this will still work.

files = list(DATA_DIR.glob("*.csv"))# ---- 1) pick ONLY the raw Google Trends exports ----
files = list(DATA_DIR.glob("time_series_*.csv"))

if len(files) < 4:
    raise FileNotFoundError(
        f"Expected 4 time_series_*.csv files in {DATA_DIR.resolve()}, "
        f"found {len(files)}: {[f.name for f in files]}"
    )

print("Found raw CSVs:")
for f in files:
    print(" -", f.name)

if len(files) < 4:
    raise FileNotFoundError(f"Expected at least 4 csv files in {DATA_DIR.resolve()}, found {len(files)}: {files}")

print("Found CSVs:")
for f in files:
    print(" -", f.name)

# ---- 2) load + clean each ----
dfs = [load_trends_csv(f) for f in files]

# ---- 3) merge on Time (wide format) ----
merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="Time", how="inner")

# Sort + final tidy
merged = merged.sort_values("Time").reset_index(drop=True)

# Optional: enforce monthly frequency sanity check
# (Won't crash your code; just prints warning if gaps exist.)
expected = pd.date_range(merged["Time"].min(), merged["Time"].max(), freq="MS")
if len(expected) != len(merged) or not (merged["Time"].values == expected.values).all():
    print("WARNING: Time index is not perfectly monthly 'MS' continuous (missing months or different anchor).")
    # You can inspect missing months:
    missing = expected.difference(merged["Time"])
    if len(missing) > 0:
        print("Missing months:", missing[:10], "..." if len(missing) > 10 else "")

# ---- 4) save cleaned outputs ----
out_wide = DATA_DIR / "trends_merged_GB_monthly.csv"
merged.to_csv(out_wide, index=False)

# Also save long format (very useful for plotting / modelling)
long_df = merged.melt(id_vars="Time", var_name="Trend", value_name="Interest")
out_long = DATA_DIR / "trends_long_GB_monthly.csv"
long_df.to_csv(out_long, index=False)


df = pd.read_csv("data/trends_merged_GB_monthly.csv")
df["Time"] = pd.to_datetime(df["Time"])

print("\n% zeros per trend:")
for col in df.columns[1:]:
    zero_pct = (df[col] == 0).mean() * 100
    print(f"{col}: {zero_pct:.1f}%")

plt.figure(figsize=(12,6))

for col in df.columns[1:]:
    plt.plot(df["Time"], df[col], label=col)

plt.legend()
plt.title("Google Trends Fashion Popularity (UK)")
plt.xlabel("Time")
plt.ylabel("Search Interest (0-100)")
plt.tight_layout()
plt.show()

print("\nSaved:")
print(" -", out_wide)
print(" -", out_long)
print("\nPreview:")
print(merged.head())
print("\nColumns:", merged.columns.tolist())


