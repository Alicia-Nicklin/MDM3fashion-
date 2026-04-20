import os
import pandas as pd

DATA_FOLDER = "data"

FILES = {
    "Micro": os.path.join(DATA_FOLDER, "Microtrends", "microtrends_combined.csv"),
    "Macro": os.path.join(DATA_FOLDER, "Macrotrends", "macrotrends_combined.csv"),
    "Mega":  os.path.join(DATA_FOLDER, "MegaTrends", "megatrends_combined.csv"),
}

OUTPUT_NO_CLASS = os.path.join(DATA_FOLDER, "all_trends_combined.csv")
OUTPUT_WITH_CLASS = os.path.join(DATA_FOLDER, "all_trends_with_classes.csv")


def main():
    all_data_with_class = []
    all_data_no_class = []

    for trend_class, file_path in FILES.items():
        print(f"Loading {trend_class}: {file_path}")

        if not os.path.exists(file_path):
            print(f"  WARNING: file not found -> {file_path}")
            continue

        df = pd.read_csv(file_path)

        # version WITH class
        df_with_class = df.copy()
        df_with_class["trend_class"] = trend_class
        all_data_with_class.append(df_with_class)

        # version WITHOUT class
        df_no_class = df.copy()
        all_data_no_class.append(df_no_class)

    if not all_data_with_class:
        print("No data loaded.")
        return

    # combine
    combined_with_class = pd.concat(all_data_with_class, ignore_index=True)
    combined_no_class = pd.concat(all_data_no_class, ignore_index=True)

    # save both
    combined_with_class.to_csv(OUTPUT_WITH_CLASS, index=False)
    combined_no_class.to_csv(OUTPUT_NO_CLASS, index=False)

    print("\nSaved:")
    print(OUTPUT_WITH_CLASS)
    print(OUTPUT_NO_CLASS)

    print("\nSummary:")
    print(combined_with_class["trend_class"].value_counts())
    print(f"Total rows: {len(combined_with_class)}")
    print(f"Unique trends: {combined_with_class['trend_name'].nunique()}")


if __name__ == "__main__":
    main()