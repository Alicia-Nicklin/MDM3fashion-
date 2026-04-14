"""
Loads all Google Trends + Wikipedia CSVs, resamples
Wikipedia from daily to monthly, aligns and normalises.

Output: output/ml_ready_data.csv
        output/trend_summary.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

os.makedirs("output", exist_ok=True)

TREND_MAP = {
    "Barbie":           ("BARBIE.csv",           "BARBIE.csv"),
    "Business Cash":    ("BUSINESS CASH.csv",    "BUSINESS CASH.csv"),
    "Cottagecore":      ("COTTAGECORE.csv",       None),
    "Dark Academia":    ("DARK ACADEMIA.csv",     None),
    "Festival Fashion": ("FESTIVAL FASHION.csv",  None),
    "Galaxy Print":     ("galaxyprint.csv",       None),
    "Indie":            ("INDIE.csv",             "INDIE.csv"),
    "Logo Mania":       ("logoMANIA.csv",         "logoMANIA.csv"),
    "Millennial Pink":  ("MILLENNIALPINK.csv",    None),
    "Mom Jeans":        ("MOMJEANS.csv",          "MOMJEANS.csv"),
    "Normcore":         ("NORMcore.csv",          None),
    "Pendulum Tops":    ("PENDULUMTOPS.csv",      None),
    "Skinny Jeans":     ("SKINNYJEANS.csv",       "SKINNYJEANS.csv"),
    "Soft Grunge":      ("SOFTGRUNGE.csv",        None),
    "Streetwear":       ("STREETWARE.csv",        None),
    "Tumblr":           ("TUMBLR.csv",            "TUMBLR.csv"),
    "Twee":             ("TWEE.csv",              "TWEE.csv"),
    "Vintage Fashion":  ("VINTAGEFASHION.csv",    "VINTAGEFASHION.csv"),
    "VSCO":             ("VSCO.csv",              "VSCO.csv"),
}

GT_DIR   = "data/google_trends_data"
WIKI_DIR = "data/wikipedia"


def load_google_trends(filepath):
    df             = pd.read_csv(filepath)
    df.columns     = ['date', 'interest']
    df['date']     = pd.to_datetime(df['date'])
    df['interest'] = pd.to_numeric(
        df['interest'].replace('<1', '0'),
        errors='coerce'
    ).fillna(0)
    return df.set_index('date').sort_index()['interest']


def load_wikipedia(filepath):
    df          = pd.read_csv(filepath)
    df.columns  = ['date', 'views']
    df['date']  = pd.to_datetime(df['date'])
    df['views'] = pd.to_numeric(
        df['views'], errors='coerce'
    ).fillna(0)
    df = df.set_index('date').sort_index()
    return df['views'].resample('MS').sum()


def smooth(arr, window=5):
    n  = len(arr)
    wl = min(window, n - 1)
    wl = wl if wl % 2 == 1 else wl - 1
    wl = max(wl, 3)
    return np.clip(
        savgol_filter(arr, window_length=wl, polyorder=2),
        0, None
    )


def normalise(arr):
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return arr * 0.0


def load_all_trends():
    processed = {}
    rows      = []

    print("Loading trends...\n")

    for trend_name, (gt_file, wiki_file) in TREND_MAP.items():

        # Google Trends
        gt_path = os.path.join(GT_DIR, gt_file)
        if not os.path.exists(gt_path):
            print(f"  SKIP {trend_name} — not found")
            continue

        gt_raw    = load_google_trends(gt_path)
        gt_smooth = smooth(gt_raw.values.astype(float))
        gt_norm   = normalise(gt_smooth)

        # Wikipedia
        wiki_norm = np.zeros(len(gt_norm))
        has_wiki  = False

        if wiki_file is not None:
            wiki_path = os.path.join(WIKI_DIR, wiki_file)
            if os.path.exists(wiki_path):
                wiki_raw     = load_wikipedia(wiki_path)
                wiki_aligned = wiki_raw.reindex(
                    gt_raw.index, fill_value=0
                )
                wiki_norm = normalise(
                    wiki_aligned.values.astype(float)
                )
                has_wiki = True
            else:
                print(f"  WARN {trend_name} — "
                      f"wiki not found")

        # Store
        processed[trend_name] = {
            'gt_normalised':   gt_norm,
            'wiki_normalised': wiki_norm,
            'dates':           gt_raw.index,
            'n':               len(gt_norm),
            'has_wiki':        has_wiki,
        }

        print(f"  {trend_name:<20} "
              f"n={len(gt_norm)}  "
              f"wiki={'yes' if has_wiki else 'no'}")

        # Rows for CSV
        for i, date in enumerate(gt_raw.index):
            rows.append({
                'trend':           trend_name,
                'date':            date,
                'period':          i,
                'gt_normalised':   float(gt_norm[i]),
                'wiki_normalised': float(wiki_norm[i]),
                'has_wiki':        has_wiki,
            })

    # Save ml_ready_data.csv
    pd.DataFrame(rows).to_csv(
        "output/ml_ready_data.csv", index=False
    )
    print("\nSaved: output/ml_ready_data.csv")

    # Save trend_summary.csv
    summary = []
    for t, d in processed.items():
        summary.append({
            'trend':    t,
            'n_months': d['n'],
            'has_wiki': d['has_wiki'],
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("output/trend_summary.csv",
                      index=False)
    print("Saved: output/trend_summary.csv\n")
    print(summary_df.to_string(index=False))

    return processed


if __name__ == '__main__':
    processed = load_all_trends()
    print("\nDone. Ready for modelling.")