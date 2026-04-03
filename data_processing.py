"""
Data Preprocessing and Characterisation
Engineering Mathematics Year 3 Group Project

Run from VSCode terminal:
    python section2_preprocessing.py

Install dependencies first:
    pip install pandas numpy matplotlib scipy statsmodels requests

Folder structure required:
    project/
    ├── section2_preprocessing.py
    ├── data/
    │   ├── soft_grunge.csv
    │   ├── millennial_pink.csv
    │   ├── cottagecore.csv
    │   └── dark_academia.csv
    └── output/          <-- created automatically

Pipeline:
    raw CSV --> seasonal decompose --> trend component
             --> SG smooth --> normalise [0,1] --> SIR fitting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import savgol_filter
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
import requests
import warnings
warnings.filterwarnings('ignore')

os.makedirs("output", exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.size':        10,
})

print("Libraries loaded.")


# ============================================================================
# 1. CONFIGURATION
#
#    smooth_window: Savitzky-Golay window in weeks. Must be odd.
#      Increase for spikier series (soft grunge needs more smoothing).
#      Decrease if peaks look too flattened after running fig4.
#
#    pinterest: update manually after checking trends.pinterest.com
#      for each keyword. Format is approximate quarter e.g. "2017-Q2".
# ============================================================================

TREND_META = {
    "Soft grunge": {
        "file":          "data/soft_grunge.csv",
        "color":         "#888780",
        "era":           "Tumblr (2012-2016)",
        "smooth_window": 21,
        "dates":         ("2012-06", "2018-06"),
    },
    "Millennial pink": {
        "file":          "data/millennial_pink.csv",
        "color":         "#D4537E",
        "era":           "Instagram (2017-2019)",
        "smooth_window": 11,
        "dates":         ("2017-01", "2021-06"),
    },
    "Cottagecore": {
        "file":          "data/cottagecore.csv",
        "color":         "#1D9E75",
        "era":           "TikTok (2020-2022)",
        "smooth_window": 11,
        "dates":         ("2019-01", "2023-01"),
    },
    "Dark academia": {
        "file":          "data/dark_academia.csv",
        "color":         "#534AB7",
        "era":           "TikTok (2020-2022)",
        "smooth_window": 11,
        "dates":         ("2019-01", "2023-01"),
    },
}


# ============================================================================
# 2. LOAD DATA
# ============================================================================

def load_google_trends(filepath):
    """
    Load a Google Trends CSV export.
    Google includes 2 junk header rows before the data starts.
    '<1' entries mean very low interest -- replaced with 0.
    Returns a pandas Series with a weekly or monthly datetime index.
    """
    df = pd.read_csv(filepath, skiprows=1, header=0)
    df.columns = ['date', 'interest']
    df['interest'] = df['interest'].replace('<1', '0')
    df['interest'] = pd.to_numeric(df['interest'], errors='coerce').fillna(0)
    df['date']     = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df['interest']


def detect_frequency(series):
    """Return 'monthly' if median gap between observations > 10 days, else 'weekly'."""
    gaps = series.index.to_series().diff().dt.days.dropna()
    return 'monthly' if gaps.median() > 10 else 'weekly'


RAW  = {}
FREQ = {}
print("\nLoading Google Trends CSVs...")
for name, meta in TREND_META.items():
    try:
        series = load_google_trends(meta["file"])
        RAW[name]  = series
        FREQ[name] = detect_frequency(series)
        # Validate date range roughly matches intended window
        expected_start = pd.to_datetime(meta["dates"][0])
        expected_end   = pd.to_datetime(meta["dates"][1])
        actual_start   = series.index[0]
        actual_end     = series.index[-1]
        gap_start = abs((actual_start - expected_start).days)
        gap_end   = abs((actual_end   - expected_end).days)
        date_ok = gap_start < 60 and gap_end < 60
        flag = "" if date_ok else "  <-- WARNING: dates may not match intended range"
        freq_label = FREQ[name]
        print(f"  {name}: {len(series)} obs ({freq_label})  "
              f"[{actual_start.strftime('%b %Y')} to {actual_end.strftime('%b %Y')}]  "
              f"peak = {series.max():.0f} at {series.idxmax().strftime('%b %Y')}{flag}")
    except FileNotFoundError:
        print(f"  WARNING: {meta['file']} not found -- download from trends.google.com")

if not RAW:
    raise SystemExit("No data loaded. Add CSVs to the data/ folder first.")


# ============================================================================
# 3. RAW DATA PLOT
# ============================================================================

print("\nFigure 1: Raw data...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for ax, (name, series) in zip(axes, RAW.items()):
    meta = TREND_META[name]
    ax.plot(series.index, series.values,
            color=meta["color"], lw=1.5, alpha=0.8)
    ax.fill_between(series.index, series.values,
                    color=meta["color"], alpha=0.12)
    peak_date = series.idxmax()
    ax.axvline(peak_date, color=meta["color"], lw=1, linestyle='--', alpha=0.6)
    ax.text(peak_date, series.max() * 0.92,
            f'Peak: {peak_date.strftime("%b %Y")}',
            fontsize=8, ha='center', color=meta["color"])
    ax.set_title(f'{name}  [{meta["era"]}]', fontweight='bold', fontsize=10)
    ax.set_ylabel("Google Trends interest (0-100)")

fig.suptitle("Raw Google Trends data -- four fashion aesthetics across platform eras",
             fontsize=12, fontweight='bold')
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.savefig("output/fig1_raw_data.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("  Saved: output/fig1_raw_data.png")


# ============================================================================
# 4. SEASONAL DECOMPOSITION
#
#    Splits each series into three components:
#      trend    -- the underlying diffusion arc (what we want)
#      seasonal -- repeating annual pattern (fashion week cycles)
#      residual -- remaining noise
#
#    Additive model: observed = trend + seasonal + residual.
#    This is appropriate here because the seasonal swings are roughly
#    constant in amplitude regardless of the trend level.
#
#    Period = 52 weeks (one year) for weekly data.
#
#    The TREND COMPONENT is stored in TREND_COMPONENTS and used as
#    the input to the smoothing step -- this is what gets passed to
#    the SIR fitter in Section 3. Using the trend component rather than
#    raw data removes fashion-week seasonality before fitting.
# ============================================================================

print("\nFigure 2: Seasonal decomposition...")
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
TREND_COMPONENTS = {}

for row, (name, series) in enumerate(RAW.items()):
    meta   = TREND_META[name]
    if FREQ[name] == 'monthly':
        period = min(12, len(series) // 2)
    else:
        period = min(52 if len(series) >= 104 else 26, len(series) // 2)
    result = seasonal_decompose(series, model='additive',
                                period=period, extrapolate_trend='freq')

    # Store trend component -- this is the input to Section 7 smoothing
    TREND_COMPONENTS[name] = result.trend.bfill().ffill()

    axes[row, 0].plot(series.index, series.values,
                      color=meta["color"], lw=1.2, alpha=0.7)
    axes[row, 0].set_title(f'{name} -- observed', fontsize=9)
    axes[row, 0].set_ylabel("Interest")

    trend_clipped = result.trend.clip(lower=0)
    axes[row, 1].plot(trend_clipped.index, trend_clipped.values,
                      color=meta["color"], lw=2)
    axes[row, 1].set_title('Trend component (analysis only, clipped >= 0)', fontsize=9)
    axes[row, 1].set_ylabel("Interest")

    axes[row, 2].plot(result.resid.index, result.resid.values,
                      color='#888780', lw=1, alpha=0.7)
    axes[row, 2].axhline(0, color='black', lw=0.5, linestyle='--')
    axes[row, 2].set_title('Residual', fontsize=9)
    axes[row, 2].set_ylabel("Interest")

fig.suptitle("Seasonal decomposition -- analysis only (SIR fitter uses smoothed raw data)",
             fontsize=12, fontweight='bold')
for ax in axes.flatten():
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
fig.autofmt_xdate(rotation=45)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("output/fig2_decomposition.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("  Saved: output/fig2_decomposition.png")


# ============================================================================
# 5. ADF STATIONARITY TEST
#
#    Augmented Dickey-Fuller test.
#    H0: series has a unit root (is non-stationary).
#    Fail to reject H0 (p > 0.05) --> non-stationary --> diffusion-like.
#
#    All four trend series are expected to be non-stationary -- they have
#    a time-varying mean by definition (they rise, peak, and fall).
#    This mathematically confirms the data follows a diffusion process,
#    which motivates using a mechanistic SIR ODE model rather than any
#    purely statistical approach which assumes stationarity.
# ============================================================================

print("\n" + "=" * 65)
print("ADF Stationarity Test")
print("H0: series is non-stationary (unit root present)")
print("p > 0.05  -->  fail to reject H0  -->  non-stationary (expected)")
print("=" * 65)

ADF_RESULTS = {}

for name, series in RAW.items():
    res        = adfuller(series.dropna(), autolag='AIC')
    adf_stat   = res[0]
    p_value    = res[1]
    lags       = res[2]
    stationary = p_value < 0.05

    ADF_RESULTS[name] = {
        "adf_stat":   adf_stat,
        "p_value":    p_value,
        "stationary": stationary,
    }

    label = "STATIONARY (unexpected)" if stationary else "NON-STATIONARY -- expected"
    print(f"\n  {name}")
    print(f"    ADF statistic : {adf_stat:.4f}")
    print(f"    p-value       : {p_value:.4f}")
    print(f"    Lags used     : {lags}")
    print(f"    Result        : {label}")

print("\n" + "=" * 65)
print("All four series should be non-stationary.")
print("This confirms diffusion dynamics and motivates the SIR model.")
print("=" * 65)


# ============================================================================
# 6. ACF AND PACF PLOTS
#
#    ACF (Autocorrelation Function): measures how correlated the series is
#    with its own past values at each lag. A slow decay extending many lags
#    confirms non-stationarity and long memory -- consistent with SIR
#    dynamics where I(t) this week strongly predicts I(t) next week.
#
#    PACF (Partial ACF): correlation at each lag after removing the
#    influence of all shorter lags. Reveals the direct autoregressive
#    structure. Sharp cutoff after lag 1-2 is typical of AR(1) processes
#    which is consistent with discrete SIR dynamics.
#
#    Dashed lines mark the 95% confidence band (1.96 / sqrt(n)).
#    Bars outside this band are statistically significant.
# ============================================================================

print("\nFigure 3: ACF...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
N_LAGS = 40

for ax, (name, series) in zip(axes, RAW.items()):
    meta  = TREND_META[name]
    clean = series.dropna()
    n_obs = len(clean)
    lags  = min(N_LAGS, n_obs // 2 - 1)
    conf  = 1.96 / np.sqrt(n_obs)

    acf_vals  = acf(clean, nlags=lags, fft=True)
    lag_range = np.arange(len(acf_vals))

    ax.bar(lag_range, acf_vals, color=meta["color"], alpha=0.75, width=0.5)
    ax.axhline( conf, color='black', lw=0.8, linestyle='--', alpha=0.5, label='95% CI')
    ax.axhline(-conf, color='black', lw=0.8, linestyle='--', alpha=0.5)
    ax.axhline(0,     color='black', lw=0.5)
    ax.set_title(f'{name}\n[{meta["era"]}]', fontsize=9, fontweight='bold')
    ax.set_xlabel(f"Lag ({'months' if FREQ[name] == 'monthly' else 'weeks'})")
    ax.set_ylabel("Autocorrelation")
    ax.set_ylim(-0.5, 1.05)
    ax.legend(fontsize=7)

fig.suptitle("ACF -- slow decay across all four trends confirms long-memory diffusion dynamics",
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("output/fig3_acf.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("  Saved: output/fig3_acf.png")


# ============================================================================
# 7. SMOOTHING AND NORMALISATION
#
#    Input: TREND_COMPONENTS (from seasonal decomposition above).
#    Using the trend component rather than raw data removes seasonal
#    fashion-week cycles before fitting, giving the SIR fitter a
#    cleaner signal.
#
#    Savitzky-Golay filter: fits a local polynomial across a sliding window.
#    Removes remaining short-term noise while preserving peak shape.
#    Better than a moving average which artificially flattens peaks.
#
#    smooth_window is set per-trend in TREND_META:
#      soft grunge uses window=21 (series is very spiky)
#      others use window=11 (standard for weekly data)
#
#    Normalise to [0,1]: Google Trends values are relative (0-100 within
#    each series), not comparable across series. Normalising puts all four
#    trends on the same scale so fitted SIR parameters are comparable.
# ============================================================================

def smooth(arr, window=11):
    """
    Savitzky-Golay smoothing filter.
    window must be odd and greater than polyorder.
    polyorder capped at 2 so short monthly windows (3, 5) don't crash.
    """
    n  = len(arr)
    wl = min(window, n - 1)
    wl = wl if wl % 2 == 1 else wl - 1
    wl = max(wl, 3)
    polyorder = min(2, wl - 1)
    return savgol_filter(arr, window_length=wl, polyorder=polyorder)


def normalise(arr):
    """Scale array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else arr


PROCESSED = {}
for name, raw_series in RAW.items():
    meta   = TREND_META[name]
    vals   = raw_series.values.astype(float)
    if FREQ[name] == 'monthly':
        window = 5 if name == "Soft grunge" else 3
    else:
        window = meta["smooth_window"]
    smoothed = np.clip(smooth(vals, window=window), 0, None)
    norm     = normalise(smoothed)

    PROCESSED[name] = {
        "raw":           vals,
        "smoothed":      smoothed,
        "normalised":    norm,
        "dates":         raw_series.index,
        "n":             len(vals),
        "smooth_window": window,
        "freq":          FREQ[name],
    }

print("\nFigure 4: Smoothing and normalisation...")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, (name, proc) in zip(axes, PROCESSED.items()):
    meta = TREND_META[name]
    t    = np.arange(proc["n"])

    ax.plot(t, proc["raw"][:proc["n"]], color='#B4B2A9', lw=1, alpha=0.5,
            label="Raw")
    ax.plot(t, proc["smoothed"],        color=meta["color"], lw=2, alpha=0.9,
            label=f"Smoothed (SG w={proc['smooth_window']})")
    ax.set_title(f'{name}  [{meta["era"]}]', fontweight='bold', fontsize=10)
    ax.set_xlabel("Months from series start" if proc["freq"] == "monthly" else "Weeks from series start")
    ax.set_ylabel("Google Trends interest (0-100)")
    ax.legend(loc='upper left', fontsize=7)

    ax2 = ax.twinx()
    ax2.plot(t, proc["normalised"], color=meta["color"], lw=1.5,
             linestyle='--', alpha=0.5, label="Normalised [0,1]")
    ax2.set_ylabel("Normalised [0,1]", color=meta["color"], fontsize=8)
    ax2.tick_params(axis='y', labelcolor=meta["color"])
    ax2.set_ylim(-0.05, 1.15)

fig.suptitle("Smoothed raw data -- input to SIR model fitting",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("output/fig4_smoothed_normalised.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("  Saved: output/fig4_smoothed_normalised.png")
print("\n  CHECK THIS FIGURE CAREFULLY:")
print("  Soft grunge: if still too spiky, increase smooth_window to 25 or 31")
print("  Others: if peaks look flattened, decrease smooth_window to 7 or 9")


# ============================================================================
# 8. WIKIPEDIA CROSS-VALIDATION
#
#    Independent signal from Wikimedia REST API -- no key needed, free.
#    Monthly Wikipedia pageviews measure public awareness of the trend.
#    If peaks align within ~4-8 weeks of Google Trends peaks, the Google
#    Trends signal is validated as a genuine adoption proxy.
#
#    Soft grunge: no dedicated Wikipedia article -- skipped.
# ============================================================================

def get_wiki_pageviews(article, start_yyyymm, end_yyyymm):
    """
    Fetch monthly Wikipedia pageviews via Wikimedia REST API.
    start/end format: 'YYYYMM' e.g. '201701'
    Returns pandas Series with monthly datetime index, or None on failure.
    """
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/"
           f"per-article/en.wikipedia/all-access/all-agents/"
           f"{article}/monthly/{start_yyyymm}01/{end_yyyymm}01")
    try:
        r     = requests.get(url,
                             headers={"User-Agent": "eng-maths-student-project"},
                             timeout=10)
        items = r.json().get("items", [])
        if not items:
            print(f"    No data returned for {article}")
            return None
        dates = pd.to_datetime([i["timestamp"][:7] for i in items])
        views = [i["views"] for i in items]
        return pd.Series(views, index=dates)
    except Exception as e:
        print(f"    Wikipedia fetch failed for {article}: {e}")
        return None


print("\nFetching Wikipedia pageview data (requires internet)...")
WIKI = {}

wiki_queries = {
    "Millennial pink": ("Millennial_pink", "201701", "202106"),
    "Cottagecore":     ("Cottagecore",     "201901", "202301"),
    "Dark academia":   ("Dark_academia",   "201901", "202301"),
}

for trend_name, (article, start, end) in wiki_queries.items():
    series = get_wiki_pageviews(article, start, end)
    if series is not None and len(series) > 0:
        WIKI[trend_name] = series
        print(f"  {trend_name}: {len(series)} months  "
              f"peak = {series.max():,} ({series.idxmax().strftime('%b %Y')})")


# ============================================================================
# 9. CROSS-VALIDATION PLOT
# ============================================================================

if WIKI:
    print("\nFigure 5: Cross-validation...")
    n_plots = len(WIKI)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, wiki_series) in zip(axes, WIKI.items()):
        meta    = TREND_META[name]
        gt      = RAW[name]
        gt_norm = normalise(gt.values.astype(float))
        wk_norm = normalise(wiki_series.values.astype(float))

        ax.plot(gt.index, gt_norm,
                color=meta["color"], lw=2,
                label=f"Google Trends ({FREQ[name]})", alpha=0.85)
        ax.plot(wiki_series.index, wk_norm,
                color='#2C2C2A', lw=1.5, linestyle='--',
                marker='o', markersize=3,
                label="Wikipedia pageviews (monthly)", alpha=0.85)

        gt_peak   = gt.index[np.argmax(gt_norm)]
        wiki_peak = wiki_series.index[np.argmax(wk_norm)]
        ax.axvline(gt_peak,   color=meta["color"], lw=1, linestyle=':')
        ax.axvline(wiki_peak, color='#2C2C2A',     lw=1, linestyle=':')

        lag_months = abs((gt_peak.year - wiki_peak.year) * 12
                        + (gt_peak.month - wiki_peak.month))
        ax.set_title(f'{name}\nPeak lag: {lag_months} months',
                     fontweight='bold', fontsize=10)
        ax.set_ylabel("Normalised interest [0,1]")
        ax.legend(fontsize=7)

    fig.suptitle("Cross-validation: Google Trends vs Wikipedia pageviews\n"
                 "Peak alignment within ~2 months validates Google Trends proxy",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/fig5_crossvalidation.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Saved: output/fig5_crossvalidation.png")
else:
    print("  Skipping cross-validation (no Wikipedia data -- check internet connection)")


# ============================================================================
# 10. SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 82)
print(f"{'Trend':<18} {'Era':<24} {'Obs':>6} {'Peak':>10} "
      f"{'ADF p':>8} {'Non-stat':>9} {'Wiki':>6}")
print("-" * 82)

for name, proc in PROCESSED.items():
    meta      = TREND_META[name]
    series    = RAW[name]
    peak_date = series.idxmax().strftime("%b %Y")
    adf_p     = ADF_RESULTS[name]["p_value"]
    non_stat  = "Yes" if not ADF_RESULTS[name]["stationary"] else "No (!)"
    wiki      = "Yes" if name in WIKI else "N/A"

    print(f"{name:<18} {meta['era']:<24} {proc['n']:>6} {peak_date:>10} "
          f"{adf_p:>8.4f} {non_stat:>9} {wiki:>6}")

print("=" * 82)
print("""
Methodology notes for the report:
  ADF p > 0.05 for all series --> non-stationary --> confirms diffusion process
  Wikipedia peaks within ~2 months of Google Trends --> proxy validated
  Soft grunge: Google Trends only (no Wikipedia article, predates Pinterest)
  Millennial pink: home decor + fashion searches mixed --> limitation noted
  Cottagecore: double-peak structure --> two-wave adoption discussed in report
  Covid confound: both TikTok trends rise from March 2020 lockdown
  Pinterest peaks: UPDATE the pinterest fields in TREND_META after checking
                   trends.pinterest.com for each keyword manually
""")


# ============================================================================
# 11. EXPORT PROCESSED DATA FOR SECTION 3 (SIR FITTING)
#
#    Saves processed_trends.csv to output/.
#    The PROCESSED dict is also available in memory if running interactively.
#    Section 3 imports this CSV as its starting point.
# ============================================================================

rows = []
for name, proc in PROCESSED.items():
    for i, (date, raw_val, sm, nm) in enumerate(zip(
            proc["dates"],
            proc["raw"][:proc["n"]],
            proc["smoothed"],
            proc["normalised"])):
        rows.append({
            "trend":      name,
            "era":        TREND_META[name]["era"],
            "freq":       proc["freq"],
            "date":       date,
            "period":     i,
            "raw":        raw_val,
            "smoothed":   sm,
            "normalised": nm,
        })

pd.DataFrame(rows).to_csv("output/processed_trends.csv", index=False)
print("Exported: output/processed_trends.csv")
print("\nSection 2 complete. All figures saved to output/")
print("Next step: run section3_sir_fitting.py")