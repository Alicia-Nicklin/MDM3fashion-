import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs("output", exist_ok=True)

TREND_META = {
    "Soft grunge":     {"color": "#888780", "era": "Tumblr (2012-2016)"},
    "Millennial pink": {"color": "#D4537E", "era": "Instagram (2017-2019)"},
    "Cottagecore":     {"color": "#1D9E75", "era": "TikTok (2020-2022)"},
    "Dark academia":   {"color": "#534AB7", "era": "TikTok (2020-2022)"},
}

plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.size':        10,
})

TRAIN_FRACTION = 0.4

#bass model equations 

def bass_incremental(t, p, q):
    """
    Bass model with M fixed at 1.0 (data is already normalised to [0,1]).
    p = innovation coefficient
    q = imitation coefficient
    """
    M = 1.0
    e = np.exp(-(p + q) * t)
    return M * ((p + q)**2 / p) * e / (1 + (q / p) * e)**2

def bass_cumulative(t, p, q):
    """Cumulative adoption with M fixed at 1.0."""
    M = 1.0
    e = np.exp(-(p + q) * t)
    return M * (1 - e) / (1 + (q / p) * e)

def peak_time(p, q):
    """Theoretical time of peak adoption."""
    return np.log(q / p) / (p + q)

def peak_value(p, q, M):
    """Theoretical peak adoption rate."""
    return M * (p + q)**2 / (4 * q)

#fitting 

def fit_bass(normalised_series, train_fraction=TRAIN_FRACTION, trend_name=""):
    """
    Fit Bass model via cumulative curve — more numerically stable
    than fitting the incremental curve directly.
    TikTok trends get a larger training window because their series
    starts before the trend ignites, leaving the first 40% near-zero.
    """
    n = len(normalised_series)

    # TikTok trends need more training data
    if trend_name in ['Cottagecore', 'Dark academia']:
        actual_fraction = 0.6
    else:
        actual_fraction = train_fraction

    n_train = max(8, int(n * actual_fraction))

    t_train = np.arange(n_train, dtype=float)
    t_full  = np.arange(n,       dtype=float)

    # Build cumulative version of observed data
    y_raw        = normalised_series[:n_train]
    y_cumulative = np.cumsum(y_raw)
    y_cum_norm   = y_cumulative / y_cumulative.max() if y_cumulative.max() > 0 else y_cumulative

    best_result = None
    best_rmse   = np.inf

    p0_grid = [
        [0.01, 0.3],
        [0.03, 0.5],
        [0.05, 0.4],
        [0.02, 0.6],
        [0.10, 0.2],
        [0.01, 0.8],
        [0.05, 0.9],
        [0.03, 0.3],
        [0.08, 0.5],
    ]

    for p0 in p0_grid:
        try:
            popt, pcov = curve_fit(
                bass_cumulative,
                t_train, y_cum_norm,
                p0     = p0,
                bounds = ([1e-4, 1e-4],
                          [0.8,  2.0]),
                maxfev = 20000
            )
            pred = bass_cumulative(t_train, *popt)
            rmse = np.sqrt(np.mean((pred - y_cum_norm)**2))
            if rmse < best_rmse:
                best_rmse   = rmse
                best_result = (popt, pcov)
        except RuntimeError:
            continue

    if best_result is None:
        return None

    popt, pcov = best_result
    p, q       = popt
    perr       = np.sqrt(np.diag(pcov))

    # Generate incremental predictions for plotting
    pred_full     = bass_incremental(t_full, p, q)
    observed_peak = normalised_series.max()
    model_peak    = pred_full.max()
    if model_peak > 0:
        pred_full = pred_full * (observed_peak / model_peak)

    test_rmse  = np.sqrt(np.mean(
        (pred_full[n_train:] - normalised_series[n_train:])**2
    ))
    train_rmse = np.sqrt(np.mean(
        (pred_full[:n_train] - normalised_series[:n_train])**2
    ))

    return {
        'p':          p,
        'q':          q,
        'M':          1.0,
        'p_err':      perr[0],
        'q_err':      perr[1],
        'M_err':      0.0,
        'pq_ratio':   q / p,
        'peak_month': peak_time(p, q),
        'peak_val':   peak_value(p, q, 1.0) * (observed_peak / model_peak),
        'train_rmse': train_rmse,
        'test_rmse':  test_rmse,
        'n_train':    n_train,
        'n_total':    n,
        'pred_full':  pred_full,
    }

#plotting 

def plot_bass_fits(results_dict, df):
    trends = list(results_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Bass Diffusion Model fits to Google Trends fashion data',
                 fontsize=14, fontweight='bold')

    for ax, trend_name in zip(axes.flat, trends):
        res    = results_dict[trend_name]
        colour = TREND_META.get(trend_name, {}).get('color', '#555555')

        trend_df = df[df['trend'] == trend_name].copy()
        era      = trend_df['era'].iloc[0]
        observed = trend_df['normalised'].values
        t_full   = np.arange(len(observed))
        n_train  = res['n_train']

        # Observed
        ax.plot(t_full, observed, color=colour,
                alpha=0.7, linewidth=1.5, label='Observed (normalised)')

        # Training window shading
        ax.axvspan(0, n_train - 1, alpha=0.08, color='steelblue')
        ax.axvline(n_train - 1, color='steelblue', linestyle=':',
                   alpha=0.6, linewidth=1.2,
                   label=f'Train/test split (t={n_train})')

        # Bass fit on training portion
        ax.plot(t_full[:n_train], res['pred_full'][:n_train],
                'k--', linewidth=2, alpha=0.8,
                label=f"Bass fit  (train RMSE={res['train_rmse']:.3f})")

        # Forecast portion
        ax.plot(t_full[n_train:], res['pred_full'][n_train:],
                color='crimson', linewidth=2, linestyle='--',
                label=f"Forecast  (test RMSE={res['test_rmse']:.3f})")

        # Peak marker
        pt = res['peak_month']
        if 0 < pt < len(observed):
            ax.axvline(pt, color='orange', linestyle='--',
                       alpha=0.5, linewidth=1)
            ax.annotate(f'Peak ≈ t{pt:.0f}',
                        xy=(pt, res['peak_val']),
                        xytext=(pt + 2, res['peak_val'] + 0.05),
                        fontsize=7, color='darkorange')

        # Parameter box
        p_note = " (boundary)" if res['p'] < 0.001 else ""
        param_text = (f"p = {res['p']:.4f}{p_note}\n"
              f"q = {res['q']:.3f}\n"
              f"q/p = {res['pq_ratio']:.1f}")
        ax.text(0.02, 0.35, param_text,
                transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='grey', alpha=0.8))

        ax.set_title(f"{trend_name}  [{era}]", fontweight='bold')
        ax.set_xlabel('Months from series start')
        ax.set_ylabel('Normalised interest [0,1]')
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig('output/fig_bass_fits.png', dpi=150, bbox_inches='tight')
    print("  Saved: output/fig_bass_fits.png")
    plt.close()


def plot_pq_comparison(results_dict):
    """
    Bar chart comparing p and q across all trends.
    Key result — shows imitation vs innovation driven spread.
    """
    trends   = list(results_dict.keys())
    p_vals   = [results_dict[t]['p']        for t in trends]
    q_vals   = [results_dict[t]['q']        for t in trends]
    p_errs   = [results_dict[t]['p_err']    for t in trends]
    q_errs   = [results_dict[t]['q_err']    for t in trends]
    pq_ratio = [results_dict[t]['pq_ratio'] for t in trends]
    colours  = [TREND_META.get(t, {}).get('color', '#555') for t in trends]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Bass Model Parameters — Innovation vs Imitation by Trend',
                 fontsize=13, fontweight='bold')

    x           = np.arange(len(trends))
    short_names = [t.replace(' ', '\n') for t in trends]

    # p coefficients
    axes[0].bar(x, p_vals, color=colours, alpha=0.85,
                yerr=p_errs, capsize=4, edgecolor='black', linewidth=0.5)
    axes[0].set_title('p  (Innovation coefficient)', fontweight='bold')
    axes[0].set_ylabel('p value')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, fontsize=9)
    axes[0].set_xlabel('Higher p = independent discovery')

    # q coefficients
    axes[1].bar(x, q_vals, color=colours, alpha=0.85,
                yerr=q_errs, capsize=4, edgecolor='black', linewidth=0.5)
    axes[1].set_title('q  (Imitation coefficient)', fontweight='bold')
    axes[1].set_ylabel('q value')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, fontsize=9)
    axes[1].set_xlabel('Higher q = social contagion')

    # q/p ratio
    bars = axes[2].bar(x, pq_ratio, color=colours, alpha=0.85,
                       edgecolor='black', linewidth=0.5)
    axes[2].set_title('q/p  (Imitation-to-Innovation ratio)', fontweight='bold')
    axes[2].set_ylabel('q/p')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, fontsize=9)
    axes[2].set_xlabel('Higher = more driven by social spread')
    axes[2].axhline(1, color='red', linestyle='--',
                    alpha=0.4, label='q/p = 1 threshold')
    axes[2].legend(fontsize=8)
    axes[2].set_yscale('log')
    axes[2].set_ylabel('q/p  (log scale)')

    for bar, val in zip(bars, pq_ratio):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f'{val:.1f}×', ha='center',
                     fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('output/fig_bass_pq_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: output/fig_bass_pq_comparison.png")
    plt.close()


def plot_cumulative_penetration(results_dict, df):
    """Cumulative adoption curves — how deeply did each trend penetrate?"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for trend_name, res in results_dict.items():
        colour   = TREND_META.get(trend_name, {}).get('color', '#555')
        t_full   = np.arange(res['n_total'])
        cum_pred = bass_cumulative(t_full, res['p'], res['q'])

        ax.plot(t_full, cum_pred, color=colour,
                linewidth=2.5,
                label=f"{trend_name}  (M={res['M']:.2f})")
        ax.axhline(res['M'], color=colour, linestyle=':',
                   alpha=0.4, linewidth=1)

    ax.set_xlabel('Months from series start', fontsize=11)
    ax.set_ylabel('Cumulative adoption F(t)', fontsize=11)
    ax.set_title('Cumulative Trend Penetration — Bass Model',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('output/fig_bass_cumulative.png', dpi=150, bbox_inches='tight')
    print("  Saved: output/fig_bass_cumulative.png")
    plt.close()


#main

def main():
    print("Loading processed_trends.csv...")
    df = pd.read_csv('all_trends_combined.csv')

    # Rename columns to match what the Bass code expects
    df = df.rename(columns={
        'trend_name': 'trend',
        'value_norm': 'normalised'
    })

    # Ensure correct types
    df['date'] = pd.to_datetime(df['date'])

    # Sort properly (VERY important for time series)
    df = df.sort_values(['trend', 'date']).reset_index(drop=True)

    # OPTIONAL: add a dummy era column if you don’t have one
    df['era'] = 'Unknown'

    trend_names = df['trend'].unique()
    print(f"Trends found: {list(trend_names)}\n")

    results = {}

    for trend_name in trend_names:
        print(f"Fitting Bass model: {trend_name}")
        trend_df   = df[df['trend'] == trend_name].copy()
        normalised = trend_df['normalised'].values

        res = fit_bass(normalised, trend_name=trend_name)
        if res is None:
            print(f"  WARNING: fitting failed for {trend_name}")
            continue

        results[trend_name] = res

        print(f"  p (innovation) = {res['p']:.4f} ± {res['p_err']:.4f}")
        print(f"  q (imitation)  = {res['q']:.4f} ± {res['q_err']:.4f}")
        print(f"  q/p ratio      = {res['pq_ratio']:.2f}")
        print(f"  Predicted peak = month {res['peak_month']:.1f}")
        print(f"  Train RMSE     = {res['train_rmse']:.4f}")
        print(f"  Test  RMSE     = {res['test_rmse']:.4f}")
        print()

    # Save parameter table
    param_rows = []
    for trend_name, res in results.items():
        era = df[df['trend'] == trend_name]['era'].iloc[0]
        param_rows.append({
            'trend':      trend_name,
            'era':        era,
            'p':          round(res['p'],          4),
            'q':          round(res['q'],          4),
            'M':          round(res['M'],          3),
            'p_err':      round(res['p_err'],      4),
            'q_err':      round(res['q_err'],      4),
            'pq_ratio':   round(res['pq_ratio'],   2),
            'peak_month': round(res['peak_month'], 1),
            'train_rmse': round(res['train_rmse'], 4),
            'test_rmse':  round(res['test_rmse'],  4),
        })

    params_df = pd.DataFrame(param_rows)
    params_df.to_csv('output/bass_parameters.csv', index=False)
    print("Saved: output/bass_parameters.csv")
    print()
    print(params_df.to_string(index=False))
    print()

    # Generate all figures
    print("Generating figures...")
    plot_bass_fits(results, df)
    plot_pq_comparison(results)
    plot_cumulative_penetration(results, df)

    print("\nDone. Check output/ folder for figures.")


if __name__ == '__main__':
    main()



