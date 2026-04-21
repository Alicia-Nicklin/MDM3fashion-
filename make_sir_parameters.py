import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =========================
# 1) PATHS
# =========================
DATA_DIR = "/Users/rose/Desktop/new fashion3/data"
OUTPUT_DIR = "/Users/rose/Desktop/new fashion3/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "Teddycoat": "teddycoat.csv",
    "Dark academia": "dark_academia.csv",
    "Hipster fashion": "hipster fashion.csv",
}

ERA_MAP = {
    "Teddycoat": "TikTok",
    "Dark academia": "TikTok",
    "Hipster fashion": "Instagram",
}

FREQ = "monthly"

# =========================
# 2) HELPERS
# =========================
def safe_normalise(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.zeros_like(x)

def fill_gaps(mask, tol=1):
    mask = list(mask)
    i = 0
    while i < len(mask):
        if not mask[i]:
            start = i
            while i < len(mask) and not mask[i]:
                i += 1
            end = i - 1
            left = start > 0 and mask[start - 1]
            right = end < len(mask) - 1 and mask[end + 1]
            if left and right and (end - start + 1) <= tol:
                for j in range(start, end + 1):
                    mask[j] = True
        else:
            i += 1
    return np.array(mask, dtype=bool)

def longest_run(mask):
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

def solve_sir_I(beta, gamma, I0, n_periods, sub=4):
    dt = 1.0 / sub
    h6 = dt / 6.0
    h2 = dt / 2.0
    total = (n_periods - 1) * sub + 1

    S = max(1.0 - I0, 1e-9)
    I = max(float(I0), 1e-9)

    I_out = np.empty(n_periods)
    I_out[0] = I
    out_idx = 1

    for step in range(1, total):
        bSI = beta * S * I
        k1s = -bSI
        k1i = bSI - gamma * I

        S2 = S + h2 * k1s
        I2 = I + h2 * k1i
        bSI = beta * S2 * I2
        k2s = -bSI
        k2i = bSI - gamma * I2

        S3 = S + h2 * k2s
        I3 = I + h2 * k2i
        bSI = beta * S3 * I3
        k3s = -bSI
        k3i = bSI - gamma * I3

        S4 = S + dt * k3s
        I4 = I + dt * k3i
        bSI = beta * S4 * I4
        k4s = -bSI
        k4i = bSI - gamma * I4

        S += h6 * (k1s + 2 * k2s + 2 * k3s + k4s)
        I += h6 * (k1i + 2 * k2i + 2 * k3i + k4i)

        S = max(S, 0.0)
        I = max(I, 0.0)

        if step % sub == 0 and out_idx < n_periods:
            I_out[out_idx] = I
            out_idx += 1

    return I_out

def fit_sir(real_series):
    n = len(real_series)

    def objective(params):
        beta, gamma, I0 = params

        if beta <= 0 or gamma <= 0 or I0 <= 0 or I0 >= 1:
            return 1e6

        pred = solve_sir_I(beta, gamma, I0, n)
        pred = safe_normalise(pred)
        return np.mean((pred - real_series) ** 2)

    x0 = np.array([0.5, 0.05, 0.03])
    bounds = [(0.001, 3.0), (0.001, 1.0), (0.001, 0.3)]

    result = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    beta, gamma, I0 = result.x
    return float(beta), float(gamma), float(I0), float(result.fun)

# =========================
# 3) MAIN
# =========================
rows = []

for trend_name, filename in FILES.items():
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        continue

    df = pd.read_csv(path)

    real = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).values.astype(float)
    real = safe_normalise(real)

    active_mask = fill_gaps(real >= 0.35, tol=1)
    run_len, run_start, run_end = longest_run(active_mask)

    if run_start is None:
        print(f"Skipped {trend_name}: no active window")
        continue

    start_idx = max(0, run_start - 5)
    end_idx = min(len(real) - 1, run_end + 5)
    real_clip = real[start_idx:end_idx + 1]

    beta, gamma, I0, mse = fit_sir(real_clip)
    R0 = beta / gamma if gamma > 0 else np.nan

    rows.append({
        "trend": trend_name,
        "era": ERA_MAP.get(trend_name, "Unknown"),
        "freq": FREQ,
        "beta": round(beta, 4),
        "beta_ci_lo": 0,
        "beta_ci_hi": round(beta, 4),
        "gamma": round(gamma, 4),
        "gamma_ci_lo": 0,
        "gamma_ci_hi": round(gamma, 4),
        "R0": round(R0, 4),
        "R0_ci_lo": 0,
        "R0_ci_hi": round(R0, 4),
        "I0": round(I0, 4),
        "fit_mse": round(mse, 6),
    })

    print(f"{trend_name}: beta={beta:.4f}, gamma={gamma:.4f}, R0={R0:.4f}, I0={I0:.4f}")

out_path = os.path.join(OUTPUT_DIR, "sir_parameters.csv")
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")