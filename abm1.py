import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# 1) SETUP
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

threshold = 0.35
buffer = 5

files = [
    ("Microtrends", "barbiecore.csv"),
    ("Macrotrends", "Cottagecore.csv"),
    ("MegaTrends", "INDIE.csv")
]

np.random.seed(42)

# =========================================================
# 2) ABM FUNCTION
# =========================================================

def simulate_trend(
    n_people=1000,
    n_steps=260,
    social_weight=0.5,
    start_chance=0.005,
    fade_base=0.03,
    peak_pos=0.4,
    peak_height=2.0,
    start_share=0.01,
    late_drop=0.08,
    noise=0.005,
    start_delay=0,
    trend_type="Macrotrends"
):
    state = np.zeros(n_people)
    state[:int(start_share * n_people)] = 1
    np.random.shuffle(state)

    series = np.zeros(n_steps)

    uptake_weight = np.random.uniform(0.9, 1.1, n_people)
    fade_weight = np.random.uniform(0.9, 1.1, n_people)

    t = np.arange(n_steps)

    # Different wave shape for Mega trends
    if trend_type == "MegaTrends":
        wave = 1 + 0.6 * (1 / (1 + np.exp(-0.05 * (t - n_steps * 0.3))))
    else:
        main_peak = 1 + peak_height * np.exp(-((t - n_steps * peak_pos) / 3.0) ** 2)
        bump_1 = 0.05 * np.exp(-((t - n_steps * (peak_pos + 0.08)) / 4.0) ** 2)
        bump_2 = 0.03 * np.exp(-((t - n_steps * (peak_pos + 0.18)) / 5.0) ** 2)
        wave = main_peak + bump_1 + bump_2

    wave = wave * (1 + np.random.normal(0, 0.01, n_steps))
    wave = np.clip(wave, 0.15, None)

    peak_step = int(peak_pos * n_steps)

    for i in range(n_steps):
        if i < start_delay:
            series[i] = 0
            continue

        current_share = state.mean()

        join_rate = (start_chance + social_weight * (current_share ** 1.1)) * wave[i]
        join_rate += np.random.normal(0, noise)

        # after the peak, new adoption slows sharply
        if i > peak_step:
            join_rate *= 0.15

        # after the peak, fading increases
        if i > peak_step:
            extra_fade = late_drop * 3.0 * ((i - peak_step) / max(1, (n_steps - peak_step)))
        else:
            extra_fade = 0.0

        leave_rate = fade_base + extra_fade
        leave_rate += np.random.normal(0, noise / 2)

        join_prob = np.clip(join_rate * uptake_weight, 0, 0.95)
        leave_prob = np.clip(leave_rate * fade_weight, 0, 0.95)

        draw_join = np.random.random(n_people)
        draw_leave = np.random.random(n_people)

        new_join = (state == 0) & (draw_join < join_prob)
        new_leave = (state == 1) & (draw_leave < leave_prob)

        state[new_join] = 1
        state[new_leave] = 0

        series[i] = state.mean()

    if series.max() > 0:
        series = series / series.max()

    return series

# =========================================================
# 3) PARAMETERS BY TREND CLASS
# =========================================================

param_map = {
    "Microtrends": {
        "social_weight": 0.18,
        "fade_base": 0.10,
        "late_drop": 0.35,
        "peak_height": 3.8
    },
    "Macrotrends": {
        "social_weight": 0.35,
        "fade_base": 0.045,
        "late_drop": 0.16,
        "peak_height": 2.4
    },
    "MegaTrends": {
        "social_weight": 0.50,
        "fade_base": 0.025,
        "late_drop": 0.08,
        "peak_height": 1.9
    }
}

# =========================================================
# 4) LOAD, CLIP, PLOT, SAVE
# =========================================================

for folder_name, file_name in files:
    path = os.path.join(DATA_DIR, folder_name, file_name)

    df = pd.read_csv(path)
    real = df.iloc[:, 1].fillna(0).values.astype(float)

    if real.max() > 0:
        real = real / real.max()

    n_steps = len(real)
    params = param_map[folder_name]

    active_idx = np.where(real >= threshold)[0]

    if len(active_idx) == 0:
        print(f"{file_name}: no values above threshold {threshold}")
        continue

    start_idx = max(0, active_idx[0] - buffer)
    end_idx = min(n_steps - 1, active_idx[-1] + buffer)

    peak_guess = np.argmax(real) / n_steps
    start_delay = max(0, active_idx[0] - 2)

    fake = simulate_trend(
        n_people=1000,
        n_steps=n_steps,
        social_weight=params["social_weight"],
        fade_base=params["fade_base"],
        late_drop=params["late_drop"],
        peak_height=params["peak_height"],
        peak_pos=np.clip(peak_guess, 0.1, 0.9),
        start_delay=start_delay,
        trend_type=folder_name
    )

    real_clip = real[start_idx:end_idx + 1]
    fake_clip = fake[start_idx:end_idx + 1]
    x_clip = np.arange(start_idx, end_idx + 1)

    label = file_name.replace(".csv", "").replace("_", " ").title()
    clean_name = file_name.replace(".csv", "").replace(" ", "_").lower()

    plt.figure(figsize=(10, 5))
    plt.plot(x_clip, real_clip, label="Real trend", linewidth=2)
    plt.plot(x_clip, fake_clip, label="ABM simulation", linestyle="--", linewidth=2)
    plt.axhline(threshold, color="grey", linestyle=":", linewidth=1, label="Threshold = 0.35")

    plt.title(
        f"{label} — Diffusion of trend interest through a population (ABM)",
        fontsize=11
    )
    plt.xlabel("Weeks")
    plt.ylabel("Normalised interest")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"{folder_name}_{clean_name}.png")
    print("Saving to:", save_path)
    plt.savefig(save_path, dpi=200)

    plt.show()
    plt.close()
