import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_folder = "/Users/rose/Desktop/new fashion3/data/"
files = ["cottagecore.csv", "dark_academia.csv", "millennial_pink.csv", "soft_grunge.csv"]


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
    noise=0.01
):
    state = np.zeros(n_people)
    state[:int(start_share * n_people)] = 1
    np.random.shuffle(state)

    series = np.zeros(n_steps)

    uptake_weight = np.random.uniform(0.8, 1.2, n_people)
    fade_weight = np.random.uniform(0.9, 1.1, n_people)

    t = np.arange(n_steps)

    main_peak = 1 + peak_height * np.exp(-((t - n_steps * peak_pos) / 5) ** 2)
    bump_1 = 0.18 * np.exp(-((t - n_steps * (peak_pos + 0.12)) / 3) ** 2)
    bump_2 = 0.10 * np.exp(-((t - n_steps * (peak_pos + 0.25)) / 4) ** 2)

    wave = main_peak + bump_1 + bump_2
    wave = wave * (1 + np.random.normal(0, 0.03, n_steps))
    wave = np.clip(wave, 0.2, None)

    time_drag = (t / n_steps) ** 2

    peak_step = int(peak_pos * n_steps)
    fade_start = peak_step + int(0.1 * n_steps)

    for i in range(n_steps):
        current_share = state.mean()

        join_rate = (start_chance + social_weight * (current_share ** 1.1)) * wave[i]
        join_rate += np.random.normal(0, noise)

        if i > fade_start:
            extra_fade = late_drop * ((i - fade_start) / (n_steps - fade_start))
        else:
            extra_fade = 0.0

        leave_rate = fade_base + 0.28 * time_drag[i] + extra_fade
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


for name in files:
    path = os.path.join(data_folder, name)
    df = pd.read_csv(path)
    real = df.iloc[:, 1].fillna(0).values

    if real.max() > 0:
        real = real / real.max()

    n_steps = len(real)
    peak_guess = np.argmax(real) / n_steps

    best_mse = 10**9
    best_line = None
    best_set = None

    for social_weight in [0.3, 0.5, 0.7]:
        for fade_base in [0.02, 0.03, 0.04]:
            for peak_shift in [-0.08, -0.04, 0, 0.04]:
                for late_drop in [0.04, 0.08, 0.12]:
                    fake = simulate_trend(
                        n_people=1000,
                        n_steps=n_steps,
                        social_weight=social_weight,
                        fade_base=fade_base,
                        peak_pos=np.clip(peak_guess + peak_shift, 0.05, 0.9),
                        late_drop=late_drop
                    )

                    mse = np.mean((fake - real) ** 2)

                    if mse < best_mse:
                        best_mse = mse
                        best_line = fake
                        best_set = (social_weight, fade_base, peak_shift, late_drop)

    label = name.replace(".csv", "").replace("_", " ").title()

    print(f"{label} | best = {best_set} | mse = {best_mse:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(real, label="Real", linewidth=2)
    plt.plot(best_line, label="Model", linestyle="--", linewidth=2)
    plt.title(f"Trend Fit: {label}", fontsize=12)
    plt.xlabel("Weeks")
    plt.ylabel("Normalized Interest")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()