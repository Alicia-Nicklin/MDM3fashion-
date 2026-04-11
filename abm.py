import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

df = pd.read_csv("trends_merged_GB_monthly.csv")
df['Time'] = pd.to_datetime(df['Time'])

trend_name = "Skinny Jeans"
real_data = df[trend_name].values
real_data = real_data / max(real_data)


def run_abm(N=1000, T=260,
            peer_strength=0.25,
            base_prob=0.005,
            decay_base=0.02,
            seasonal_amp=0.5):

    agents = np.zeros(N)
    seed = random.sample(range(N), 1)
    for i in seed:
        agents[i] = 1

    history = []

    for t in range(T):
        new_agents = agents.copy()
        #当前流行程度的一个情况
        adoption_level = agents.sum() / N
        #季节波动
        season = 1 + seasonal_amp * np.sin(2 * np.pi * t / 12)
        #越到后期越容易过气的一个情况
        time_penalty = (t / T) ** 6

        for i in range(N):

            if agents[i] == 0:
                neighbours = random.sample(range(N), 15)
                peer_ratio = sum(agents[j] for j in neighbours) / 15#衡量有多少人穿，越多人穿越流行的一个情况

                prob = (base_prob + peer_strength * (peer_ratio ** 2)) * season * (1 - time_penalty)

                if random.random() < max(0, prob):
                    new_agents[i] = 1

            else:
                #adoption 是为我们的 评判有多少接受这个， 越多人接受趋势不再更新
                decay = decay_base + 0.02 * adoption_level + 0.5 * time_penalty

                if random.random() < decay:
                    new_agents[i] = 0

        agents = new_agents
        history.append(agents.sum() / N)
    res = np.array(history)
    return res / res.max() if res.max() > 0 else res

best_error = 1e9
best_params = None
best_sim = None

for peer in [0.25, 0.35]:
    for base in [0.0005, 0.001]:
        for d_base in [0.01, 0.02]:
            for s_amp in [0.4, 0.5]:

                sim = run_abm(peer_strength=peer,
                              base_prob=base,
                              decay_base=d_base,
                              seasonal_amp=s_amp,
                              T=len(real_data))

                error = np.mean((sim - real_data) ** 2)

                if error < best_error:
                    best_error = error
                    best_params = (peer, base, d_base, s_amp)
                    best_sim = sim

print("best params[peer(社交影响),base(自然跟风),decaybase, mseasonal]:", best_params)
#best_

plt.figure(figsize=(12, 6))
plt.plot(real_data, label="real", alpha=0.3, color='black')
plt.plot(best_sim, label="abm", linewidth=2, color='orange')
plt.title(f"{trend_name}")
plt.legend()
plt.show()