from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np
import os

# Create the directory for the plots if it doesn't exist already
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)


def simulate_scaled_random_walk(n, T):
    """_summary_

    Args:
        n (_type_): _description_
        T (_type_): _description_

    Returns:
        _type_: _description_
    """
    steps = np.random.choice([-1, 1], size=n)
    increments = steps * np.sqrt(T / n)
    W = np.zeros(n + 1)
    W[1:] = np.cumsum(increments)
    return W


# Simulate and plot random walks for the first part of the question
n_values = [10, 200, 1000]
T_values = [0.5, 1.0, 2.0]
for n in n_values:
    for T in T_values:
        W = simulate_scaled_random_walk(n, T)
        t = np.linspace(0, T, n + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(t, W, "b-", lw=0.5)
        plt.title(rf"Random Walk ($n = {n}$, $T = {T}$)")
        plt.xlabel(r"Time $t$")
        plt.ylabel(rf"$W^{{({n})}}(t)$")
        plt.grid(True)
        filename = os.path.join(save_dir, f"{n}-{T}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

# Simulate and plot 400 paths of a scaled random walk
n = 500
T = 2.0
num_paths = 400
plt.figure(figsize=(10, 6))
for _ in range(num_paths):
    W = simulate_scaled_random_walk(n, T)
    t = np.linspace(0, T, n + 1)
    plt.plot(t, W, alpha=0.5, lw=0.5)
plt.title(
    rf"Simulation of {num_paths} Paths of a Scaled Random Walk ($n = 500$, $T = 2.0$)"
)
plt.xlabel(r"Time $t$")
plt.ylabel(rf"$W^{{({n})}}(t)$")
plt.grid(True)
filename = os.path.join(save_dir, f"paths.png")
plt.savefig(filename, dpi=150, bbox_inches="tight")
plt.close()

# Simulate and plot histogram of final values
n = 1000
T = 2.0
num_sim = 10000
final_values = np.empty(num_sim)
for i in range(num_sim):
    W = simulate_scaled_random_walk(n, T)
    final_values[i] = W[-1]
plt.figure(figsize=(10, 6))
count, bins, _ = plt.hist(
    final_values, bins=50, density=True, alpha=0.6, color="skyblue"
)
x = np.linspace(bins[0], bins[-1], 200)
pdf = norm.pdf(x, 0, np.sqrt(T))
plt.plot(x, pdf, "r-", lw=2, label=r"$\mathcal{N}(0,2)$")
plt.title(r"Histogram of Final Values ($n = 1000$, $T = 2.0$)")
plt.xlabel("Final Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
filename = os.path.join(save_dir, f"histogram.png")
plt.savefig(filename, dpi=150, bbox_inches="tight")
plt.close()
