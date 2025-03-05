import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("timing.txt")
n_values = sorted(df["n"].unique())
plt.figure(figsize=(12, 8))
colours = plt.get_cmap("tab10")
for i, n in enumerate(n_values):
    df_n = df[df["n"] == n]
    plt.plot(
        df_n["m"],
        df_n["time"],
        marker="o",
        linestyle="-",
        color=colours(i),
        label=f"n = {n}",
        linewidth=2,
        markersize=8,
    )
plt.xlabel("Number of rows (m)", fontsize=14)
plt.ylabel("Time taken (seconds)", fontsize=14)
plt.title("Time vs. Number of Rows", fontsize=16)
plt.xscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.gca().set_facecolor("#f4f4f4")
plt.tight_layout()
plt.savefig("timing.png")
