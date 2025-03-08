import matplotlib.pyplot as plt
import os
import pandas as pd
import sys


def ensure_directory(directory):
    """Creates the specified directory if it doesn't already exist.

    Args:
        directory (str): Path to the directory that should be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_matrix_sizes(df):
    """Generates performance plots for different matrix sizes.

    Args:
        df (pandas.DataFrame): Contains benchmark results.
    """
    ensure_directory("plots")
    matrix_sizes = sorted(df["n"].unique())
    for size in matrix_sizes:
        df_size = df[df["n"] == size]
        if len(df_size) == 0:
            continue
        df_size = df_size.sort_values("threads_per_block")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(
            df_size["threads_per_block"], df_size["row_speedup"], "o-", label="Row Sum"
        )
        ax1.plot(
            df_size["threads_per_block"],
            df_size["col_speedup"],
            "s-",
            label="Column Sum",
        )
        ax1.plot(
            df_size["threads_per_block"],
            df_size["row_reduce_speedup"],
            "^-",
            label="Row Reduction",
        )
        ax1.plot(
            df_size["threads_per_block"],
            df_size["col_reduce_speedup"],
            "d-",
            label="Column Reduction",
        )
        ax1.set_xlabel("Threads per Block")
        ax1.set_ylabel("Speedup")
        ax1.set_title(f"Speedup vs. Threads per Block ({size}x{size})")
        ax1.set_xscale("log", base=2)
        ax1.grid(True)
        ax1.legend()
        ax2.plot(
            df_size["threads_per_block"], df_size["row_error"], "o-", label="Row Sum"
        )
        ax2.plot(
            df_size["threads_per_block"], df_size["col_error"], "s-", label="Column Sum"
        )
        ax2.set_xlabel("Threads per Block")
        ax2.set_ylabel("Relative Error")
        ax2.set_title(f"Precision vs. Threads per Block ({size}x{size})")
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log")
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{size}.png")
        print(f"Generated plot for matrix size {size}x{size}")
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot.py <csv_file>")
        sys.exit(1)
    csv_file = sys.argv[1]
    print(f"Analysing results from {csv_file}...")
    df = pd.read_csv(csv_file)
    plot_matrix_sizes(df)
    print("Plots generated successfully")
