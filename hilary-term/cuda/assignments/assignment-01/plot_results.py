#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_matrix_sizes(df):
    """Plot performance for each matrix size across all thread configurations"""
    ensure_directory("plots")
    
    # Get unique matrix sizes
    matrix_sizes = sorted(df['n'].unique())
    
    # For each matrix size, create a plot
    for size in matrix_sizes:
        df_size = df[df['n'] == size]
        
        if len(df_size) == 0:
            continue
            
        # Sort by threads per block
        df_size = df_size.sort_values('threads_per_block')
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot speedups
        ax1.plot(df_size['threads_per_block'], df_size['row_speedup'], 'o-', label='Row Sum')
        ax1.plot(df_size['threads_per_block'], df_size['col_speedup'], 's-', label='Column Sum')
        ax1.plot(df_size['threads_per_block'], df_size['row_reduce_speedup'], '^-', label='Row Reduce')
        ax1.plot(df_size['threads_per_block'], df_size['col_reduce_speedup'], 'd-', label='Column Reduce')
        
        ax1.set_xlabel('Threads per Block')
        ax1.set_ylabel('Speedup (CPU time / GPU time)')
        ax1.set_title(f'Speedup vs Threads per Block (Matrix {size}x{size})')
        ax1.set_xscale('log', base=2)
        ax1.grid(True)
        ax1.legend()
        
        # Plot relative errors
        ax2.plot(df_size['threads_per_block'], df_size['row_error'], 'o-', label='Row Sum')
        ax2.plot(df_size['threads_per_block'], df_size['col_error'], 's-', label='Column Sum')
        
        ax2.set_xlabel('Threads per Block')
        ax2.set_ylabel('Relative Error')
        ax2.set_title(f'Precision vs Threads per Block (Matrix {size}x{size})')
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/matrix_size_{size}_threads.png')
        print(f"Generated plot for matrix size {size}x{size}")
        plt.close()

if __name__ == "__main__":
    # Default CSV file
    csv_file = "perf_results.csv"
    
    # Check if a CSV file was provided
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    print(f"Analyzing results from {csv_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Generate only the matrix size vs threads plots
        plot_matrix_sizes(df)
        
        print("\nPlots generated successfully in the 'plots' directory.")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        sys.exit(1)
