"""
Parallelized Hyperparameter Search for Simulated Annealing
===========================================================
This script performs a grid search over SA hyperparameters using multiprocessing.
Optimized for HPC execution with multiple CPU cores.

Author: Team 12
Date: 2026-01-15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import itertools
import argparse

from sunspot_OP import simulated_annealing, get_init_params, solar_model, cost_function


# Global variables for shared data (initialized in worker processes)
GLOBAL_T_REAL = None
GLOBAL_Y_REAL = None


def init_worker(t_real, y_real):
    """Initialize global data in each worker process to avoid data copying overhead."""
    global GLOBAL_T_REAL, GLOBAL_Y_REAL
    GLOBAL_T_REAL = t_real
    GLOBAL_Y_REAL = y_real


def run_single_hyperparameter_config(config):
    """
    Run a single SA optimization with given hyperparameters.
    
    Args:
        config: tuple of (config_id, T_start, sigma, cooling_rate, iterations, num_repeats)
    
    Returns:
        dict with hyperparameters and results
    """
    config_id, T_start, sigma, cooling_rate, iterations, num_repeats = config
    
    # Run multiple times to get stable results (SA is stochastic)
    mse_results = []
    best_overall_mse = float('inf')
    best_overall_params = None
    
    for repeat in range(num_repeats):
        # Set different random seed for each repeat
        np.random.seed(int(time.time() * 1000) % (2**31) + config_id * 1000 + repeat)
        
        # Get initial parameters
        init_params = get_init_params()
        
        # Run optimization
        best_params, best_mse, _ = simulated_annealing(
            init_params, GLOBAL_T_REAL, GLOBAL_Y_REAL, 
            T_start, cooling_rate, iterations, sigma
        )
        
        mse_results.append(best_mse)
        
        if best_mse < best_overall_mse:
            best_overall_mse = best_mse
            best_overall_params = best_params.copy()
    
    return {
        'config_id': config_id,
        'T_start': T_start,
        'sigma': sigma,
        'cooling_rate': cooling_rate,
        'iterations': iterations,
        'best_mse': best_overall_mse,
        'mean_mse': np.mean(mse_results),
        'std_mse': np.std(mse_results),
        'min_mse': np.min(mse_results),
        'max_mse': np.max(mse_results),
        'best_params': best_overall_params
    }


def hyperparameter_search_parallel(num_cores=None, iterations=20000, num_repeats=2):
    """
    Perform parallelized hyperparameter grid search.
    
    Args:
        num_cores: Number of CPU cores to use (default: all available)
        iterations: Number of SA iterations per run
        num_repeats: Number of repeats per configuration for stability
    """
    print("="*80)
    print("PARALLELIZED HYPERPARAMETER SEARCH FOR SIMULATED ANNEALING")
    print("="*80)
    
    # Load the data
    data = pd.read_csv("data_Team12.csv")
    t_real = data["Time"].values.astype(np.float64)
    y_real = data["SN"].values.astype(np.float64)
    
    # Determine number of cores
    if num_cores is None:
        num_cores = cpu_count()
    else:
        num_cores = min(num_cores, cpu_count())
    
    print(f"Using {num_cores} CPU cores")
    print(f"Iterations per run: {iterations}")
    print(f"Repeats per config: {num_repeats}")
    
    # ==========================================================================
    # HYPERPARAMETER GRID - Optimized for ~30 min runtime
    # ==========================================================================
    T_starts = [100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0]
    sigmas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
    cooling_rates = [0.95]  # Fixed cooling rate to reduce search space
    
    # Create all combinations
    all_configs = list(itertools.product(T_starts, sigmas, cooling_rates))
    total_configs = len(all_configs)
    
    print(f"\nHyperparameter Grid:")
    print(f"  T_start values: {T_starts}")
    print(f"  sigma values: {sigmas}")
    print(f"  cooling_rate values: {cooling_rates}")
    print(f"\nTotal configurations: {total_configs}")
    print(f"Total SA runs: {total_configs * num_repeats}")
    print("="*80 + "\n")
    
    # Prepare task list with config_id
    tasks = [
        (i, T_start, sigma, cooling_rate, iterations, num_repeats)
        for i, (T_start, sigma, cooling_rate) in enumerate(all_configs)
    ]
    
    # Run parallel search
    start_time = time.time()
    
    with Pool(processes=num_cores, initializer=init_worker, initargs=(t_real, y_real)) as pool:
        # Use imap for progress tracking
        results = []
        for i, result in enumerate(pool.imap_unordered(run_single_hyperparameter_config, tasks)):
            results.append(result)
            # Progress update
            if (i + 1) % 10 == 0 or (i + 1) == total_configs:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (total_configs - i - 1)
                print(f"Progress: {i+1}/{total_configs} ({100*(i+1)/total_configs:.1f}%) "
                      f"- Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
    
    elapsed_time = time.time() - start_time
    
    # Extract best params before converting to DataFrame
    best_params_list = [r.pop('best_params') for r in results]
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values('best_mse')
    
    # Display results
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*80)
    print(f"Total configurations tested: {total_configs}")
    print(f"Total SA runs: {total_configs * num_repeats}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Average time per config: {elapsed_time/total_configs:.2f} seconds")
    
    print("\n" + "-"*80)
    print("TOP 20 BEST CONFIGURATIONS (sorted by best_mse):")
    print("-"*80)
    display_cols = ['T_start', 'sigma', 'cooling_rate', 'iterations', 
                    'best_mse', 'mean_mse', 'std_mse']
    print(results_df_sorted[display_cols].head(20).to_string(index=False))
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION:")
    print("="*80)
    best_row = results_df_sorted.iloc[0]
    best_idx = results_df_sorted.iloc[0]['config_id']
    print(f"  T_start:      {best_row['T_start']}")
    print(f"  sigma:        {best_row['sigma']}")
    print(f"  cooling_rate: {best_row['cooling_rate']}")
    print(f"  iterations:   {best_row['iterations']}")
    print(f"  Best MSE:     {best_row['best_mse']:.4f}")
    print(f"  Mean MSE:     {best_row['mean_mse']:.4f} ± {best_row['std_mse']:.4f}")
    print("="*80)
    
    # Save results
    output_file = f"hyperparameter_results_parallel_{iterations}iter.csv"
    results_df_sorted.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Create visualizations
    create_visualizations(results_df, iterations)
    
    return results_df_sorted, best_params_list


def create_visualizations(results_df, iterations):
    """Create visualization plots for hyperparameter analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Heatmap: T_start vs sigma (averaged over cooling_rates)
    ax1 = axes[0, 0]
    pivot1 = results_df.pivot_table(values='best_mse', index='sigma', columns='T_start', aggfunc='min')
    im1 = ax1.imshow(pivot1.values, cmap='viridis', aspect='auto')
    ax1.set_xticks(range(len(pivot1.columns)))
    ax1.set_yticks(range(len(pivot1.index)))
    ax1.set_xticklabels([f'{x:.0f}' for x in pivot1.columns], rotation=45)
    ax1.set_yticklabels([f'{y:.2f}' for y in pivot1.index])
    ax1.set_xlabel('T_start', fontsize=12)
    ax1.set_ylabel('sigma', fontsize=12)
    ax1.set_title('Best MSE: T_start vs sigma\n(min over cooling_rates)', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Best MSE')
    
    # 2. Heatmap: T_start vs cooling_rate
    ax2 = axes[0, 1]
    pivot2 = results_df.pivot_table(values='best_mse', index='cooling_rate', columns='T_start', aggfunc='min')
    im2 = ax2.imshow(pivot2.values, cmap='viridis', aspect='auto')
    ax2.set_xticks(range(len(pivot2.columns)))
    ax2.set_yticks(range(len(pivot2.index)))
    ax2.set_xticklabels([f'{x:.0f}' for x in pivot2.columns], rotation=45)
    ax2.set_yticklabels([f'{y:.3f}' for y in pivot2.index])
    ax2.set_xlabel('T_start', fontsize=12)
    ax2.set_ylabel('cooling_rate', fontsize=12)
    ax2.set_title('Best MSE: T_start vs cooling_rate\n(min over sigmas)', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Best MSE')
    
    # 3. Heatmap: sigma vs cooling_rate
    ax3 = axes[1, 0]
    pivot3 = results_df.pivot_table(values='best_mse', index='cooling_rate', columns='sigma', aggfunc='min')
    im3 = ax3.imshow(pivot3.values, cmap='viridis', aspect='auto')
    ax3.set_xticks(range(len(pivot3.columns)))
    ax3.set_yticks(range(len(pivot3.index)))
    ax3.set_xticklabels([f'{x:.2f}' for x in pivot3.columns], rotation=45)
    ax3.set_yticklabels([f'{y:.3f}' for y in pivot3.index])
    ax3.set_xlabel('sigma', fontsize=12)
    ax3.set_ylabel('cooling_rate', fontsize=12)
    ax3.set_title('Best MSE: sigma vs cooling_rate\n(min over T_starts)', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='Best MSE')
    
    # 4. Box plot: MSE distribution by cooling_rate
    ax4 = axes[1, 1]
    cooling_rates = sorted(results_df['cooling_rate'].unique())
    box_data = [results_df[results_df['cooling_rate'] == cr]['best_mse'].values for cr in cooling_rates]
    bp = ax4.boxplot(box_data, labels=[f'{cr:.3f}' for cr in cooling_rates], patch_artist=True)
    ax4.set_xlabel('cooling_rate', fontsize=12)
    ax4.set_ylabel('Best MSE', fontsize=12)
    ax4.set_title('MSE Distribution by Cooling Rate', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f"hyperparameter_analysis_parallel_{iterations}iter.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Analysis plot saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Parallelized Hyperparameter Search for SA')
    parser.add_argument('--cores', type=int, default=None, 
                        help='Number of CPU cores to use (default: all available)')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of SA iterations per run (default: 10000)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Number of repeats per configuration (default: 3)')
    
    args = parser.parse_args()
    
    hyperparameter_search_parallel(
        num_cores=args.cores,
        iterations=args.iterations,
        num_repeats=args.repeats
    )


if __name__ == "__main__":
    main()
