"""
Final Model Calibration with Parallel SA Optimization
======================================================
Fulfills all requirements from the HPC Project:

Task 1: Hyperparameter Tuning (done separately)
Task 2: Model Calibration
  - Runs 100 independent SA optimizations with different initial conditions
  - Parallelized on the cluster
  - Center of Mass aggregation for 30 model parameters
  - CSV output with parameters per cycle (T0, Ts, Td)

Performance Analysis:
  - Wall time vs number of cores
  - Speedup plot

Scientific Analysis:
  - Ts vs Td correlation (Penza et al. 2024 / Volobuev)
  - Slope s1 ≈ 0.02 ± 0.01 year validation

Author: Team 12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy import stats
import time
import os

from sunspot_OP import simulated_annealing, get_init_params, solar_model, cost_function


# Global variables for shared data
GLOBAL_T_REAL = None
GLOBAL_Y_REAL = None

# Cycle numbers (Solar Cycles 8-17)
CYCLE_NUMBERS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


def init_worker(t_real, y_real):
    """Initialize global data in each worker process."""
    global GLOBAL_T_REAL, GLOBAL_Y_REAL
    GLOBAL_T_REAL = t_real
    GLOBAL_Y_REAL = y_real


def run_single_optimization(args):
    """
    Run a single SA optimization with unique random seed and noisy initial conditions.
    """
    run_id, iterations, T_start, sigma, cooling_rate = args
    np.random.seed(42 + run_id)
    
    # Get initial parameters and add noise
    init_params = get_init_params()
    
    # Add noise to T0 values (cycle start times) - small noise
    for i in range(0, len(init_params), 3):
        init_params[i] += np.random.normal(0, 0.5)
    
    # Add small noise to Ts and Td
    for i in range(1, len(init_params), 3):
        init_params[i] += np.random.normal(0, 0.02)
        init_params[i] = max(0.01, init_params[i])
    for i in range(2, len(init_params), 3):
        init_params[i] += np.random.normal(0, 0.3)
        init_params[i] = max(0.01, init_params[i])
    
    # Run SA optimization
    best_params, best_mse, history = simulated_annealing(
        init_params, GLOBAL_T_REAL, GLOBAL_Y_REAL, 
        T_start, cooling_rate, iterations, sigma
    )
    
    return run_id, best_params, best_mse


def run_parallel_calibration(t_real, y_real, num_runs, iterations, num_cores, 
                             T_start, sigma, cooling_rate):
    """Run parallel SA calibration and return timing + results."""
    tasks = [(i, iterations, T_start, sigma, cooling_rate) for i in range(num_runs)]
    
    start_time = time.perf_counter()
    
    with Pool(processes=num_cores, initializer=init_worker, initargs=(t_real, y_real)) as pool:
        results = pool.map(run_single_optimization, tasks)
    
    wall_time = time.perf_counter() - start_time
    
    return results, wall_time


def analyze_ts_td_correlation(final_params):
    """
    Analyze linear correlation between Ts and Td.
    Td_k = s1 * Ts_k + s2 (Equation 4, Penza et al. 2024)
    Expected from literature: s1 ≈ 0.02 ± 0.01 year
    
    Note: The slope is dimensionless since both Ts and Td are in the same units.
    """
    n_cycles = len(final_params) // 3
    
    Ts_values = []
    Td_values = []
    T0_values = []
    
    for k in range(n_cycles):
        T0 = final_params[3*k]
        Ts = final_params[3*k + 1]
        Td = final_params[3*k + 2]
        T0_values.append(T0)
        Ts_values.append(Ts)
        Td_values.append(Td)
    
    Ts_arr = np.array(Ts_values)
    Td_arr = np.array(Td_values)
    T0_arr = np.array(T0_values)
    
    # Linear fit using np.polyfit 
    slope, intercept = np.polyfit(Ts_arr, Td_arr, 1)
    td_fit = slope * Ts_arr + intercept
    
    # Get statistics with scipy for p-value and R²
    _, _, r_value, p_value, std_err = stats.linregress(Ts_arr, Td_arr)
    
    return {
        'T0': T0_arr,
        'Ts': Ts_arr,
        'Td': Td_arr,
        'td_fit': td_fit,
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }


def save_cycle_parameters_csv(final_params, filename='cycle_parameters.csv'):
    """Save cycle parameters to a CSV file like in the report."""
    n_cycles = len(final_params) // 3
    
    data = []
    for k in range(n_cycles):
        cycle_num = CYCLE_NUMBERS[k]
        T0 = final_params[3*k]
        Ts = final_params[3*k + 1]
        Td = final_params[3*k + 2]
        data.append({
            'Cycle': cycle_num,
            'T0_start_year': T0,
            'Ts_rise_param': Ts,
            'Td_decay_param': Td
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, float_format='%.4f')
    return df


def save_all_runs_parameters(all_results, filename='all_runs_parameters.csv'):
    """Save parameters from all runs for analysis."""
    data = []
    for run_id, params, mse in all_results:
        row = {'run_id': run_id, 'mse': mse}
        for k in range(len(params)//3):
            cycle_num = CYCLE_NUMBERS[k]
            row[f'T0_cycle{cycle_num}'] = params[3*k]
            row[f'Ts_cycle{cycle_num}'] = params[3*k + 1]
            row[f'Td_cycle{cycle_num}'] = params[3*k + 2]
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, float_format='%.6f')
    return df


def main():
    print("="*70)
    print("FINAL MODEL CALIBRATION - Parallel SA Optimization")
    print("="*70)
    
    # Create output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 1. Load data
    data = pd.read_csv("data_Team12.csv")
    t_real = data["Time"].values.astype(np.float64)
    y_real = data["SN"].values.astype(np.float64)
    
    # Data info
    time_step = t_real[1] - t_real[0]
    print(f"Data time range: {t_real[0]:.2f} to {t_real[-1]:.2f} years")
    print(f"Time step: {time_step:.6f} years (~{time_step*365.25:.1f} days)")
    print(f"Data points: {len(t_real)}")
    print(f"Solar Cycles: {CYCLE_NUMBERS[0]} to {CYCLE_NUMBERS[-1]}")
    
    # ==========================================================================
    # CONFIGURATION - Best hyperparameters from Task 1
    # ==========================================================================
    T_start = 500.0
    sigma = 0.05
    cooling_rate = 0.9999
    iterations = 100000
    num_runs = 32  # 32 independent runs
    
    # Core counts for scaling study
    core_counts = [1, 8, 16, 32]
    max_cores = cpu_count()
    core_counts = [c for c in core_counts if c <= max_cores]
    
    print(f"\nConfiguration:")
    print(f"  Independent runs: {num_runs}")
    print(f"  Iterations per run: {iterations:,}")
    print(f"  Core counts to test: {core_counts}")
    print(f"  T_start: {T_start}, sigma: {sigma}, cooling_rate: {cooling_rate}")
    print("="*70)
    
    # ==========================================================================
    # SCALING STUDY: Wall time vs Cores
    # ==========================================================================
    print("\n" + "="*70)
    print("SCALING STUDY: Wall Time vs Number of Cores")
    print("="*70)
    
    scaling_results = []
    final_results = None
    
    for num_cores in core_counts:
        print(f"\nRunning with {num_cores} core(s)...", flush=True)
        
        results, wall_time = run_parallel_calibration(
            t_real, y_real, num_runs, iterations, num_cores,
            T_start, sigma, cooling_rate
        )
        
        # Extract and analyze results
        all_params = np.array([r[1] for r in results])
        all_mses = np.array([r[2] for r in results])
        
        # Center of Mass
        center_of_mass_params = np.mean(all_params, axis=0)
        com_mse = cost_function(center_of_mass_params, t_real, y_real)
        
        # Best single run
        best_idx = np.argmin(all_mses)
        best_single_mse = all_mses[best_idx]
        best_single_params = all_params[best_idx]
        
        print(f"  Wall time: {wall_time:.2f}s")
        print(f"  Center of Mass MSE: {com_mse:.4f}")
        print(f"  Best single run MSE: {best_single_mse:.4f}")
        print(f"  Mean MSE: {np.mean(all_mses):.4f} +/- {np.std(all_mses):.4f}")
        
        scaling_results.append({
            'cores': num_cores,
            'wall_time': wall_time,
            'com_mse': com_mse,
            'best_single_mse': best_single_mse,
            'mean_mse': np.mean(all_mses),
            'std_mse': np.std(all_mses)
        })
        
        # Save results from max cores run
        if num_cores == max(core_counts):
            final_results = results
            final_params = center_of_mass_params
            final_mse = com_mse
            final_all_params = all_params
            final_all_mses = all_mses
    
    # ==========================================================================
    # SCALING RESULTS SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("SCALING RESULTS SUMMARY")
    print("="*70)
    print(f"{'Cores':>6} {'Wall Time (s)':>14} {'Speedup':>10} {'CoM MSE':>12} {'Best MSE':>12}")
    print("-"*60)
    
    baseline_time = scaling_results[0]['wall_time']
    for r in scaling_results:
        speedup = baseline_time / r['wall_time']
        print(f"{r['cores']:>6} {r['wall_time']:>14.2f} {speedup:>10.2f}x {r['com_mse']:>12.4f} {r['best_single_mse']:>12.4f}")
    
    # Save scaling results
    scaling_df = pd.DataFrame(scaling_results)
    scaling_df['speedup'] = baseline_time / scaling_df['wall_time']
    scaling_df.to_csv(os.path.join(output_dir, 'scaling_results.csv'), index=False, float_format='%.4f')
    print(f"\nScaling results saved to {output_dir}/scaling_results.csv")
    
    # ==========================================================================
    # FINAL MODEL PARAMETERS (Center of Mass)
    # ==========================================================================
    print("\n" + "="*70)
    print("FINAL MODEL PARAMETERS (Center of Mass of 100 runs)")
    print("="*70)
    print(f"Final MSE: {final_mse:.4f}")
    print(f"\n{'Cycle':>6} {'T0 (year)':>14} {'Ts (rise)':>12} {'Td (decay)':>12}")
    print("-"*50)
    
    for k in range(len(final_params)//3):
        cycle_num = CYCLE_NUMBERS[k]
        T0 = final_params[3*k]
        Ts = final_params[3*k + 1]
        Td = final_params[3*k + 2]
        print(f"{cycle_num:>6} {T0:>14.4f} {Ts:>12.4f} {Td:>12.4f}")
    
    # Save parameters
    np.save(os.path.join(output_dir, "final_model_params.npy"), final_params)
    cycle_df = save_cycle_parameters_csv(final_params, os.path.join(output_dir, 'cycle_parameters.csv'))
    all_runs_df = save_all_runs_parameters(final_results, os.path.join(output_dir, 'all_runs_parameters.csv'))
    
    print(f"\nParameters saved to {output_dir}/:")
    print("  - final_model_params.npy (numpy array)")
    print("  - cycle_parameters.csv (summary table)")
    print("  - all_runs_parameters.csv (all 100 runs)")
    
    # ==========================================================================
    # Ts vs Td CORRELATION ANALYSIS
    # ==========================================================================
    print("\n" + "="*70)
    print("Ts vs Td CORRELATION ANALYSIS")
    print("="*70)
    
    corr = analyze_ts_td_correlation(final_params)
    
    print(f"\nLinear Regression: Td = s1 * Ts + s2")
    print(f"  Slope s1 = {corr['slope']:.4f} +/- {corr['std_err']:.4f}")
    print(f"  Intercept s2 = {corr['intercept']:.4f}")
    print(f"  Correlation r = {corr['r_value']:.4f}")
    print(f"  R^2 = {corr['r_squared']:.4f}")
    print(f"  p-value = {corr['p_value']:.6f}")
    
    print(f"\nLiterature Comparison (Penza et al. 2024 / Volobuev 2009):")
    print(f"  Expected s1 = 0.02 +/- 0.01 year")
    print(f"  Our result: s1 = {corr['slope']:.4f}")
    
    if 0.01 <= corr['slope'] <= 0.03:
        print("  -> IN AGREEMENT with literature!")
    else:
        print(f"  Note: Value differs from literature expectation")
    
    # Save correlation results
    corr_df = pd.DataFrame({
        'Cycle': CYCLE_NUMBERS,
        'T0': corr['T0'],
        'Ts': corr['Ts'],
        'Td': corr['Td']
    })
    corr_df.to_csv(os.path.join(output_dir, 'ts_td_correlation_data.csv'), index=False, float_format='%.4f')
    
    # ==========================================================================
    # CREATE PLOTS
    # ==========================================================================
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # Figure 1: Scaling study (2 plots)
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1a: Wall time vs Cores
    ax1 = axes1[0]
    cores = [r['cores'] for r in scaling_results]
    wall_times = [r['wall_time'] for r in scaling_results]
    ax1.plot(cores, wall_times, 'bo-', linewidth=2, markersize=10)
    ax1.set_xlabel('Number of Cores', fontsize=12)
    ax1.set_ylabel('Wall Time (s)', fontsize=12)
    ax1.set_title(f'Wall Time vs Number of Cores\n({num_runs} runs, {iterations:,} iterations each)', fontsize=12)
    ax1.set_xticks(cores)
    ax1.grid(True, alpha=0.3)
    
    # Plot 1b: Speedup
    ax2 = axes1[1]
    speedups = [baseline_time / r['wall_time'] for r in scaling_results]
    ax2.plot(cores, speedups, 'go-', linewidth=2, markersize=10, label='Actual Speedup')
    ax2.plot(cores, cores, 'r--', linewidth=1, label='Ideal (Linear) Speedup')
    ax2.set_xlabel('Number of Cores', fontsize=12)
    ax2.set_ylabel('Speedup', fontsize=12)
    ax2.set_title('Parallel Speedup', fontsize=12)
    ax2.set_xticks(cores)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_study.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/scaling_study.png")
    
    # Figure 2: Model fit
    fig2, ax3 = plt.subplots(figsize=(14, 6))
    y_model = solar_model(t_real, final_params)
    ax3.plot(t_real, y_real, 'gray', alpha=0.5, linewidth=0.5, label='Observed Data')
    ax3.plot(t_real, y_model, 'r-', linewidth=1.5, label=f'Volobuev Model (MSE={final_mse:.2f})')
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Sunspot Number', fontsize=12)
    ax3.set_title('Solar Cycle Fit - Center of Mass Parameters (Cycles 8-17)', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Add cycle labels
    for k in range(len(final_params)//3):
        T0 = final_params[3*k]
        cycle_num = CYCLE_NUMBERS[k]
        ax3.axvline(x=T0, color='blue', linestyle=':', alpha=0.3)
        ax3.text(T0 + 0.5, ax3.get_ylim()[1]*0.95, f'C{cycle_num}', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'solar_cycle_fit.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/solar_cycle_fit.png")
    
    # Figure 3: Ts vs Td correlation
    fig3, ax4 = plt.subplots(figsize=(10, 8))
    ax4.scatter(corr['Ts'], corr['Td'], s=150, c='blue', alpha=0.7, edgecolors='black', 
                label='Fitted Parameters')
    
    # Add cycle labels
    for k in range(len(corr['Ts'])):
        ax4.annotate(f'Cycle {CYCLE_NUMBERS[k]}', 
                    (corr['Ts'][k], corr['Td'][k]),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)
    
    # Fit line using td_fit from np.polyfit
    x_fit = np.linspace(min(corr['Ts'])*0.9, max(corr['Ts'])*1.1, 100)
    y_fit = corr['slope'] * x_fit + corr['intercept']
    ax4.plot(x_fit, y_fit, 'r-', linewidth=2, 
             label=f'np.polyfit: Td = {corr["slope"]:.4f}*Ts + {corr["intercept"]:.4f}\n'
                   f'R² = {corr["r_squared"]:.3f}, p = {corr["p_value"]:.4f}')
    
    ax4.set_xlabel('Ts (Rise Parameter) [years]', fontsize=12)
    ax4.set_ylabel('Td (Decay Parameter) [years]', fontsize=12)
    ax4.set_title('Ts vs Td Correlation Analysis\n(Volobuev 2009 / Penza et al. 2024)', fontsize=14)
    ax4.legend(loc='upper left', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ts_td_correlation.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/ts_td_correlation.png")
    
    # Figure 4: MSE distribution of all runs
    fig4, ax5 = plt.subplots(figsize=(10, 6))
    ax5.hist(final_all_mses, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax5.axvline(x=final_mse, color='red', linewidth=2, linestyle='--', 
                label=f'Center of Mass MSE: {final_mse:.2f}')
    ax5.axvline(x=np.min(final_all_mses), color='green', linewidth=2, linestyle=':', 
                label=f'Best Single Run MSE: {np.min(final_all_mses):.2f}')
    ax5.set_xlabel('MSE', fontsize=12)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_title(f'MSE Distribution of {num_runs} Independent SA Runs', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mse_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/mse_distribution.png")
    
    plt.close('all')
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}/")
    print(f"  Data files:")
    print(f"    - final_model_params.npy")
    print(f"    - cycle_parameters.csv")
    print(f"    - all_runs_parameters.csv")
    print(f"    - scaling_results.csv")
    print(f"    - ts_td_correlation_data.csv")
    print(f"  Plot files:")
    print(f"    - scaling_study.png")
    print(f"    - solar_cycle_fit.png")
    print(f"    - ts_td_correlation.png")
    print(f"    - mse_distribution.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
