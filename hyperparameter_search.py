import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sunspot_OP import simulated_annealing, get_init_params, solar_model, cost_function
import time

def hyperparameter_search():
    # Load the data
    data = pd.read_csv("data_Team12.csv")
    t_real = data["Time"].values.astype(np.float64)
    y_real = data["SN"].values.astype(np.float64)
    
    # Define hyperparameter grid (optimizing T_start and sigma only)
    T_starts = [500.0, 750.0, 1000.0, 1500.0, 2000.0]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # Fixed hyperparameters
    cooling_rate = 0.95
    iterations = 5000
    
    # Store results
    results = []
    total_runs = len(T_starts) * len(sigmas)
    current_run = 0
    
    start_time = time.time()
    
    for T_start in T_starts:
        for sigma in sigmas:
            current_run += 1
            
            # Get initial parameters
            init_params = get_init_params()
            
            # Run optimization
            print(f"Run {current_run}/{total_runs}: T_start={T_start}, sigma={sigma}")
            
            best_params, best_mse, history = simulated_annealing(
                init_params, t_real, y_real, T_start, cooling_rate, iterations, sigma
            )
            
            # Store results
            results.append({
                'T_start': T_start,
                'sigma': sigma,
                'best_mse': best_mse,
                'final_temp': T_start * (cooling_rate ** iterations)
            })
    
    elapsed_time = time.time() - start_time
    
    # Convert to DataFrame and sort by MSE
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values('best_mse')
    
    # Display results
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*80)
    print(f"Optimizing: T_start, sigma")
    print(f"Fixed: cooling_rate={cooling_rate}, iterations={iterations}")
    print(f"\nTotal runs: {total_runs}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print("\nTop 10 Best Results:")
    print(results_df_sorted.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION:")
    best_row = results_df_sorted.iloc[0]
    print(f"T_start: {best_row['T_start']}")
    print(f"sigma: {best_row['sigma']}")
    print(f"Best MSE: {best_row['best_mse']:.4f}")
    print("="*80)
    
    # Save results to CSV
    results_df_sorted.to_csv("hyperparameter_results.csv", index=False)
    print("\nResults saved to hyperparameter_results.csv")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a heatmap of T_start vs sigma
    pivot_table = results_df.pivot_table(values='best_mse', index='sigma', columns='T_start')
    
    im = ax.imshow(pivot_table.values, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_xticklabels([f'{x:.0f}' for x in pivot_table.columns])
    ax.set_yticklabels([f'{y:.1f}' for y in pivot_table.index])
    ax.set_xlabel('T_start', fontsize=12)
    ax.set_ylabel('sigma', fontsize=12)
    ax.set_title('MSE Heatmap: T_start vs sigma', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Best MSE', fontsize=12)
    
    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            text = ax.text(j, i, f'{pivot_table.values[i, j]:.1f}',
                          ha="center", va="center", color="white", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("hyperparameter_analysis.png", dpi=150)
    print("Analysis plot saved to hyperparameter_analysis.png")
    plt.show()
    
    return results_df_sorted

if __name__ == "__main__":
    hyperparameter_search()
