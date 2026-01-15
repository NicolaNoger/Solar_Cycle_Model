import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count


from sunspot_OP import simulated_annealing, get_init_params, solar_model, cost_function


GLOBAL_T_REAL = None
GLOBAL_Y_REAL = None

def init_worker(t_real, y_real):
    """Initializes the global data in each worker process."""
    global GLOBAL_T_REAL, GLOBAL_Y_REAL
    GLOBAL_T_REAL = t_real
    GLOBAL_Y_REAL = y_real

def run_single_optimization(seed_offset):
    """
    This function runs a single optimization instance with a unique random seed.
    seed_offset: an integer to offset the random seed for different runs
    """
    np.random.seed(int(time.time()) + seed_offset)
    
    # 1. Get initial parameters
    init_params = get_init_params()
    
   
    # using (slightly) different initial conditions
    noise = np.random.normal(0, 0.1, len(init_params))

    for i in range(0, len(init_params), 3):
        init_params[i] += np.random.normal(0, 0.5) # noise
    
    # 2. Hyperparameter 
    T_start = 1000.0
    cooling_rate = 0.995
    iterations = 100000 
    
    # 3. Start optimization
    best_params, best_mse, _ = simulated_annealing(
        init_params, GLOBAL_T_REAL, GLOBAL_Y_REAL, T_start, cooling_rate, iterations
    )
    
    return best_params, best_mse

def main():
    # 1. load data
    data = pd.read_csv("data_Team12.csv")
    t_real = data["Time"].values.astype(np.float64)
    y_real = data["SN"].values.astype(np.float64)
    
    # 2. Parallelization Setup
    num_cores = 32 # Oder cpu_count() für lokal
    try:
        real_cores = cpu_count()
        if num_cores > real_cores:
            num_cores = real_cores
    except:
        pass
        
    
    # 3. Create pool and distribute work
    tasks = range(num_cores)
    
    # Pool runs 'run_single_optimization' in parallel
    with Pool(processes=num_cores, initializer=init_worker, initargs=(t_real, y_real)) as pool:
        results = pool.map(run_single_optimization, tasks)
        

    
    # 4. Analyse results
    all_params = np.array([r[0] for r in results])
    all_mses = np.array([r[1] for r in results])
    
    # Bester einzelner Run (nur zur Info)
    best_idx = np.argmin(all_mses)
    print(f"Best run MSE: {all_mses[best_idx]:.4f}")
    
    # CENTER OF MASS
    final_params = np.mean(all_params, axis=0)
    
    final_mse = cost_function(final_params, t_real, y_real)
    print(f"Center of Mass (Final) MSE: {final_mse:.4f}")
    
    # 5. Plotten
    y_best = solar_model(t_real, final_params)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_real, y_real, label="Real Data", color="gray", alpha=0.5)
    
        
    plt.plot(t_real, y_best, label="Center of Mass Model", color="red", linewidth=2)
    plt.title(f"HPC Optimization Result (Avg of {num_cores} runs)")
    plt.legend()
    plt.savefig("hpc_result.png")
    plt.show()

if __name__ == "__main__":
    main()