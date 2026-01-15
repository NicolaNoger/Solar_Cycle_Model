import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sunspot_OP import simulated_annealing, get_init_params, solar_model, cost_function

def main():
    # Load the date
    data = pd.read_csv("data_Team12.csv")
    # Convert to numpy arrays
    t_real = data["Time"].values.astype(np.float64)
    y_real = data["SN"].values.astype(np.float64)

    # 2. Setup
    init_params = get_init_params()
    
    # HYPERPARAMETER SETUP
    T_start = 750
    sigma = 0.2
    cooling_rate = 0.95
    iterations = 100000
    
    print(f"Start-MSE: {cost_function(init_params, t_real, y_real):.4f}")
    
    # 3. Optimisation with Simulated Annealing
    best_params, best_mse, history = simulated_annealing(
        init_params, t_real, y_real, T_start, cooling_rate, iterations, sigma
    )

    print(f"Best MSE: {best_mse:.4f}")
    
    # 4. Plot results
    # Calculate the curve with the best parameters
    y_best = solar_model(t_real, best_params)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_real, y_real, label="Real Data", color="gray", alpha=0.5)
    plt.plot(t_real, y_best, label="Your Model (SA)", color="red", linewidth=2)
    plt.title(f"Solar Cycle Fit (MSE: {best_mse:.2f})")
    plt.xlabel("Year")
    plt.ylabel("Sunspot Number")
    plt.legend()
    plt.grid(True)
    plt.savefig("solar_fit.png") # Save the image
    plt.show()

if __name__ == "__main__":
    main()