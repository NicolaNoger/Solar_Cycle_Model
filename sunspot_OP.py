import numpy as np
import pandas as pd
from numba import jit



def get_init_params():
    '''Generate initial parameters for the solar model.
    Each cycle is represented by three parameters: T0, Ts, Td.
    '''
    # Hathaway paper(Table 2)
    hathaway_minima = [
        1833.9,  # Cycle 8 
        1843.5,  # Cycle 9
        1856.0,  # Cycle 10
        1867.2,  # Cycle 11
        1878.9,  # Cycle 12
        1889.6,  # Cycle 13
        1901.7,  # Cycle 14
        1913.6,  # Cycle 15
        1923.6,  # Cycle 16
        1933.8,  # Cycle 17
    ]

    n_cycles = 10

    init_params = np.zeros(n_cycles * 3)

    for k in range(n_cycles):
        base_idx = k * 3

        init_params[base_idx] = hathaway_minima[k]  # T0
        init_params[base_idx + 1] = 0.3 # Ts
        init_params[base_idx + 2] = 5.0 # Td
    return init_params

@jit(nopython=True)
def solar_model(t, params): 
    ''' Solar activity model based on given parameters.
    params: array of parameters [T0_1, Ts_1, Td_1..., T0_n, Ts_n, Td_n]
    T0: start time of cycle
    Ts: amplitude scaling of cycle
    Td: rise time of cycle
    t: time array
    '''
    cycles = len(params)//3
    y_model = np.zeros_like(t)
    for k in range(cycles):
        # Extract parameters for cycle k
        T0 = params[3*k]
        Ts = params[3*k + 1]
        Td = params[3*k + 2]

        # Avoid division by zero
        if Ts == 0 or Td == 0:
            continue
        
        dt = t - T0
        #calculate Xk(t)
        xkt = (dt/Ts)**2 * np.exp(-(dt/Td)**2)
        xkt = xkt * (dt > 0)
        y_model += xkt

    return y_model

@jit(nopython=True)
def cost_function(params, t_real, y_real):
    '''Cost function to evaluate the fit of the model to the real data.
    Uses Mean Squared Error (MSE) as the metric.'''

    # Generate model data
    y_model = solar_model(t_real, params)
    # Calculate Mean Squared Error
    mse = np.mean((y_real - y_model)**2)

    return mse

@jit(nopython=True)
def get_neighbors(params, sigma):
    '''Generate a neighboring solution by perturbing one parameter.
     params: current parameter set
     sigma: standard deviation for the normal distribution used for perturbation
    '''
    # Create a copy of the current parameters
    new_params = params.copy()

    # Randomly select an index to perturb
    idx = np.random.randint(0, len(params))

    current_sigma = sigma

    if idx % 3 != 0:  
        current_sigma = sigma * 0.1
    # Perturb the selected parameter
    change = np.random.normal(0.0, current_sigma)
    
    new_params[idx] += change

    # Enforce constraints: Ts and Td must be positive
    remainder = idx % 3
    if remainder == 1 or remainder == 2: 
        if new_params[idx] < 0.01:
            new_params[idx] = 0.01

    return new_params

@jit(nopython=True)
def simulated_annealing(init_params, t_real, y_real, init_temp, cooling_rate, max_iter, sigma):

    # Initialize parameters
    current_params = init_params.copy()
    best_params = init_params.copy()

    current_cost = cost_function(current_params, t_real, y_real)
    best_cost = current_cost

    temp = init_temp
    #Saving cost history for analysis
    cost_history = np.zeros(max_iter) 

    # Main optimization loop
    for i in range(max_iter):
        #Create a neighbor
        new_params = get_neighbors(current_params, sigma)
        new_cost = cost_function(new_params, t_real, y_real)

        # Calculate cost difference
        delta_cost = new_cost - current_cost
        accept = False

        if delta_cost < 0:
            accept = True
        else:
            prob = np.exp(-delta_cost / temp)
            if np.random.random() < prob:
                accept = True
            
        # Update current solution if accepted
        if accept:
            current_params = new_params
            current_cost = new_cost
            # Update best solution found
            if current_cost < best_cost:
                best_cost = current_cost
                best_params = current_params.copy()


        # Cool down the temperature
        temp = temp * cooling_rate
        #save cost history
        cost_history[i] = best_cost

    return best_params, best_cost, cost_history

        

