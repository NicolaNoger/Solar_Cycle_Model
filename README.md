# HPC Project - Solar Cycle Model Calibration

Team 12

## Overview

Calibration of a solar cycle model (Volobuev 2009) using Simulated Annealing.
The model fits 10 solar cycles (Cycle 8-17) to real sunspot data.

## Files

- main_final_calibration.py : Main script for HPC cluster (parallel, with JIT)
- run_final_calibration.sh  : SLURM job script for earth-3 partition
- sunspot_OP.py             : Model and optimization functions (with Numba JIT)
- analyze_correlation.py    : Ts vs Td correlation analysis
- data_Team12.csv           : Input data (sunspot numbers)

## Installation

```
pip install -r requirements.txt
```

## Usage

Local:
```
python main_final_calibration.py
```

On HPC cluster:
```
sbatch run_final_calibration.sh
```

## Results

Results are saved in a folder results_YYYYMMDD_HHMMSS:
- solar_cycle_fit.png       : Model vs. real data
- scaling_study.png         : Wall time vs. cores
- cycle_parameters.csv      : Optimized parameters per cycle
- ts_td_correlation.png     : Correlation between Ts and Td

## Parameters

The model uses 3 parameters per cycle:
- T0 : Start time of the cycle
- Ts : Rise parameter
- Td : Decay parameter

Hyperparameters for Simulated Annealing:
- T_start = 500
- cooling_rate = 0.9999
- sigma = 0.02
- iterations = 100000