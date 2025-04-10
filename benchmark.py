import time
import numpy as np
import torch
import os
import optuna

# Import simulation, data structures, optimizers, and cost function.
from src.data.simulation import Simulation
from src.data.dataclasses import CoilConfig
from src.optimizers.simulated_annealing import SimulatedAnnealingOptimizer
from src.optimizers.torch_optimizer import TorchOptimizer  
from src.costs.b1_homogeneity import B1HomogeneityCost  

# Set up paths: benchmark.py is assumed to be at the project root.
project_root = os.path.abspath(os.path.dirname(__file__))
simulations_dir = os.path.join(project_root, 'data', 'simulations')
coil_path = os.path.join(project_root, 'data', 'antenna', 'antenna.h5')

# Optionally, print the paths for debugging.
print("Project root:", project_root)
print("Simulations directory:", simulations_dir)
print("Coil path:", coil_path)

def get_simulation_files(directory):
    """
    Scans the given directory and returns a list of dictionaries with simulation file names.
    Only files ending with '.h5' are included.
    """
    files = os.listdir(directory)
    simulation_files = []
    for file in files:
        if file.endswith(".h5"):
            simulation_files.append({
                "name": os.path.splitext(file)[0],  # file name without extension
                "filename": file
            })
    # Optionally sort the list for consistency.
    simulation_files.sort(key=lambda x: x["name"])
    return simulation_files

sim_files = get_simulation_files(simulations_dir)
if not sim_files:
    raise ValueError("No simulation files found in directory: " + simulations_dir)
# Build a list of full paths.
simulation_paths = [os.path.join(simulations_dir, sim_info['filename']) for sim_info in sim_files]
print("Using simulation files:")
for path in simulation_paths:
    print("  ", path)

# Create the simulation instance and cost function.
sim = Simulation(path=simulation_paths, coil_path=coil_path)
cost_function = B1HomogeneityCost()  # Assumes cost_function.direction is set to "maximize"

######################################
# Objective functions for hyperparameter tuning
######################################

def objective_sa(trial):
    """
    Objective function for tuning SimulatedAnnealingOptimizer.
    Optuna will suggest values for the hyperparameters. We run one optimizer instance and return 
    the negative best cost (so that higher cost yields a lower objective value).
    """
    # Sample hyperparameters for Simulated Annealing.
    initial_temperature = trial.suggest_float("initial_temperature", 0.1, 5.0, log=True)
    cooling_rate = trial.suggest_float("cooling_rate", 0.90, 0.999)
    max_iter = trial.suggest_int("max_iter", 500, 1000)
    perturb_std = trial.suggest_float("perturb_std", 0.01, 0.1)

    # For reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)

    optimizer = SimulatedAnnealingOptimizer(
        cost_function,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        max_iter=max_iter,
        perturb_std=perturb_std
    )
    
    start_time = time.perf_counter()
    best_config = optimizer.optimize(sim)
    duration = time.perf_counter() - start_time

    best_data = sim(best_config)
    best_cost = cost_function(best_data)
    print(f"(SA) Trial completed in {duration:.2f} s with best cost {best_cost:.4f}")

    # Return negative best_cost because we want to maximize best_cost but Optuna minimizes.
    return -best_cost

def objective_torch(trial):
    """
    Objective function for tuning TorchOptimizer (using Adam).
    Optuna will suggest values for the hyperparameters. We run one optimizer instance and return 
    the negative best cost.
    """
    # Sample hyperparameters for TorchOptimizer.
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    max_iter = trial.suggest_int("max_iter", 500, 1500)

    torch.manual_seed(42)
    np.random.seed(42)

    optimizer = TorchOptimizer(
        cost_function,
        lr=lr,
        max_iter=max_iter,
        optimizer_class=torch.optim.Adam  # Explicitly specifying Adam.
    )
    
    start_time = time.perf_counter()
    best_config = optimizer.optimize(sim)
    duration = time.perf_counter() - start_time

    best_data = sim(best_config)
    best_cost = cost_function(best_data)
    print(f"(Torch) Trial completed in {duration:.2f} s with best cost {best_cost:.4f}")

    return -best_cost

######################################
# Main hyperparameter tuning execution
######################################

def tune_optimizer(name, objective_fn, n_trials=20):
    print(f"\nTuning hyperparameters for {name} with {n_trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_fn, n_trials=n_trials)
    print(f"\nBest hyperparameters for {name}: {study.best_params}")
    print(f"Best objective value (negative best cost): {study.best_value}")
    return study.best_params

def main():
    n_trials = 5  # Number of trials for tuning

    # Tuning for Simulated Annealing Optimizer.
    best_params_sa = tune_optimizer("Simulated Annealing Optimizer", objective_sa, n_trials=n_trials)
    
    # Tuning for Torch Optimizer.
    best_params_torch = tune_optimizer("Torch Optimizer (Adam)", objective_torch, n_trials=n_trials)
    
    # Optionally, you can now run a final benchmark comparison with the tuned hyperparameters.
    print("\n--- Final Benchmark Comparison with Tuned Hyperparameters ---")
    sa_final_cost = objective_sa(optuna.trial.FixedTrial(best_params_sa))
    torch_final_cost = objective_torch(optuna.trial.FixedTrial(best_params_torch))
    
    print(f"Final tuned Simulated Annealing best cost (negative value): {sa_final_cost:.4f}")
    print(f"Final tuned Torch Optimizer best cost (negative value): {torch_final_cost:.4f}")

if __name__ == "__main__":
    main()
