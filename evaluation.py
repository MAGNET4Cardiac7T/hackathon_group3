from main import run

from src.costs import B1HomogeneityCost
from src.data import Simulation
from src.utils import evaluate_coil_config
import plot_final_results as group3_plt

import numpy as np
import json

if __name__ == "__main__":
    # Load simulation data
    simulation = Simulation("data/simulations/children_3_tubes_10_id_6299.h5")
    
    # Define cost function
    cost_function = B1HomogeneityCost()
    
    # Run optimization
    best_coil_config = run(simulation=simulation, cost_function=cost_function)
    
    # Evaluate best coil configuration
    #result = evaluate_coil_config(best_coil_config, simulation, cost_function)
    result, B1plus_map_default, B1plus_map_bestConfig = evaluate_coil_config(best_coil_config, simulation, cost_function, return_B1 = True)


    # Save results to JSON file
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)

    group3_plt.plot_final_results(B1plus_map_default, B1plus_map_bestConfig, simulation)

    print("end")
