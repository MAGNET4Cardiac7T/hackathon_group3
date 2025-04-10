from .data import CoilConfig, Simulation
from .costs.base import BaseCost
from typing import Dict, Any
import torch

def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.item()  # scalar
        else:
            return value.tolist()
    elif isinstance(value, list):
        return [to_jsonable(v) for v in value]
    elif isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    else:
        return value

def evaluate_coil_config(coil_config: CoilConfig, 
                         simulation: Simulation,
                         cost_function: BaseCost,
                         return_B1 = True,
                         return_SAR = True) -> Dict[str, Any]:
    """
    Evaluates the coil configuration using the cost function.

    Args:
        coil_config: Coil configuration to evaluate.
        simulation: Simulation object.
        cost_function: Cost function object.

    Returns:
        A dictionary containing the best coil configuration, cost, and cost improvement.
    """
    default_coil_config = CoilConfig()

    simulation_data = simulation(coil_config)
    simulation_data_default = simulation(default_coil_config)

    # Calculate cost for both configurations
    default_coil_config_cost, B1map_default, SAR_default = cost_function(simulation_data_default, simulation, return_B1 = return_B1, return_SAR = return_SAR)
    best_coil_config_cost, B1map_bestConfig, SAR_bestConfig = cost_function(simulation_data, simulation, return_B1 = return_B1, return_SAR = return_SAR)


    # Cost improvements
    cost_improvement_absolute = default_coil_config_cost - best_coil_config_cost
    cost_improvement_relative = (
        (best_coil_config_cost - default_coil_config_cost) / default_coil_config_cost
    )

    result = {
        "best_coil_phase": to_jsonable(coil_config.phase),
        "best_coil_amplitude": to_jsonable(coil_config.amplitude),
        "best_coil_config_cost": to_jsonable(best_coil_config_cost),
        "default_coil_config_cost": to_jsonable(default_coil_config_cost),
        "cost_improvement_absolute": to_jsonable(cost_improvement_absolute),
        "cost_improvement_relative": to_jsonable(cost_improvement_relative),
        "cost_function_name": cost_function.__class__.__name__,
        "cost_function_direction": cost_function.direction,
        "simulation_data": simulation_data.simulation_name,
    }

    return result, B1map_default, B1map_bestConfig, SAR_default, SAR_bestConfig
