from src.costs.base import BaseCost
from src.costs import B1HomogeneityCost, B1HomogeneityMinMaxCost
from src.optimizers import DummyOptimizer, TorchOptimizer, MultiStartTorchOptimizer
from src.data import Simulation, CoilConfig


def run(simulation: Simulation, cost_function: BaseCost) -> CoilConfig:
    """
    Main function to run the optimization, returns the best coil configuration

    Args:
        simulation: Simulation object
        cost_function: Cost function object
        timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    optimizer = MultiStartTorchOptimizer(
        cost_function=cost_function,
        downsampling_factor=2,
        max_iter_explore=20,
        num_starts=10,
        max_iter_downsampled=865,
        max_iter=100,
    )
    best_coil_config = optimizer.optimize(simulation)

    return best_coil_config
