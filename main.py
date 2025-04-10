from src.costs.base import BaseCost
from src.costs import B1HomogeneityCost, B1HomogeneityMinMaxCost
from src.optimizers import DummyOptimizer, TorchOptimizer
from src.data import Simulation, CoilConfig

import numpy as np

import signal
from contextlib import contextmanager


@contextmanager
def timeout_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Function execution exceeded the timeout limit.")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def run(
    simulation: Simulation, cost_function: BaseCost, timeout: int = 100
) -> CoilConfig:
    """
    Main function to run the optimization, returns the best coil configuration

    Args:
        simulation: Simulation object
        cost_function: Cost function object
        timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    optimizer = TorchOptimizer(cost_function=cost_function)
    best_coil_config = optimizer.optimize(simulation)
    """
    try:
        with timeout_limit(timeout):
            best_coil_config = optimizer.optimize(simulation)
    except TimeoutError:
        print("Optimization process timed out.")
        best_coil_config = None  # Or handle it as per your requirements
    """
    return best_coil_config


if __name__ == "__main__":
    # Example usage
    simulation = Simulation(path="./data/simulations/children_3_tubes_10_id_6299.h5")
    cost_function = (
        B1HomogeneityCost()
    )  # Replace with actual cost function initialization
    best_coil_config = run(simulation, cost_function, timeout=100)
    print(best_coil_config)
