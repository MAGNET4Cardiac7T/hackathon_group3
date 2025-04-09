from src.costs.base import BaseCost
from src.optimizers import DummyOptimizer
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
    optimizer = DummyOptimizer(cost_function=cost_function)
    try:
        with timeout_limit(timeout):
            best_coil_config = optimizer.optimize(simulation)
    except TimeoutError:
        print("Optimization process timed out.")
        best_coil_config = None  # Or handle it as per your requirements
    return best_coil_config
