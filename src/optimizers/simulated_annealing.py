import torch
import numpy as np
from copy import deepcopy
from tqdm import trange
from ..data.dataclasses import CoilConfig
from ..data.simulation import Simulation
from .base import BaseOptimizer

class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    Global optimizer implementing the Simulated Annealing (SA) algorithm.

    Args:
        cost_function (BaseCost): The cost function object. Its direction attribute must be either "minimize" or "maximize".
        initial_temperature (float, optional): The starting temperature for the SA algorithm. Default is 1.0.
        cooling_rate (float, optional): The factor by which the temperature decreases after each iteration. Default is 0.99.
        max_iter (int, optional): Number of iterations for the optimization. Default is 1000.
        perturb_std (float, optional): Standard deviation for the Gaussian noise used to perturb the configuration. Default is 0.05.
    """
    def __init__(self, cost_function, initial_temperature=1.0, cooling_rate=0.99, max_iter=1000, perturb_std=0.05):
        super().__init__(cost_function)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.perturb_std = perturb_std

    def perturb(self, config: CoilConfig) -> CoilConfig:
        """
        Generates a new configuration by perturbing the current configuration
        using Gaussian noise for both phase and amplitude.

        Args:
            config (CoilConfig): The current coil configuration.

        Returns:
            CoilConfig: A new perturbed configuration.
        """
        new_phase = config.phase + torch.randn_like(config.phase) * self.perturb_std
        new_amplitude = config.amplitude + torch.randn_like(config.amplitude) * self.perturb_std
        return CoilConfig(phase=new_phase, amplitude=new_amplitude)

    def optimize(self, simulation: Simulation) -> CoilConfig:
        """
        Runs the Simulated Annealing optimization process on the coil configuration.
        
        This method uses the provided simulation (a callable that processes a CoilConfig)
        to obtain simulation data, evaluates the cost using the cost_function, and updates the configuration
        according to simulated annealing rules. The optimizer supports both minimization and maximization,
        as specified by the cost_function.direction attribute.

        Args:
            simulation (Simulation): The simulation instance used to evaluate the coil configuration.

        Returns:
            CoilConfig: The best coil configuration found during the optimization.
        """
        # Initialize with a random configuration.
        current_config = CoilConfig(
            phase=torch.rand(8, dtype=torch.double),
            amplitude=torch.rand(8, dtype=torch.double)
        )
        current_data = simulation(current_config)
        current_cost = self.cost_function(current_data)

        best_config = deepcopy(current_config)
        best_cost = current_cost

        temperature = self.initial_temperature

        pbar = trange(self.max_iter)
        for i in pbar:
            # Generate a candidate configuration by perturbing the current configuration.
            candidate_config = self.perturb(current_config)
            candidate_data = simulation(candidate_config)
            candidate_cost = self.cost_function(candidate_data)

            # Compute delta cost based on whether we are minimizing or maximizing.
            if self.direction == "minimize":
                delta_cost = candidate_cost - current_cost
            else:  # For maximization
                delta_cost = current_cost - candidate_cost

            # Accept the new configuration if it improves the cost,
            # or with a certain probability if it does not.
            if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / temperature):
                current_config = candidate_config
                current_cost = candidate_cost

                # Update the best configuration found so far.
                if (self.direction == "minimize" and current_cost < best_cost) or \
                   (self.direction == "maximize" and current_cost > best_cost):
                    best_cost = current_cost
                    best_config = deepcopy(current_config)

            pbar.set_postfix_str(f"Best cost: {best_cost:.4f}, Temp: {temperature:.4f}")
            temperature *= self.cooling_rate

        return best_config
