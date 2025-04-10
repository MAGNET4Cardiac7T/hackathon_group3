import torch.optim as optim
import torch
from ..data.dataclasses import CoilConfig
from ..data.simulation import Simulation
from tqdm import trange
import numpy as np
from .base import BaseOptimizer


class TorchOptimizer(BaseOptimizer):
    def __init__(
        self, cost_function, lr=0.01, max_iter=100, optimizer_class=optim.Adam
    ):
        super().__init__(cost_function)
        self.cost_function = cost_function
        self.lr = lr
        self.max_iter = max_iter
        self.optimizer_class = optimizer_class

    def optimize(self, simulation):
        # Initialize parameters (e.g., coil configuration) as torch tensors
        phase = torch.rand(8, requires_grad=True, dtype=torch.double)
        amplitude = torch.rand(8, requires_grad=True, dtype=torch.double)

        # Define the optimizer and pass the parameters
        optimizer = self.optimizer_class([phase, amplitude], lr=self.lr)

        best_cost = -np.inf if self.direction == "maximize" else np.inf
        best_coil_config = None

        pbar = trange(self.max_iter)

        for _ in pbar:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: simulate and compute cost
            coil_config = CoilConfig(phase=phase, amplitude=amplitude)
            simulation_data = simulation(coil_config)
            cost = self.cost_function(simulation_data, simulation, return_B1 = False)

            if self.direction == "maximize":
                cost_to_optimize = -cost  # Negate cost for maximization
            else:
                cost_to_optimize = cost  # Use cost for minimization

            cost_to_optimize.backward()

            # Optimizer step
            optimizer.step()

            # Track the best configuration
            if (self.direction == "minimize" and cost < best_cost) or (
                self.direction == "maximize" and cost > best_cost
            ):
                best_cost = cost
                best_coil_config = coil_config
                pbar.set_postfix_str(f"Best cost {best_cost:.2f}")

        return best_coil_config
