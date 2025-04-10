import torch.optim as optim
import torch
from ..data.dataclasses import CoilConfig
from ..data.simulation import Simulation
from ..data.downsampling import DownsampledSimulation
from tqdm import trange
import numpy as np
from .base import BaseOptimizer
from .init_values import cp_init, init_cp_and_random
from .tensor_b1_homogeneity import TensorB1HomogeneityCost


class MultiStartTorchOptimizer(BaseOptimizer):
    def __init__(
        self,
        cost_function=TensorB1HomogeneityCost(),
        lr=0.01,
        max_iter_explore=100,
        max_iter=100,
        max_iter_downsampled=100,
        optimizer_class=optim.Adam,
        downsampling_factor: int = 1,
        num_starts: int = 10,
    ):
        super().__init__(cost_function)
        self.cost_function = cost_function
        self.lr = lr
        self.max_iter = max_iter
        self.max_iter_explore = max_iter_explore
        self.max_iter_downsampled = max_iter_downsampled
        self.optimizer_class = optimizer_class
        self.downsampling_factor = downsampling_factor
        self.num_starts = num_starts

    def optimize(self, simulation):
        best_cost = -np.inf if self.direction == "maximize" else np.inf
        best_coil_config = None

        init_values = init_cp_and_random(num_starts=self.num_starts)

        for start, (phase, amplitude) in enumerate(init_values):
            print(f"Starting optimization run {start}/{self.num_starts}")

            if start == 0:
                phase, amplitude = cp_init()
            else:
                # Initialize parameters (e.g., coil configuration) as torch tensors
                phase = torch.tensor(
                    np.random.uniform(0, 2 * np.pi, size=8),
                    requires_grad=True,
                    dtype=torch.double,
                )
                amplitude = torch.rand(8, requires_grad=True, dtype=torch.double)

            # Downsampled simulation
            downsampled_simulation = DownsampledSimulation.from_simulation(
                simulation, resolution=self.downsampling_factor
            )

            # Define the optimizer and pass the parameters
            optimizer = self.optimizer_class([phase, amplitude], lr=self.lr)

            # Optimize on the downsampled simulation
            pbar = trange(self.max_iter_explore, desc=f"Run {start}")
            for _ in pbar:
                optimizer.zero_grad()
                coil_config = CoilConfig(phase=phase, amplitude=amplitude)
                simulation_data = downsampled_simulation(coil_config)
                cost = self.cost_function(simulation_data)

                if self.direction == "maximize":
                    cost_to_optimize = -cost
                else:
                    cost_to_optimize = cost

                cost_to_optimize.backward()
                optimizer.step()

                # Track the best configuration
                if (self.direction == "minimize" and cost < best_cost) or (
                    self.direction == "maximize" and cost > best_cost
                ):
                    best_cost = cost
                    best_coil_config = CoilConfig(
                        phase=phase.detach().clone(),
                        amplitude=amplitude.detach().clone(),
                    )
                    pbar.set_postfix_str(f"Best cost {best_cost:.2f}")

        # Fine-tune the best configuration on the original simulation
        print("Fine-tuning the best configuration on the original simulation...")
        phase = best_coil_config.phase.clone().detach().requires_grad_(True)
        amplitude = best_coil_config.amplitude.clone().detach().requires_grad_(True)
        optimizer = self.optimizer_class([phase, amplitude], lr=self.lr)

        pbar = trange(self.max_iter_downsampled, desc="Fine-tuning Downsampled")
        for _ in pbar:
            optimizer.zero_grad()
            coil_config = CoilConfig(phase=phase, amplitude=amplitude)
            simulation_data = downsampled_simulation(coil_config)
            cost = self.cost_function(simulation_data)
            if self.direction == "maximize":
                cost_to_optimize = -cost
            else:
                cost_to_optimize = cost

            cost_to_optimize.backward()
            optimizer.step()

            # Track the best configuration
            if (self.direction == "minimize" and cost < best_cost) or (
                self.direction == "maximize" and cost > best_cost
            ):
                best_cost = cost
                best_coil_config = CoilConfig(
                    phase=phase.detach().clone(),
                    amplitude=amplitude.detach().clone(),
                )
                pbar.set_postfix_str(f"Best cost {best_cost:.2f}")

        phase = best_coil_config.phase.clone().detach().requires_grad_(True)
        amplitude = best_coil_config.amplitude.clone().detach().requires_grad_(True)
        optimizer = self.optimizer_class([phase, amplitude], lr=self.lr)
        pbar = trange(self.max_iter, desc="Fine-tuning Original")
        for _ in pbar:
            optimizer.zero_grad()
            coil_config = CoilConfig(phase=phase, amplitude=amplitude)
            simulation_data = simulation(coil_config)
            cost = self.cost_function(simulation_data)

            if self.direction == "maximize":
                cost_to_optimize = -cost
            else:
                cost_to_optimize = cost

            cost_to_optimize.backward()
            optimizer.step()

            # Track the best configuration
            if (self.direction == "minimize" and cost < best_cost) or (
                self.direction == "maximize" and cost > best_cost
            ):
                best_cost = cost
                best_coil_config = CoilConfig(
                    phase=phase.detach().clone(),
                    amplitude=amplitude.detach().clone(),
                )
                pbar.set_postfix_str(f"Best cost {best_cost:.2f}")

        print(f"Final cost on downsampled: {best_cost}")
        cost_original = self.cost_function(simulation(best_coil_config))
        print(f"Final cost on original: {cost_original}")

        return best_coil_config
