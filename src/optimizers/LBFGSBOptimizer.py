from src.optimizers.base import BaseOptimizer
from src.optimizers.coil_config_utils import apply_coil_config
from src.data import CoilConfig
from scipy.optimize import minimize
import numpy as np

class ScipyOptimizer(BaseOptimizer):
    def __init__(self, cost_function, max_iter=100, timeout=None):
        self.cost_function = cost_function
        self.max_iter = max_iter
        self.timeout = timeout

    def optimize(self, simulation):
        num_coils = 8
        x0 = [0.5] * num_coils + [0.0] * num_coils
        bounds = [(0, 1)] * num_coils + [(0, 2 * np.pi)] * num_coils

        def objective(x):
            amps = x[:num_coils]
            phases = x[num_coils:]
            config = CoilConfig(amplitude=amps, phase=phases)
            sim_data = simulation(config)  # ✅ simulation returns SimulationData
            cost = self.cost_function.calculate_cost(sim_data)
            return -cost  # we want to maximize

        result = minimize(
            objective,
            x0,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter}
        )

        best_amps = result.x[:num_coils].tolist()
        best_phases = result.x[num_coils:].tolist()
        best_cost = -result.fun

        return CoilConfig(amplitude=best_amps, phase=best_phases)

