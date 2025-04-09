from scipy.optimize import differential_evolution
from src.optimizers.base import BaseOptimizer
from src.data import CoilConfig
import numpy as np

class GlobalScipyOptimizer(BaseOptimizer):
    def __init__(self, cost_function, max_iter=20, popsize=5):
        self.cost_function = cost_function
        self.max_iter = max_iter
        self.popsize = popsize

    def optimize(self, simulation):
        num_coils = 8
        bounds = [(0, 1)] * num_coils + [(0, 2 * np.pi)] * num_coils

        def objective(x):
            amps = x[:num_coils]
            phases = x[num_coils:]
            config = CoilConfig(amplitude=amps, phase=phases)
            sim_data = simulation(config)
            cost = self.cost_function.calculate_cost(sim_data)
            return -cost  # we want to maximize

        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=self.max_iter,
            popsize=self.popsize,
            tol=0.01,
            updating='deferred',  # better for parallel evals
            polish=True  # refine with L-BFGS-B at the end
        )

        best_amps = result.x[:num_coils].tolist()
        best_phases = result.x[num_coils:].tolist()
        best_cost = -result.fun

        return CoilConfig(amplitude=best_amps, phase=best_phases)
