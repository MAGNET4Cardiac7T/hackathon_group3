from ..data.simulation import SimulationData
from abc import ABC, abstractmethod


class BaseCost(ABC):
    def __init__(self) -> None:
        self.direction = "minimize"
        assert self.direction in [
            "minimize",
            "maximize",
        ], f"Invalid direction: {self.direction}"

    def __call__(
        self, simulation_data: SimulationData, return_B1=False, return_SAR=False
    ) -> float:
        return self.calculate_cost(simulation_data, return_B1, return_SAR)

    @abstractmethod
    def calculate_cost(self, simulation_data: SimulationData) -> float:
        raise NotImplementedError
