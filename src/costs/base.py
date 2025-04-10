from ..data.simulation import SimulationData
from abc import ABC, abstractmethod

class BaseCost(ABC):
    def __init__(self) -> None:
        self.direction = "minimize"
        assert self.direction in ["minimize", "maximize"], f"Invalid direction: {self.direction}"
        
    def __call__(self, simulation_data: SimulationData, simulation, return_B1 = False) -> float:
        return self.calculate_cost(simulation_data, simulation, return_B1)

    @abstractmethod
    def calculate_cost(self, simulation_data: SimulationData, simulation = 0) -> float:
        raise NotImplementedError