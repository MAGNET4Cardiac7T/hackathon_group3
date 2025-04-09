from dataclasses import dataclass, field
import torch


@dataclass
class CoilConfig:
    """
    Stores the coil configuration data i.e. the phase and amplitude of each coil.
    """

    phase: torch.Tensor = field(
        default_factory=lambda: torch.zeros((8,), dtype=torch.double)
    )
    amplitude: torch.Tensor = field(
        default_factory=lambda: torch.ones((8,), dtype=torch.double)
    )

    def __post_init__(self):

        assert (
            self.phase.shape == self.amplitude.shape
        ), "Phase and amplitude must have the same shape."
        assert self.phase.shape == (8,), "Phase and amplitude must have shape (8,)."


@dataclass
class SimulationData:
    """
    Stores the simulation data for a specific coil configuration.
    """

    simulation_name: str
    properties: torch.Tensor
    field: torch.Tensor
    subject: torch.Tensor
    coil_config: CoilConfig


@dataclass
class SimulationRawData:
    """
    Stores the raw simulation data. Each coil contribution is stored separately along an additional dimension.
    """

    simulation_name: str
    properties: torch.Tensor
    field: torch.Tensor
    subject: torch.Tensor
    coil: torch.Tensor
