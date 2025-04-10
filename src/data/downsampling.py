import torch
import h5py
import os
import einops

from typing import Tuple
from .dataclasses import SimulationRawData, SimulationData, CoilConfig
from .simulation import Simulation


class DownsampledSimulation(Simulation):
    def __init__(
        self, path: str, coil_path: str = "data/antenna/antenna.h5", resolution: int = 1
    ):
        super().__init__(path, coil_path)
        self.resolution = resolution

        self.simulation_raw_data = self._load_raw_simulation_data(resolution=resolution)

    @classmethod
    def from_simulation(
        cls, simulation: Simulation, resolution: int
    ) -> "DownsampledSimulation":
        """
        Create a DownsampledSimulation instance from an existing Simulation instance.

        Parameters:
            simulation (Simulation): The original simulation instance.
            resolution (int): The downsampling factor.

        Returns:
            DownsampledSimulation: A new instance with downsampled data.
        """
        # Create a new instance of DownsampledSimulation
        instance = cls(simulation.path, simulation.coil_path, resolution)
        # Downsample each component of simulation_raw_data
        instance.simulation_raw_data = SimulationRawData(
            simulation_name=simulation.simulation_raw_data.simulation_name,
            properties=DownsampledSimulation.downsample(
                simulation.simulation_raw_data.properties, resolution, dims=(1, 2, 3)
            ),
            field=DownsampledSimulation.downsample(
                simulation.simulation_raw_data.field, resolution, dims=(3, 4, 5)
            ),
            subject=DownsampledSimulation.downsample(
                simulation.simulation_raw_data.subject, resolution, dims=(0, 1, 2)
            ),
            coil=DownsampledSimulation.downsample(
                simulation.simulation_raw_data.coil, resolution, dims=(0, 1, 2)
            ),
        )

        return instance

    @staticmethod
    def downsample(
        data: torch.Tensor, resolution: int = 1, dims: Tuple[int, ...] = None
    ) -> torch.Tensor:
        """
        Downsample a PyTorch tensor along selected dimensions using slicing.

        Parameters:
        data (torch.Tensor): The input tensor to downsample.
        resolution (int): The downsampling factor.
        dims (Tuple[int, ...]): The dimensions to downsample. If None, all dimensions are downsampled.

        Returns:
        torch.Tensor: The downsampled tensor.
        """
        if dims is None:
            dims = tuple(range(data.ndim))  # Default to all dimensions if dims is None
        slices = [slice(None)] * data.ndim  # Create a slice for each dimension
        for dim in dims:
            slices[dim] = slice(
                None, None, resolution
            )  # Apply step size to selected dimensions
        return data[tuple(slices)]

    def _load_raw_simulation_data(self, resolution: int = 1) -> SimulationRawData:
        # Load raw simulation data from path with optional resolution parameter

        def downsample(
            data: torch.Tensor, resolution: int = 1, dims: Tuple[int, ...] = None
        ) -> torch.Tensor:
            """
            Downsample a PyTorch tensor along selected dimensions using slicing.

            Parameters:
            data (torch.Tensor): The input tensor to downsample.
            resolution (int): The downsampling factor.
            dims (Tuple[int, ...]): The dimensions to downsample. If None, all dimensions are downsampled.

            Returns:
            torch.Tensor: The downsampled tensor.
            """
            if dims is None:
                dims = tuple(
                    range(data.ndim)
                )  # Default to all dimensions if dims is None
            slices = [slice(None)] * data.ndim  # Create a slice for each dimension
            for dim in dims:
                slices[dim] = slice(
                    None, None, resolution
                )  # Apply step size to selected dimensions
            return data[tuple(slices)]

        def read_field() -> Tuple[torch.Tensor, torch.Tensor]:
            with h5py.File(self.path) as f:
                re_efield, im_efield = downsample(
                    torch.tensor(f["efield"]["re"][:], dtype=torch.double),
                    resolution,
                    dims=(1, 2, 3),
                ), downsample(
                    torch.tensor(f["efield"]["im"][:], dtype=torch.double),
                    resolution,
                    dims=(1, 2, 3),
                )
                re_hfield, im_hfield = downsample(
                    torch.tensor(f["hfield"]["re"][:], dtype=torch.double),
                    resolution,
                    dims=(1, 2, 3),
                ), downsample(
                    torch.tensor(f["hfield"]["im"][:], dtype=torch.double),
                    resolution,
                    dims=(1, 2, 3),
                )
                field = torch.stack(
                    [
                        torch.stack([re_efield, im_efield], dim=0),
                        torch.stack([re_hfield, im_hfield], dim=0),
                    ],
                    dim=0,
                )
            return field

        def read_physical_properties() -> torch.Tensor:
            with h5py.File(self.path) as f:
                physical_properties = downsample(
                    torch.tensor(f["input"][:], dtype=torch.double),
                    resolution,
                    dims=(1, 2, 3),
                )
            return physical_properties

        def read_subject_mask() -> torch.Tensor:
            with h5py.File(self.path) as f:
                subject = torch.tensor(f["subject"][:], dtype=torch.double)
                subject = torch.max(subject, dim=-1).values
                subject = downsample(subject, resolution, dims=(0, 1, 2))
            return subject

        def read_coil_mask() -> torch.Tensor:
            with h5py.File(self.coil_path) as f:
                coil = downsample(
                    torch.tensor(f["masks"][:], dtype=torch.double), resolution
                )
            return coil

        def read_simulation_name() -> str:
            return os.path.basename(self.path)[:-3]

        simulation_raw_data = SimulationRawData(
            simulation_name=read_simulation_name(),
            properties=read_physical_properties(),
            field=read_field(),
            subject=read_subject_mask(),
            coil=read_coil_mask(),
        )

        return simulation_raw_data

    def _shift_field(
        self, field: torch.Tensor, phase: torch.Tensor, amplitude: torch.Tensor
    ) -> torch.Tensor:
        """
        Shift the field calculating field_shifted = field * amplitude (e ^ (phase * 1j)) and summing over all coils.
        """
        re_phase = torch.cos(phase) * amplitude
        im_phase = torch.sin(phase) * amplitude
        coeffs_real = torch.stack((re_phase, -im_phase), dim=0)
        coeffs_im = torch.stack((im_phase, re_phase), dim=0)
        coeffs = torch.stack((coeffs_real, coeffs_im), dim=0)
        coeffs = einops.repeat(
            coeffs, "reimout reim coils -> hf reimout reim coils", hf=2
        )
        field_shift = einops.einsum(
            field,
            coeffs,
            "hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...",
        )
        return field_shift

    def phase_shift(self, coil_config: CoilConfig) -> SimulationData:

        field_shifted = self._shift_field(
            self.simulation_raw_data.field, coil_config.phase, coil_config.amplitude
        )

        simulation_data = SimulationData(
            simulation_name=self.simulation_raw_data.simulation_name,
            properties=self.simulation_raw_data.properties,
            field=field_shifted,
            subject=self.simulation_raw_data.subject,
            coil_config=coil_config,
        )
        return simulation_data

    def __call__(self, coil_config: CoilConfig) -> SimulationData:
        return self.phase_shift(coil_config)
