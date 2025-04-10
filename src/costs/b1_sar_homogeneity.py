from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator

import torch


def SAR(simulation_data):

    E_field = simulation_data.field[0]
    E_field_complex = E_field[0] + 1j * E_field[1]
    E_field_abs = torch.abs(E_field_complex)
    E_field_abs2 = E_field_abs[0] ** 2 + E_field_abs[1] ** 2 + E_field_abs[2] ** 2

    # conductivity, permitivity, density
    sigma = simulation_data.properties[0]
    epsilon = simulation_data.properties[1]
    rho = simulation_data.properties[2]

    # get mask
    mask = simulation_data.subject

    SAR_3D = E_field_abs2 * sigma / rho * mask
    SAR_max = torch.max(SAR_3D)

    return SAR_3D, SAR_max


class B1SARHomogeneityCost(BaseCost):
    def __init__(self) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator()

    def calculate_cost(
        self, simulation_data: SimulationData, return_B1=False, return_SAR=False
    ) -> float:

        lambda_SAR = 1

        b1_field = self.b1_calculator(simulation_data)
        subject = simulation_data.subject

        SAR_3D, SAR_max = SAR(simulation_data)
        SAR_max = torch.nn.Parameter(SAR_max)

        b1_field_abs = torch.abs(b1_field)
        b1_field_subject_voxels = b1_field_abs[subject]
        b1_field_min = torch.min(b1_field_subject_voxels)
        if return_B1 == False:
            return torch.mean(b1_field_subject_voxels) / torch.std(
                b1_field_subject_voxels
            ) - lambda_SAR * b1_field_min / torch.sqrt(SAR_max)
        else:
            if return_SAR == False:
                return (
                    torch.mean(b1_field_subject_voxels)
                    / torch.std(b1_field_subject_voxels)
                ) - lambda_SAR * SAR_max, b1_field_abs
            else:
                return (
                    (
                        torch.mean(b1_field_subject_voxels)
                        / torch.std(b1_field_subject_voxels)
                    )
                    - lambda_SAR * SAR_max,
                    b1_field_abs,
                    SAR_3D,
                )
