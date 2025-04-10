import torch
import numpy as np
from typing import Tuple
from scipy.stats import qmc


def cp_init():
    # cp init
    phase = torch.ones((8), requires_grad=True, dtype=torch.double)
    amplitude = torch.ones((8), requires_grad=True, dtype=torch.double)

    with torch.no_grad():
        amplitude[:] = 1.0 / np.sqrt(8.0)
        for i in range(8):
            phase[i] = torch.pi / 4.0 * i

    return phase, amplitude


def init_cp_and_random(num_starts: int = 10, param_dim: int = 16):
    return [cp_init()] + latin_hypercube_init(
        num_starts=num_starts, param_dim=param_dim
    )


def latin_hypercube_init(
    num_starts: int = 10, param_dim: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate `num_starts` initial configurations distributed optimally in parameter space.

    Parameters:
        num_starts (int): Number of initial configurations to generate.
        param_dim (int): Dimensionality of the parameter space (e.g., number of coils).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Initialized phase and amplitude tensors.
    """
    # Use Latin Hypercube Sampling to generate well-distributed samples
    sampler = qmc.LatinHypercube(d=param_dim)
    samples = sampler.random(n=num_starts)

    phases = [
        torch.tensor(
            samples[i, : (param_dim // 2)] * 2 * np.pi,
            dtype=torch.double,
            requires_grad=True,
        )
        for i in range(num_starts)
    ]  # [0, 2π]

    amplitudes = [
        torch.tensor(
            samples[i, (param_dim // 2) :] * 1.0,
            dtype=torch.double,
            requires_grad=True,
        )  # [0, 1]
        for i in range(num_starts)
    ]

    return list(zip(phases, amplitudes))
