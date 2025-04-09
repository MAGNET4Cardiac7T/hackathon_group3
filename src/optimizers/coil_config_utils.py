from src.data import CoilConfig

def apply_coil_config(simulation, amplitudes, phases):
    config = CoilConfig(amplitude=amplitudes, phase=phases)
    simulation.phase_shift(config)
