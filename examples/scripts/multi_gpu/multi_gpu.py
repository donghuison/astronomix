"""
Multi-GPU: run a 3D driven-turbulence simulation sharded across several GPUs.

The primitive state and helper data are domain-decomposed with a NamedSharding;
astronomix then performs the halo exchange between devices automatically. The
same script scales to more devices by changing ``NUM_GPUS`` and the mesh split.
"""

# ==== GPU selection ====
from autocvd import autocvd
NUM_GPUS = 2
autocvd(num_gpus=NUM_GPUS)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp
from jax.sharding import (
    PartitionSpec as P,
    NamedSharding,
)

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import (
    ISOTHERMAL,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)

# astronomix functions
from astronomix import (
    time_integration,
    get_helper_data,
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
    initialize_interface_fields,
)


# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# configure a 3D driven-turbulence box
num_cells = 128
sound_speed = 0.5
B_0 = 0.1

config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    equation_of_state = ISOTHERMAL,
    mhd = True,
    progress_bar = True,
    dimensionality = 3,
    num_cells = num_cells,
    box_size = 1.0,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    turbulent_forcing_config = TurbulentForcingConfig(turbulent_forcing = True),
)

registered_variables = get_registered_variables(config)

params = SimulationParams(
    C_cfl = 1.5,
    isothermal_sound_speed = sound_speed,
    t_end = 2.0 * 0.5,
    minimum_density = 0.02,
    turbulent_forcing_params = TurbulentForcingParams(energy_injection_rate = 1.65),
)

# uniform medium threaded by a uniform field along z, at rest
rho = jnp.ones((num_cells, num_cells, num_cells))
u_x = jnp.zeros_like(rho)
u_y = jnp.zeros_like(rho)
u_z = jnp.zeros_like(rho)
B_x = jnp.zeros_like(rho)
B_y = jnp.zeros_like(rho)
B_z = B_0 * jnp.ones_like(rho)
bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    magnetic_field_x = B_x,
    magnetic_field_y = B_y,
    magnetic_field_z = B_z,
    interface_magnetic_field_x = bxb,
    interface_magnetic_field_y = byb,
    interface_magnetic_field_z = bzb,
)

config = finalize_config(config, initial_state.shape)

# domain-decompose the state and helper data along the x axis across the GPUs
mesh = jax.make_mesh((1, NUM_GPUS, 1, 1), (VARAXIS, XAXIS, YAXIS, ZAXIS))
sharding = NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))
initial_state = jax.device_put(initial_state, sharding)
helper_data = get_helper_data(config, sharding)

jax.debug.visualize_array_sharding(initial_state[registered_variables.density_index, :, :, 0])

# run the simulation across the GPUs
final_state = time_integration(
    initial_state, config, params, registered_variables, sharding=sharding
)

# plot a central density slice
z = num_cells // 2
density = final_state[registered_variables.density_index][:, :, z]

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(density.T, origin="lower", cmap="viridis")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(figures_dir / "multi_gpu_density.png", dpi=200, bbox_inches="tight")
