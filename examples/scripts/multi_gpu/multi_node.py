"""
Multi-node: run a 3D driven-turbulence simulation sharded across GPUs on
several nodes.

This is the multi-node counterpart of ``multi_gpu.py``. The only differences are
that (1) ``jax.distributed.initialize()`` is called first so the processes form a
single global JAX runtime, and (2) the device mesh spans *all* global devices
(``jax.devices()``) rather than one node's GPUs. astronomix handles the
inter-device halo exchange — across nodes — automatically.

Device assignment is done by the cluster scheduler (e.g. SLURM), so ``autocvd``
is not used here. Launch one process per node, e.g. with SLURM:

    srun --nodes=2 --gpus-per-node=8 python multi_node.py

``jax.distributed.initialize()`` auto-detects the coordinator address, process
count and process id from the SLURM environment; pass them explicitly on other
launchers.
"""

# ==== distributed initialization (must precede any other JAX use) ====
import jax
jax.distributed.initialize()
# =====================================================================

# jax
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

if jax.process_index() == 0:
    print(f"Running on {jax.process_count()} processes, "
          f"{jax.device_count()} devices in total.")

# configure a 3D driven-turbulence box
num_cells = 256
sound_speed = 0.5
B_0 = 0.1

config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    equation_of_state = ISOTHERMAL,
    mhd = True,
    progress_bar = jax.process_index() == 0,
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

# uniform medium threaded by a uniform field along z, at rest (identical on
# every process, so the replicated arrays can be resharded onto the global mesh)
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

# domain-decompose across ALL global devices along the x axis
mesh = jax.make_mesh((1, jax.device_count(), 1, 1), (VARAXIS, XAXIS, YAXIS, ZAXIS))
sharding = NamedSharding(mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))
initial_state = jax.device_put(initial_state, sharding)
helper_data = get_helper_data(config, sharding)

# run the simulation across all nodes
final_state = time_integration(
    initial_state, config, params, registered_variables, sharding=sharding
)

# gather a central density slice to process 0 and plot it there
z = num_cells // 2
density_slice = jax.device_get(final_state[registered_variables.density_index][:, :, z])

if jax.process_index() == 0:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(density_slice.T, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(figures_dir / "multi_node_density.png", dpi=200, bbox_inches="tight")
