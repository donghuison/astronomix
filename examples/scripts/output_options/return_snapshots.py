"""
Output options: return_snapshots for in-memory diagnostics.

With ``return_snapshots`` the time integrator returns a SnapshotData object
instead of just the final state. Everything stays on the GPU, which is ideal for
building losses over several points in time. The SnapshotSettings give
fine-grained control over what is returned — here we ask only for integrated
energies (cheap), not the full per-snapshot states (the biggest allocation).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    HLLC,
    MINMOD,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix import (
    SnapshotSettings,
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
)

# astronomix functions
from astronomix import (
    time_integration,
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
)


# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# configure the 2D Kelvin-Helmholtz test case with snapshot diagnostics
box_size = 1.0
num_cells = 256

config = SimulationConfig(
    riemann_solver = HLLC,
    limiter = MINMOD,
    progress_bar = True,
    dimensionality = 2,
    box_size = box_size,
    num_cells = num_cells,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    return_snapshots = True,
    num_snapshots = 100,
    snapshot_settings = SnapshotSettings(
        # cheap integrated diagnostics — no full per-snapshot states
        return_total_energy = True,
        return_kinetic_energy = True,
        return_internal_energy = True,
    ),
)

registered_variables = get_registered_variables(config)

params = SimulationParams(t_end = 2.0, C_cfl = 0.4)

# two counter-streaming shear layers with a small sinusoidal perturbation
grid_spacing = box_size / num_cells
x = jnp.linspace(grid_spacing / 2, box_size - grid_spacing / 2, num_cells)
X, Y = jnp.meshgrid(x, x, indexing="ij")

rho = jnp.where((Y > 0.25) & (Y < 0.75), 2.0, 1.0)
u_x = jnp.where((Y > 0.25) & (Y < 0.75), -0.5, 0.5)
u_y = 0.01 * jnp.sin(2 * jnp.pi * X)
p = 2.5 * jnp.ones_like(X)

initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    gas_pressure = p,
)

config = finalize_config(config, initial_state.shape)

# run the simulation — returns a SnapshotData object (kept on the GPU)
snapshots = time_integration(initial_state, config, params, registered_variables)

# plot the energy budget over time
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(snapshots.time_points, snapshots.total_energy, label="total")
ax.plot(snapshots.time_points, snapshots.kinetic_energy, label="kinetic")
ax.plot(snapshots.time_points, snapshots.internal_energy, label="internal")
ax.set_xlabel("time")
ax.set_ylabel("energy")
ax.legend()
fig.savefig(figures_dir / "khi_energy_budget.png", dpi=200, bbox_inches="tight")
