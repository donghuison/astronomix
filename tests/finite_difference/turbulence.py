"""
Minimal driven turbulence test setup in astronomix.
"""

# ==== GPU selection ====
from autocvd import autocvd
from matplotlib.colors import LogNorm
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# astronomix
from astronomix._fluid_equations._equations import get_absolute_velocity
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig, TurbulentForcingParams
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SnapshotSettings,
    finalize_config
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

# Configure the simulation
config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    progress_bar = True,
    dimensionality = 3,
    num_cells = 64,
    box_size = 1.0,
    mhd = True,
    boundary_settings=BoundarySettings(
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
    ),
    turbulent_forcing_config = TurbulentForcingConfig(
        turbulent_forcing = True,
    ),
    return_snapshots = True,
    num_snapshots = 50,
    snapshot_settings = SnapshotSettings(
        return_states = True,
    )
)

# Set the simulation parameters
params = SimulationParams(
    C_cfl = 1.5,
    t_end = 1.0,
    turbulent_forcing_params = TurbulentForcingParams(
        energy_injection_rate = 1.65
    ),
)

# Initialize the registered variables
registered_variables = get_registered_variables(config)

# Initialize the simulation state
density = jnp.ones((config.num_cells, config.num_cells, config.num_cells), dtype=jnp.float32)
velocity_x = jnp.zeros_like(density)
velocity_y = jnp.zeros_like(density)
velocity_z = jnp.zeros_like(density)
pressure = jnp.ones_like(density)
magnetic_field_x = jnp.zeros_like(density)
magnetic_field_y = jnp.zeros_like(density)
magnetic_field_z = jnp.ones_like(density) * 10.0
bxb, byb, bzb = initialize_interface_fields(magnetic_field_x, magnetic_field_y, magnetic_field_z)
initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=density,
    velocity_x=velocity_x,
    velocity_y=velocity_y,
    velocity_z=velocity_z,
    gas_pressure=pressure,
    magnetic_field_x=magnetic_field_x,
    magnetic_field_y=magnetic_field_y,
    magnetic_field_z=magnetic_field_z,
    interface_magnetic_field_x=bxb,
    interface_magnetic_field_y=byb,
    interface_magnetic_field_z=bzb,
)

# Finalize the config
config = finalize_config(config, initial_state.shape)

# Run the simulation
result = time_integration(initial_state, config, params, registered_variables)
final_state = result.states[-1]

# Calculate the rms velocity and the turbulent crossing time
v = get_absolute_velocity(final_state, config, registered_variables)
v_rms = jnp.sqrt(jnp.mean(v**2))
t_cross = config.box_size.x / v_rms
print(f"RMS velocity: {v_rms:.3f}")
print(f"Turbulent crossing time: {t_cross:.3f}")

# Calculate the final Mach number
final_density = final_state[registered_variables.density_index]
final_pressure = final_state[registered_variables.pressure_index]
final_sound_speed = jnp.sqrt(params.gamma * final_pressure / final_density)
final_mach_number = v_rms / jnp.mean(final_sound_speed)
print(f"Final Mach number: {final_mach_number:.3f}")

# Calculate the final Alfvén Mach number
final_magnetic_field_x = final_state[registered_variables.magnetic_index.x]
final_magnetic_field_y = final_state[registered_variables.magnetic_index.y]
final_magnetic_field_z = final_state[registered_variables.magnetic_index.z]
final_magnetic_energy_density = 0.5 * (final_magnetic_field_x**2 + final_magnetic_field_y**2 + final_magnetic_field_z**2)
final_alfven_speed = jnp.sqrt(2 * final_magnetic_energy_density / final_density)
final_alfven_mach_number = v_rms / jnp.mean(final_alfven_speed)
print(f"Final Alfvén Mach number: {final_alfven_mach_number:.3f}")

# Make animation
print("Creating animation...")
output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)

z_slice = config.num_cells.z // 2
density_states = result.states[:, registered_variables.density_index, :, :, z_slice]

fig, ax = plt.subplots()
im = ax.imshow(
    density_states[0],
    origin="lower",
    extent=(0, config.box_size.x, 0, config.box_size.y),
    norm=LogNorm(vmin=jnp.min(density_states), vmax=jnp.max(density_states)),
    cmap="viridis"
)
cbar = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cbar, label="density")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame_idx):
    im.set_data(density_states[frame_idx])
    ax.set_title(f"Density Field (frame {frame_idx + 1}/{len(density_states)})")
    return (im,)

ani = FuncAnimation(fig, update, frames=len(density_states), interval=10, blit=False)
ani.save(output_dir / "turbulence_test_density.gif", writer=PillowWriter(fps=12))
plt.close(fig)