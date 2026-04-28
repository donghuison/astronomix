# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
import jax.numpy as jnp

# setup
from astronomix import SimulationConfig
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, OPEN_BOUNDARY, BoundarySettings, BoundarySettings1D
)
from astronomix import SimulationParams
from astronomix import get_registered_variables
from astronomix import get_helper_data
from astronomix import construct_primitive_state

# main time integration function
from astronomix import time_integration

# stellar wind
from astronomix import WindParams
from astronomix.option_classes import WindConfig

# turbulent forcing
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig, TurbulentForcingParams

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astronomix.option_classes.simulation_config import finalize_config

# interpolation
from astronomix._finite_difference._maths._differencing import finite_difference_int6
from astronomix._finite_difference._maths._interpolate import interp_center_to_face
from astronomix._finite_difference._maths._interpolate import interp_face_to_center
from astronomix._finite_difference._maths._interpolate import point_values_to_averages

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

XAXIS = 0
YAXIS = 1
ZAXIS = 2

# simulation settings
gamma = 5/3

# spatial domain
box_size = 24.0
num_cells = 256
grid_spacing = box_size / num_cells
x_center = box_size / 2.0
y_center = box_size / 2.0
z_center = box_size / 2.0

# simulation config
config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    grid_spacing = grid_spacing,
    mhd = True,
    enforce_positivity = True,
    progress_bar = True,
    dimensionality = 3,
    box_size = box_size,
    num_cells = num_cells,
    boundary_settings =  BoundarySettings(
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY
        )
    ),
)

# get the variable registry
registered_variables = get_registered_variables(config)

# time domain
C_CFL = 0.8

# initial state
rho_0 = 1.0
p_0 = 1.0

# grid lines
x_l = jnp.linspace(grid_spacing, config.box_size, config.num_cells, endpoint=True)
y_l = jnp.linspace(grid_spacing, config.box_size, config.num_cells, endpoint=True)
z_l = jnp.linspace(grid_spacing, config.box_size, config.num_cells, endpoint=True)

# cell centers
x_c = jnp.linspace(grid_spacing/2, config.box_size + grid_spacing/2, config.num_cells, endpoint=False)
y_c = jnp.linspace(grid_spacing/2, config.box_size + grid_spacing/2, config.num_cells, endpoint=False)
z_c = jnp.linspace(grid_spacing/2, config.box_size + grid_spacing/2, config.num_cells, endpoint=False)

# --- X-Parallel Edges (Diagram: Omega_x) ---
# Intersection of Y-planes and Z-planes.
# Index: (i, j+1/2, k+1/2)
x_yz_edge = x_c  # Center in X
y_yz_edge = y_l  # Line in Y
z_yz_edge = z_l  # Line in Z

x_yz, y_yz, z_yz = jnp.meshgrid(x_yz_edge, y_yz_edge, z_yz_edge, indexing="ij")
r_yz = jnp.sqrt((x_yz - x_center)**2 + (y_yz - y_center)**2 + (z_yz - z_center)**2)

# --- Y-Parallel Edges (Diagram: Omega_y) ---
# Intersection of Z-planes and X-planes.
# Index: (i+1/2, j, k+1/2)
x_zx_edge = x_l  # Line in X
y_zx_edge = y_c  # Center in Y
z_zx_edge = z_l  # Line in Z

x_zx, y_zx, z_zx = jnp.meshgrid(x_zx_edge, y_zx_edge, z_zx_edge, indexing="ij")
r_zx = jnp.sqrt((x_zx - x_center)**2 + (y_zx - y_center)**2 + (z_zx - z_center)**2)

# --- Z-Parallel Edges (Diagram: Omega_z) ---
# Intersection of X-planes and Y-planes.
# Index: (i+1/2, j+1/2, k)
x_xy_edge = x_l  # Line in X
y_xy_edge = y_l  # Line in Y
z_xy_edge = z_c  # Center in Z

x_xy, y_xy, z_xy = jnp.meshgrid(x_xy_edge, y_xy_edge, z_xy_edge, indexing="ij")
r_xy = jnp.sqrt((x_xy - x_center)**2 + (y_xy - y_center)**2 + (z_xy - z_center)**2)

A0 = 20.0

A_x = -jnp.exp(-r_yz ** 2) * (y_yz - y_center)
A_y = jnp.exp(-r_zx ** 2) * (x_zx - x_center)
A_z = 0.5 * A0 * jnp.exp(-r_xy ** 2)

bxb = + 1 / config.grid_spacing * finite_difference_int6(A_z, YAXIS) \
      - 1 / config.grid_spacing * finite_difference_int6(A_y, ZAXIS)

byb = + 1 / config.grid_spacing * finite_difference_int6(A_x, ZAXIS) \
      - 1 / config.grid_spacing * finite_difference_int6(A_z, XAXIS)

bzb = + 1 / config.grid_spacing * finite_difference_int6(A_y, XAXIS) \
      - 1 / config.grid_spacing * finite_difference_int6(A_x, YAXIS)

B_x = interp_face_to_center(bxb, XAXIS)
B_y = interp_face_to_center(byb, YAXIS)
B_z = interp_face_to_center(bzb, ZAXIS)

rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0
u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0

# simulation params
params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = 0.1,
    t_end = 5.0,
    gamma = gamma,
    minimum_density = 1e-2 * rho_0,
    minimum_pressure = 1e-2 * p_0,
)

# construct primitive state
initial_state = construct_primitive_state(
    config = config,
    registered_variables=registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    gas_pressure = p,
    magnetic_field_x = B_x,
    magnetic_field_y = B_y,
    magnetic_field_z = B_z,
    interface_magnetic_field_x = bxb,
    interface_magnetic_field_y = byb,
    interface_magnetic_field_z = bzb,
)

config = finalize_config(config, initial_state.shape)

B_mag = jnp.sqrt(B_x**2 + B_y**2 + B_z**2)
print(jnp.max(B_mag))
print(jnp.argmax(B_mag))

final_state = time_integration(initial_state, config, params, registered_variables)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
y_index = num_cells // 2
ax.imshow(final_state[registered_variables.density_index, :, y_index, :].T, cmap="YlOrRd")

fig.savefig("figures/mhd_jet3D_density.png", dpi=300)