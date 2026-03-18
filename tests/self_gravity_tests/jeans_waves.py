"""
Jeans Linear Waves are a self-gravity setup with 
analytical solution.

\rho = \rho_B + \rho_B \eps sin(kx - wt)
v = \eps w k / k^2 sin(kx - wt)
P = c_s^2 \rho_B / \gamma + c_s^2 \rho_B \eps sin(kx - wt)

with

w = sqrt(c_s^2 k^2 - 4 \pi G \rho_B)

Here with \rho_B = 1, c_s^2 = 1, \gamma = 5/3,
4 \pi G = 1, k = 2 \pi (2,4,4)^T, \eps = 1e-6.

For the box length one must ensure proper periodicity 
of the wave. The wavelength is given by

\lambda = 2 \pi / k

and for each dimension a multiple of the 
wavelength must fit in the box.

For the above, e.g. L = 1.0 in all dimensions.

The period is T = 2 \pi / w, when we run for full
periods, we should see the initial conditions exactly reproduced.

We will run for N_periods = 3.

"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================


# numerics
import jax
import jax.numpy as jnp

# enable 64-bit precision for better error measurement
jax.config.update("jax_enable_x64", True)

# plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astronomix.plotting_helpers.power_law_indicators import add_power_law_indicators

# astronomix classes
from astronomix import SimulationConfig
from astronomix import SimulationParams

# astronomix functions
from astronomix import get_helper_data
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import FINITE_VOLUME, PERIODIC_ROLL, finalize_config
from astronomix import get_registered_variables

# astronomix constants
from astronomix.option_classes.simulation_config import (
    PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    FINITE_DIFFERENCE, SIMPLE_SOURCE_TERM
)

def jeans_analytical(
    x, t, rho_B, eps, c_s_squared, k_vec, gamma, G
):
    k_squared = jnp.sum(k_vec ** 2)
    w_squared = c_s_squared * k_squared - 4 * jnp.pi * G * rho_B
    w = jnp.sqrt(w_squared)

    phase = k_vec[0] * x[0] + k_vec[1] * x[1] + k_vec[2] * x[2] - w * t

    rho = rho_B + rho_B * eps * jnp.sin(phase)
    vx = eps * w * k_vec[0] / k_squared * jnp.sin(phase)
    vy = eps * w * k_vec[1] / k_squared * jnp.sin(phase)
    vz = eps * w * k_vec[2] / k_squared * jnp.sin(phase)
    P = c_s_squared * rho_B / gamma + c_s_squared * rho_B * eps * jnp.sin(phase)

    return rho, (vx, vy, vz), P, w

def jeans_simulation(
    num_cells, num_periods = 1, solver_mode = FINITE_DIFFERENCE, animate = False
):
    """
    Return the L1 density error.
    """

    box_size = jnp.pi # in each dimension
    G = 1 / (4 * jnp.pi)

    # setup simulation config
    config = SimulationConfig(
        solver_mode=solver_mode,
        memory_analysis=True,
        progress_bar=True,
        self_gravity=True,
        self_gravity_version=SIMPLE_SOURCE_TERM,
        boundary_handling=PERIODIC_ROLL,
        num_ghost_cells=0,
        mhd=(solver_mode == FINITE_DIFFERENCE),
        dimensionality=3,
        box_size=box_size,
        num_cells=num_cells,
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
        return_snapshots=animate,
        num_snapshots = 48 * num_periods,
    )

    helper_data = get_helper_data(config)

    registered_variables = get_registered_variables(config)

    x = helper_data.geometric_centers
    # shape (num_cells, num_cells, num_cells, 3)
    # to shape (3, num_cells, num_cells, num_cells)
    x = jnp.transpose(x, (3, 0, 1, 2))

    rho_B = 1.0
    c_s_squared = 1.0
    gamma = 5/3
    k_vec = jnp.array([2, 4, 4])
    eps = 1e-6

    initial_rho, initial_v, initial_P, w = jeans_analytical(
        x, t=0.0, rho_B=rho_B, eps=eps, c_s_squared=c_s_squared, k_vec=k_vec, gamma=gamma, G=G
    )

    period = 2 * jnp.pi / w
    t_end = num_periods * period

    # setup simulation params
    params = SimulationParams(
        C_cfl= 1.5 if solver_mode == FINITE_DIFFERENCE else 0.8,
        t_end=t_end,
        gamma=5/3,
        gravitational_constant=G
    )

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=initial_rho,
        velocity_x=initial_v[0],
        velocity_y=initial_v[1],
        velocity_z=initial_v[2],
        gas_pressure=initial_P,
    )

    # finalize config
    config = finalize_config(config, initial_state.shape)

    # run the simulation
    result = time_integration(
        initial_state, config, params, registered_variables
    )

    if animate:
        final_state = result.states[-1]
    else:
        final_state = result

    # plot the initial and final density slices
    fig_comp, axs_comp = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axs_comp[0].imshow(initial_state[registered_variables.density_index, :, :, num_cells // 2], origin='lower', extent=(0, box_size, 0, box_size))
    axs_comp[0].set_title('Initial Density Slice')
    fig_comp.colorbar(im0, ax=axs_comp[0])
    im1 = axs_comp[1].imshow(final_state[registered_variables.density_index, :, :, num_cells // 2], origin='lower', extent=(0, box_size, 0, box_size))
    axs_comp[1].set_title('Final Density Slice')
    fig_comp.colorbar(im1, ax=axs_comp[1])

    projected_abs_diff = jnp.abs(final_state[registered_variables.density_index] - initial_rho).sum(axis=2)
    im2 = axs_comp[2].imshow(projected_abs_diff, origin='lower', extent=(0, box_size, 0, box_size))
    axs_comp[2].set_title('Projected Absolute Density Difference')
    fig_comp.colorbar(im2, ax=axs_comp[2])

    fig_comp.tight_layout()
    fig_comp.savefig(f"figures/jeans_waves_comparison_{num_cells}cells_{'FD' if solver_mode == FINITE_DIFFERENCE else 'FV'}.svg")

    # if animate do a func animation of the density slice evolution
    if animate:
        fig_anim, axs_anim = plt.subplots(1, 2, figsize=(10, 5))

        im = axs_anim[0].imshow(initial_state[registered_variables.density_index, :, :, num_cells // 2], origin='lower', extent=(0, box_size, 0, box_size))
        axs_anim[0].set_title('Density Slice Evolution (Simulated)')
        fig_anim.colorbar(im, ax=axs_anim[0])

        analytical_solution = jeans_analytical(
            x, t=result.time_points[0], rho_B=rho_B, eps=eps, c_s_squared=c_s_squared, k_vec=k_vec, gamma=gamma, G=G
        )
        im_analytical = axs_anim[1].imshow(analytical_solution[0][:, :, num_cells // 2], origin='lower', extent=(0, box_size, 0, box_size))
        axs_anim[1].set_title('Density Slice Evolution (Analytical)')
        fig_anim.colorbar(im_analytical, ax=axs_anim[1])

        def update(frame):
            im.set_data(result.states[frame][registered_variables.density_index, :, :, num_cells // 2])
            analytical_solution = jeans_analytical(
                x, t=result.time_points[frame], rho_B=rho_B, eps=eps, c_s_squared=c_s_squared, k_vec=k_vec, gamma=gamma, G=G
            )
            im_analytical.set_data(analytical_solution[0][:, :, num_cells // 2])
            return im, im_analytical

        anim = FuncAnimation(fig_anim, update, frames=len(result.states), blit=True)
        anim.save(f"figures/jeans_waves_evolution_{num_cells}cells_{'FD' if solver_mode == FINITE_DIFFERENCE else 'FV'}.gif")

    # calculate L1 error
    return jnp.mean(jnp.abs(final_state[registered_variables.density_index] - initial_rho))

resolutions = [16, 32, 64, 96, 128]
errors_fd = []
errors_fv = []

for num_cells in resolutions:
    error = jeans_simulation(num_cells, solver_mode=FINITE_DIFFERENCE)
    errors_fd.append(error)
    print(f"Resolution: {num_cells}, L1 Density Error (FD): {error}")

    error = jeans_simulation(num_cells, solver_mode=FINITE_VOLUME)
    errors_fv.append(error)
    print(f"Resolution: {num_cells}, L1 Density Error (FV): {error}")

# save the errors to a file
# jnp.savez("jeans_waves_errors.npz", resolutions=resolutions, errors_fd=errors_fd, errors_fv=errors_fv)

# plot the errors
fig_errors, ax_errors = plt.subplots(figsize=(8, 5))
ax_errors.plot(resolutions, errors_fd, marker='o', label='FD')
ax_errors.plot(resolutions, errors_fv, marker='s', label='FV')
ax_errors.set_xscale('log')
ax_errors.set_yscale('log')

anchor = (20, 1e-9)

add_power_law_indicators(
    ax=ax_errors,
    anchor=anchor,
    exponents=[-2, -5],
    x_span=2.0,
    scales=[1.0, 1.0],
    x_label='N'
)

ax_errors.set_xlabel('number of cells per dimension')
ax_errors.set_ylabel('L1 density error')
ax_errors.set_title('Jeans Waves L1 Density Error vs Resolution')
ax_errors.legend()
fig_errors.tight_layout()
fig_errors.savefig(f"figures/jeans_waves_error_convergence.svg")