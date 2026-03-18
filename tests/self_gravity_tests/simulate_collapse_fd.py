# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

# numerics
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

# astronomix classes
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix.option_classes.simulation_config import (
    DONOR_ACCOUNTING,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLLC_LM,
    MIDPOINT_OPTIM,
    RIEMANN_SPLIT,
    RIEMANN_SPLIT_UNSTABLE,
    BoundarySettings,
    BoundarySettings1D,
    SnapshotSettings
)

# astronomix functions
from astronomix import get_helper_data
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix.option_classes.simulation_config import finalize_config
from astronomix import get_registered_variables
from astronomix.plotting_helpers.power_law_indicators import add_power_law_indicators

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD, OSHER, 
    PERIODIC_BOUNDARY, REFLECTIVE_BOUNDARY, 
    BoundarySettings, BoundarySettings1D,
    DOUBLE_MINMOD,
    LAX_FRIEDRICHS,
    MUSCL,
    RK2_SSP,
    SIMPLE_SOURCE_TERM,
    SPLIT,
    UNSPLIT,
    DOUBLE_MINMOD,
    LAX_FRIEDRICHS,
    MUSCL,
    RK2_SSP,
    SIMPLE_SOURCE_TERM,
    SPLIT,
    UNSPLIT,
)

self_gravity_version = SIMPLE_SOURCE_TERM

# simulation settings
gamma = 5/3

# spatial domain
box_size = 4.0

# animate
animate = False

baseline_config_FD = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    runtime_debugging = False,
    progress_bar = True,
    self_gravity = True,
    enforce_positivity=False,
    self_gravity_version = self_gravity_version,
    poisson_manual_open_boundaries=True,
    mhd = True,
    dimensionality = 3,
    box_size = box_size,
    differentiation_mode = FORWARDS,
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
    return_snapshots = False,
    snapshot_settings = SnapshotSettings(
        return_states = animate,
        return_final_state = True,
        return_total_energy = True,
        return_internal_energy = True,
        return_kinetic_energy = True,
        return_gravitational_energy = True
    ),
    num_snapshots = 60
)


baseline_config_fv = SimulationConfig(
    runtime_debugging = False,
    progress_bar = True,
    self_gravity = True,
    self_gravity_version = self_gravity_version,
    poisson_manual_open_boundaries=True,
    first_order_fallback = False,
    dimensionality = 3,
    box_size = box_size,
    split = UNSPLIT,
    differentiation_mode = FORWARDS,
    limiter = MINMOD,
    time_integrator = RK2_SSP, # MIDPOINT_OPTIM
    riemann_solver = HLLC,
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
    return_snapshots = False,
    snapshot_settings = SnapshotSettings(
        return_states = animate,
        return_final_state=True,
        return_total_energy=True,
        return_internal_energy=True,
        return_kinetic_energy=True,
        return_gravitational_energy=True
    ),
    num_snapshots = 60
)

# -------------------------------------------------------------
# =================== ↓ Evrard's Collapse ↓ ===================
# -------------------------------------------------------------

def simulate_collapse(num_cells, t_end = 1.5, return_snapshots = True, solver_mode = FINITE_DIFFERENCE):

    if solver_mode == FINITE_DIFFERENCE:
        baseline_config = baseline_config_FD
    elif solver_mode == FINITE_VOLUME:
        baseline_config = baseline_config_fv
    else:
        raise ValueError("Invalid solver mode")

    print("👷 Setting up simulation...")
    # setup simulation config
    config = baseline_config._replace(
        num_cells = num_cells,
        return_snapshots = return_snapshots,
    )

    helper_data = get_helper_data(config)

    registered_variables = get_registered_variables(config)
  
    R = 1.0
    M = 1.0

    dx = config.box_size / (config.num_cells - 1)

    # initialize density field
    rho = jnp.where(helper_data.r <= R, M / (2 * jnp.pi * R ** 2 * helper_data.r), 1e-4)

    total_injected_mass = jnp.sum(jnp.where(helper_data.r <= R, rho, 0)) * dx ** 3
    print(f"Injected mass: {total_injected_mass}")

    # better ball edges
    # overlap_weights = (R + dx / 2 - helper_data.r) / dx
    # rho = jnp.where((helper_data.r > R - dx / 2) & (helper_data.r < R + dx / 2), rho * overlap_weights, rho)

    # Initialize velocity fields to zero
    v_x = jnp.zeros_like(rho)
    v_y = jnp.zeros_like(rho)
    v_z = jnp.zeros_like(rho)

    # initial thermal energy per unit mass = 0.05
    e = 0.05
    p = (gamma - 1) * rho * e

    B0 = 1e-4

    B_x = jnp.zeros_like(rho)
    B_y = jnp.zeros_like(rho)
    B_z = B0 * jnp.ones_like(rho)

    bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

    # Construct the initial primitive state for the 3D simulation.
    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = v_x,
        velocity_y = v_y,
        velocity_z = v_z,
        gas_pressure = p,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        interface_magnetic_field_x = bxb,
        interface_magnetic_field_y = byb,
        interface_magnetic_field_z = bzb,
    )

    params = SimulationParams(
        t_end = t_end,
        C_cfl = 0.4,
        minimum_density = 1e-5,
        minimum_pressure = 1e-5,
    )

    config = finalize_config(config, initial_state.shape)

    return jax.block_until_ready(
        time_integration(initial_state, config, params, registered_variables)
    ), config, params, helper_data, registered_variables

def energy_error_convergence(num_cells_list = [16, 32, 64, 128], only_plot = True):
    
    # error of the final total energy
    # compared to the initial total energy
    energy_errors_fd = []
    energy_errors_fv = []

    if not only_plot:

        for num_cells in num_cells_list:
            print(f"Running simulation for {num_cells} cells...")

            snapshots_fd, _, _, _, _ = simulate_collapse(num_cells, solver_mode = FINITE_DIFFERENCE)
            snapshots_fv, _, _, _, _ = simulate_collapse(num_cells, solver_mode = FINITE_VOLUME)

            initial_total_energy_fd = snapshots_fd.total_energy[0]
            final_total_energy_fd = snapshots_fd.total_energy[-1]
            energy_error_fd = jnp.abs(final_total_energy_fd - initial_total_energy_fd) / jnp.abs(initial_total_energy_fd)
            energy_errors_fd.append(energy_error_fd)

            initial_total_energy_fv = snapshots_fv.total_energy[0]
            final_total_energy_fv = snapshots_fv.total_energy[-1]
            energy_error_fv = jnp.abs(final_total_energy_fv - initial_total_energy_fv) / jnp.abs(initial_total_energy_fv)
            energy_errors_fv.append(energy_error_fv)

        jnp.savez(
            "collapse_energy_errors.npz",
            num_cells_list = num_cells_list,
            energy_errors_fd = energy_errors_fd,
            energy_errors_fv = energy_errors_fv
        )
    
    else:
        data = jnp.load("collapse_energy_errors.npz")
        num_cells_list = data["num_cells_list"]
        energy_errors_fd = data["energy_errors_fd"]
        energy_errors_fv = data["energy_errors_fv"]

    fig_convergence, ax_convergence = plt.subplots(1, 1, figsize=(6, 4))
    ax_convergence.plot(num_cells_list, energy_errors_fd, label="Finite Difference", marker='o')
    ax_convergence.plot(num_cells_list, energy_errors_fv, label="Finite Volume", marker='o')

    anchor = (20, 2e-1)

    add_power_law_indicators(
        ax=ax_convergence,
        anchor=anchor,
        exponents=[-1, -2],
        x_span=2.0,
        scales=[1.0, 1.0],
        x_label='N'
    )

    ax_convergence.set_xlabel("Number of Cells")
    ax_convergence.set_ylabel("Relative Energy Error")
    ax_convergence.set_xscale("log")
    ax_convergence.set_yscale("log")
    ax_convergence.set_title("Energy Error Convergence")
    ax_convergence.legend()
    fig_convergence.savefig(f"figures/collapse_energy_error_convergence.svg")

        

def resolution_study_collapse():

    num_cells_list = [64, 64, 128, 128]
    solver_modes = [FINITE_VOLUME, FINITE_DIFFERENCE, FINITE_VOLUME, FINITE_DIFFERENCE]
    line_styles = ['-', '--', '-.', ':']

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for num_cells, line_style, solver_mode in zip(num_cells_list, line_styles, solver_modes):
        print(f"Running simulation for {num_cells} cells...")

        solver_string = " (FD)" if solver_mode == FINITE_DIFFERENCE else " (FV)"

        snapshots, _, _, _, registered_variables = simulate_collapse(num_cells, solver_mode = solver_mode)
        total_energy = snapshots.total_energy
        internal_energy = snapshots.internal_energy
        kinetic_energy = snapshots.kinetic_energy
        gravitational_energy = snapshots.gravitational_energy
        time = snapshots.time_points
        ax.plot(time, total_energy, label="Total Energy, N = " + str(num_cells) + solver_string, color = 'black', linestyle = line_style)
        ax.plot(time, internal_energy, label="Internal Energy, N = " + str(num_cells) + solver_string, color = 'green', linestyle = line_style)
        ax.plot(time, kinetic_energy, label="Kinetic Energy, N = " + str(num_cells) + solver_string, color = 'red', linestyle = line_style)
        ax.plot(time, gravitational_energy, label="Gravitational Energy, N = " + str(num_cells) + solver_string, color = 'blue', linestyle = line_style)
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")

        if animate:
            fig_anim, ax_anim = plt.subplots(1, 3, figsize=(15, 5))
            states = snapshots.states
            mid_plane = num_cells // 2

            density_series = states[:, registered_variables.density_index, :, :, mid_plane]
            pressure_series = states[:, registered_variables.pressure_index, :, :, mid_plane]
            v_sq_series = (
                states[:, registered_variables.velocity_index.x, :, :, mid_plane] ** 2
                + states[:, registered_variables.velocity_index.y, :, :, mid_plane] ** 2
                + states[:, registered_variables.velocity_index.z, :, :, mid_plane] ** 2
            )

            def log_bounds(series):
                min_positive = jnp.min(jnp.where(series > 0, series, jnp.inf))
                min_positive = jnp.where(jnp.isfinite(min_positive), min_positive, 1e-30)
                max_value = jnp.max(series)
                max_value = jnp.maximum(max_value, min_positive * 10)
                return float(min_positive), float(max_value)

            density_vmin, density_vmax = log_bounds(density_series)
            pressure_vmin, pressure_vmax = log_bounds(pressure_series)
            v_sq_vmin, v_sq_vmax = log_bounds(v_sq_series)

            im_density = ax_anim[0].imshow(
                jnp.maximum(density_series[0], density_vmin),
                origin='lower',
                extent=(0, box_size, 0, box_size),
                norm=LogNorm(vmin=density_vmin, vmax=density_vmax),
            )
            ax_anim[0].set_title("Density")
            ax_anim[0].set_xlabel("x")
            ax_anim[0].set_ylabel("y")
            fig_anim.colorbar(im_density, ax=ax_anim[0])

            im_pressure = ax_anim[1].imshow(
                jnp.maximum(pressure_series[0], pressure_vmin),
                origin='lower',
                extent=(0, box_size, 0, box_size),
                norm=LogNorm(vmin=pressure_vmin, vmax=pressure_vmax),
            )
            ax_anim[1].set_title("Pressure")
            ax_anim[1].set_xlabel("x")
            ax_anim[1].set_ylabel("y")
            fig_anim.colorbar(im_pressure, ax=ax_anim[1])

            im_v_sq = ax_anim[2].imshow(
                jnp.maximum(v_sq_series[0], v_sq_vmin),
                origin='lower',
                extent=(0, box_size, 0, box_size),
                norm=LogNorm(vmin=v_sq_vmin, vmax=v_sq_vmax),
            )
            ax_anim[2].set_title("Velocity Squared")
            ax_anim[2].set_xlabel("x")
            ax_anim[2].set_ylabel("y")
            fig_anim.colorbar(im_v_sq, ax=ax_anim[2])

            fig_anim.suptitle(f"Collapse Slice Evolution (t = {float(time[0]):.4f})")

            def update(frame):
                im_density.set_data(jnp.maximum(density_series[frame], density_vmin))
                im_pressure.set_data(jnp.maximum(pressure_series[frame], pressure_vmin))
                im_v_sq.set_data(jnp.maximum(v_sq_series[frame], v_sq_vmin))
                fig_anim.suptitle(f"Collapse Slice Evolution (t = {float(time[frame]):.4f})")
                return im_density, im_pressure, im_v_sq

            anim = FuncAnimation(
                fig_anim,
                update,
                frames=density_series.shape[0],
                interval=120,
                blit=False,
            )
            anim.save(
                f"collapse_slice_evolution_{num_cells}_{'simple' if self_gravity_version == SIMPLE_SOURCE_TERM else 'conservative'}.gif",
                fps=10,
            )

            plt.close(fig_anim)

    ax.set_ylim(-2.5, 2.5)
    # ax.set_ylim(-0.7, -0.4)
    # ax.set_xlim(0.8, 1.0)

    ax.legend(fontsize="x-small", ncol=len(num_cells_list))
    ax.set_title("Resolution Study for Evrard's Collapse")

    plt.savefig(f"figures/collapse_resolution_study.svg")

def radial_profile_study():

    num_cells_list = [64,]

    for num_cells in num_cells_list:

        print(f"Running radial profile simulation for {num_cells} cells...")

        final_state, _, params, helper_data, registered_variables = simulate_collapse(num_cells, t_end = 0.8, return_snapshots = False)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        ax1.scatter(helper_data.r.flatten(), final_state[registered_variables.density_index].flatten(), label="Final Density", s = 1)
        # x and y log scale
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim(1e-2, 6e-1)
        ax1.set_ylim(1e-2, 1e3)
        ax1.set_xlabel("r")
        ax1.set_ylabel("Density")

        # velocity profile
        v_r = -jnp.sqrt(final_state[registered_variables.velocity_index.x] ** 2 + final_state[registered_variables.velocity_index.y] ** 2 + final_state[registered_variables.velocity_index.z] ** 2)

        ax2.scatter(helper_data.r.flatten(), v_r.flatten(), label="Radial Velocity", s = 1)
        # log x scale
        ax2.set_xscale("log")
        ax2.set_xlim(1e-2, 6e-1)
        ax2.set_xlabel("r")
        ax2.set_ylabel("Velocity")

        # plot P / rho^gamma
        ax3.scatter(helper_data.r.flatten(), final_state[registered_variables.pressure_index].flatten() / final_state[registered_variables.density_index].flatten() ** params.gamma, label="P / rho^gamma", s = 1)
        ax3.set_xlim(4.0 / num_cells, 6e-1)
        ax3.set_ylim(0, 0.2)
        ax3.set_xlabel("r")
        ax3.set_ylabel("P / rho^gamma")
        ax3.set_xscale("log")

        fig.suptitle("3D Collapse Test")

        plt.tight_layout()

        plt.savefig(f"collapse_radial_profile_{num_cells}.png")


# resolution_study_collapse()
# radial_profile_study()

energy_error_convergence(
    num_cells_list = [16, 32, 64, 96, 128, 160]
)


# -------------------------------------------------------------
# =================== ↑ Evrard's Collapse ↑ ===================
# -------------------------------------------------------------