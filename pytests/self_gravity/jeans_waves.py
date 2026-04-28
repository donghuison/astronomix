"""
3D Jeans Linear Wave convergence pytest
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax

# Enable double precision for better accuracy in convergence tests
# (the perturbation amplitude is eps = 1e-6, so float32 cannot resolve
# evolution errors at high resolution).
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.option_classes.simulation_config import (
    FD_FLUX_GRAVITY,
    FINITE_DIFFERENCE,
    SIMPLE_SOURCE_TERM,
    WENO_FLUX_GRAVITY,
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
    solver_mode_to_string,
)
from astronomix.time_stepping.time_integration import time_integration
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import get_registered_variables

# Adjust the import path to match where you saved the jeans_waves logic
from astronomix.test_setups.self_gravity.jeans_waves import (
    setup_jeans_wave,
    jeans_wave_solution,
)


def _gravity_version_to_string(version: int) -> str:
    if version == SIMPLE_SOURCE_TERM:
        return "simple source"
    if version == FD_FLUX_GRAVITY:
        return "flux-based source"
    if version == WENO_FLUX_GRAVITY:
        return "corrected flux-based source"
    raise ValueError(f"Unknown self-gravity version: {version}")


def _config_label(config: SimulationConfig) -> str:
    return (
        f"{solver_mode_to_string(config.solver_mode)}, "
        f"{_gravity_version_to_string(config.self_gravity_version)}"
    )


# Box size matching the wavelengths 2π/k for k_vec = (2, 4, 4):
# Lx = π, Ly = Lz = π/2, so num_cells = (2N, N, N) gives uniform dx.
_BOX = StaticFloatVector(float(jnp.pi), float(jnp.pi / 2), float(jnp.pi / 2))

# Setup configurations without num_cells, which will be injected dynamically inside the loop
config_list = [
    SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        box_size=_BOX,
        mhd=False,
        self_gravity=True,
        self_gravity_version=SIMPLE_SOURCE_TERM,
        dimensionality=3,
        progress_bar=True,
    ),
    SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        box_size=_BOX,
        mhd=False,
        self_gravity=True,
        self_gravity_version=FD_FLUX_GRAVITY,
        dimensionality=3,
        progress_bar=True,
    ),
    SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        box_size=_BOX,
        mhd=False,
        self_gravity=True,
        self_gravity_version=WENO_FLUX_GRAVITY,
        dimensionality=3,
        progress_bar=True,
    ),
]


def test_jeans_wave_convergence():

    # Resolutions to test: N (where the grid is 2N x N x N)
    N_values = [8, 16, 32, 48]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Dictionary to store L1 errors
    errors_dict = {_config_label(config): [] for config in config_list}

    for base_config in config_list:

        for N in N_values:
            # Set the specific 3D grid size for this iteration
            config = base_config._replace(num_cells=StaticIntVector(2 * N, N, N))

            # get the initial state
            # Note: lower CFL is typically needed for explicit FV solvers compared to SSPRK FD solvers
            initial_state, config, params = setup_jeans_wave(
                config,
                SimulationParams(
                    C_cfl=1.5 if config.solver_mode == FINITE_DIFFERENCE else 0.4,
                ),
            )

            # setup the simulation
            registered_variables = get_registered_variables(config)
            helper_data = get_helper_data(config)

            # run the simulation
            final_state = time_integration(
                initial_state,
                config,
                params,
                registered_variables,
            )

            # get the reference analytic solution
            true_final_state = jeans_wave_solution(
                config,
                registered_variables,
                params,
                helper_data,
            )

            # Calculate L1 error for all 5 primitive variables individually
            err_rho = jnp.mean(jnp.abs(final_state[registered_variables.density_index] - true_final_state[registered_variables.density_index]))

            # Vector components: Velocity
            err_vx = jnp.mean(jnp.abs(final_state[registered_variables.velocity_index.x] - true_final_state[registered_variables.velocity_index.x]))
            err_vy = jnp.mean(jnp.abs(final_state[registered_variables.velocity_index.y] - true_final_state[registered_variables.velocity_index.y]))
            err_vz = jnp.mean(jnp.abs(final_state[registered_variables.velocity_index.z] - true_final_state[registered_variables.velocity_index.z]))

            err_p = jnp.mean(jnp.abs(final_state[registered_variables.pressure_index] - true_final_state[registered_variables.pressure_index]))

            # Average over the 5 primitives
            total_l1_error = (err_rho + err_vx + err_vy + err_vz + err_p) / 5.0

            errors_dict[_config_label(base_config)].append(total_l1_error)

            # Basic assertion: ensures error strictly decreases as resolution increases
            # if len(errors_dict[_config_label(base_config)]) > 1:
            #     assert total_l1_error < errors_dict[_config_label(base_config)][-2], \
            #         f"Error did not decrease for {_config_label(base_config)} at N={N}"

        # Plot the L1 errors for the current config
        ax.loglog(
            N_values,
            errors_dict[_config_label(base_config)],
            marker='o',
            linewidth=2,
            label=_config_label(base_config),
        )

    # ==========================================
    # Plot mathematical reference slopes
    # ==========================================
    N_arr = np.array(N_values)
    ref_2nd_order = (N_arr / N_arr[0]) ** (-2.0)
    ref_5th_order = (N_arr / N_arr[0]) ** (-5.0)

    # Scale reference lines to start near the actual errors for visual clarity
    max_err_start = max([errs[0] for errs in errors_dict.values()])
    min_err_start = min([errs[0] for errs in errors_dict.values()])

    ax.loglog(N_arr, max_err_start * ref_2nd_order, 'k--', alpha=0.7, label='$O(N^{-2})$ reference')
    ax.loglog(N_arr, min_err_start * ref_5th_order, 'k:',  alpha=0.7, label='$O(N^{-5})$ reference')

    # Formatting
    ax.set_xlabel('N (Grid size: 2N x N x N)', fontsize=12)
    ax.set_ylabel('Average $L_1$ Error (Primitive Variables)', fontsize=12)
    ax.set_title('3D Jeans Linear Wave Convergence', fontsize=14)
    ax.set_xticks(N_values)
    ax.set_xticklabels([str(n) for n in N_values])
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    fig.tight_layout()
    fig.savefig("figures/jeans_wave_convergence_test.svg")


if __name__ == "__main__":
    test_jeans_wave_convergence()