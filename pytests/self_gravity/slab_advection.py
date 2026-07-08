"""
3D self-gravitating slab advection convergence pytest.

Advects a self-gravitating density slab across a periodic cubic box and compares
the final state against the analytic solution for the three finite-difference
self-gravity treatments (simple source, flux-based source, corrected flux-based
source). Produces an L1-error-versus-resolution convergence plot in ``figures/``.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# Convergence tests measure a shrinking discretisation error, so we run in double
# precision to keep round-off from setting the floor before the grid does.
jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    GravityConfig,
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.option_classes.simulation_config import solver_mode_to_string
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.time_stepping.time_integration import time_integration
from astronomix.test_setups.self_gravity.slab_advection import (
    setup_slab_advection,
    slab_advection_solution,
)


def _gravity_version_to_string(version: int) -> str:
    """Return a human-readable label for a self-gravity version constant.

    Args:
        version: One of the self-gravity version constants from the simulation
            configuration.

    Returns:
        A short descriptive string naming the self-gravity treatment.
    """
    if version == SIMPLE_SOURCE:
        return "simple source"
    if version == SECOND_ORDER_CONSERVATIVE:
        return "flux-based source"
    if version == FOURTH_ORDER_CONSERVATIVE:
        return "corrected flux-based source"
    raise ValueError(f"Unknown self-gravity version: {version}")


def _config_label(config: SimulationConfig) -> str:
    """Return a plot label combining the solver mode and self-gravity version.

    Args:
        config: The simulation configuration to label.

    Returns:
        A string of the form ``"<solver mode>, <gravity version>"``.
    """
    return (
        f"{solver_mode_to_string(config.solver_mode)}, "
        f"{_gravity_version_to_string(config.gravity_config.self_gravity_version)}"
    )


# Cubic box of side length 3 pi (one wavelength along each axis).
BOX_LENGTH = float(3.0 * jnp.pi)
BOX = StaticFloatVector(BOX_LENGTH, BOX_LENGTH, BOX_LENGTH)

# One base configuration per self-gravity treatment. The resolution (num_cells)
# is injected inside the convergence loop rather than fixed here.
config_list = [
    SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        box_size=BOX,
        mhd=False,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=SIMPLE_SOURCE,
        ),
        dimensionality=3,
        progress_bar=True,
    ),
    SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        box_size=BOX,
        mhd=False,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=SECOND_ORDER_CONSERVATIVE,
        ),
        dimensionality=3,
        progress_bar=True,
    ),
    SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        box_size=BOX,
        mhd=False,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=FOURTH_ORDER_CONSERVATIVE,
        ),
        dimensionality=3,
        progress_bar=True,
    ),
]


def test_slab_advection_convergence():
    """Sweep resolution for each self-gravity treatment and plot the L1 error."""

    # -------------------------------------------------------------
    # =============== ↓ Resolution sweep ↓ ========================
    # -------------------------------------------------------------

    # Resolutions to test: N (cubic N x N x N grid).
    N_values = [16, 32, 64, 96]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Collect the per-resolution L1 error for every configuration.
    errors_dict = {_config_label(config): [] for config in config_list}

    for base_config in config_list:

        for N in N_values:
            # Set the specific 3D grid size for this iteration.
            config = base_config._replace(num_cells=StaticIntVector(N, N, N))

            # Get the initial state. Explicit FV solvers typically need a lower
            # CFL than the SSPRK finite-difference solvers used here.
            initial_state, config, params = setup_slab_advection(
                config,
                SimulationParams(
                    C_cfl=1.5 if config.solver_mode == FINITE_DIFFERENCE else 0.4,
                ),
            )

            registered_variables = get_registered_variables(config)
            helper_data = get_helper_data(config)

            final_state = time_integration(
                initial_state,
                config,
                params,
                registered_variables,
            )

            # Analytic reference solution at the final time.
            true_final_state = slab_advection_solution(
                config,
                registered_variables,
                params,
                helper_data,
            )

            # L1 error of each of the five primitive variables individually.
            error_density = jnp.mean(jnp.abs(
                final_state[registered_variables.density_index]
                - true_final_state[registered_variables.density_index]
            ))
            error_velocity_x = jnp.mean(jnp.abs(
                final_state[registered_variables.velocity_index.x]
                - true_final_state[registered_variables.velocity_index.x]
            ))
            error_velocity_y = jnp.mean(jnp.abs(
                final_state[registered_variables.velocity_index.y]
                - true_final_state[registered_variables.velocity_index.y]
            ))
            error_velocity_z = jnp.mean(jnp.abs(
                final_state[registered_variables.velocity_index.z]
                - true_final_state[registered_variables.velocity_index.z]
            ))
            error_pressure = jnp.mean(jnp.abs(
                final_state[registered_variables.pressure_index]
                - true_final_state[registered_variables.pressure_index]
            ))

            # Average the L1 error over the five primitives.
            total_l1_error = (
                error_density
                + error_velocity_x
                + error_velocity_y
                + error_velocity_z
                + error_pressure
            ) / 5.0

            errors_dict[_config_label(base_config)].append(total_l1_error)

        # Plot the L1 errors for the current config.
        ax.loglog(
            N_values,
            errors_dict[_config_label(base_config)],
            marker='o',
            linewidth=2,
            label=_config_label(base_config),
        )

    # -------------------------------------------------------------
    # =============== ↑ Resolution sweep ↑ ========================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =============== ↓ Reference slopes and formatting ↓ =========
    # -------------------------------------------------------------

    # Mathematical reference slopes to eyeball the observed order against.
    N_arr = np.array(N_values)
    ref_2nd_order = (N_arr / N_arr[0]) ** (-2.0)
    ref_5th_order = (N_arr / N_arr[0]) ** (-5.0)

    # Anchor the reference lines near the actual errors for visual clarity.
    max_err_start = max([errs[0] for errs in errors_dict.values()])
    min_err_start = min([errs[0] for errs in errors_dict.values()])

    ax.loglog(N_arr, max_err_start * ref_2nd_order, 'k--', alpha=0.7, label='$O(N^{-2})$ reference')
    ax.loglog(N_arr, min_err_start * ref_5th_order, 'k:',  alpha=0.7, label='$O(N^{-5})$ reference')

    ax.set_xlabel('N (Grid size: N x N x N)', fontsize=12)
    ax.set_ylabel('Average $L_1$ Error (Primitive Variables)', fontsize=12)
    ax.set_title('3D Self-Gravitating Slab Advection Convergence', fontsize=14)
    ax.set_xticks(N_values)
    ax.set_xticklabels([str(n) for n in N_values])
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    fig.tight_layout()
    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(figures_dir / "slab_advection_convergence_test.svg")

    # -------------------------------------------------------------
    # =============== ↑ Reference slopes and formatting ↑ =========
    # -------------------------------------------------------------


if __name__ == "__main__":
    test_slab_advection_convergence()
