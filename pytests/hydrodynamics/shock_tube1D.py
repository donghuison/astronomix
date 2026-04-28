"""
1D shock tube pytest
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax.numpy as jnp
import matplotlib.pyplot as plt

from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    FINITE_DIFFERENCE,
    HLL,
    HLLC,
    MINMOD,
    SUPERBEE,
    SimulationConfig,
    config_to_string
)
from astronomix.time_stepping.time_integration import time_integration
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.plotting_helpers.inset_box import add_inset_box
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.test_setups.hydrodynamics.shock_tube1D import setup_sod_shock_tube, sod_shock_tube_solution

num_cells = 200

config_list = [
    SimulationConfig(
        solver_mode = FINITE_VOLUME,
        riemann_solver = HLLC,
        limiter = MINMOD,
        num_cells = num_cells,
    ),
    SimulationConfig(
        solver_mode = FINITE_VOLUME,
        riemann_solver = HLLC,
        limiter = SUPERBEE,
        num_cells = num_cells,
    ),
    SimulationConfig(
        solver_mode = FINITE_DIFFERENCE,
        num_cells = num_cells,
    ),
]

def test_shock_tube1D(tol = 1e-2):
    
    # Create figure
    fig, (ax_density, ax_velocity, ax_pressure) = plt.subplots(1, 3, figsize=(15, 5))

    # Get high resolution analytic solution
    config_high_res = SimulationConfig(
        dimensionality = 1,
        num_cells = 1000,
    )

    registered_variables = get_registered_variables(config_high_res)
    helper_data_high_res = get_helper_data(config_high_res)
    _, config_high_res, params_high_res = setup_sod_shock_tube(
        config_high_res,
        registered_variables,
        SimulationParams(),
        helper_data_high_res,
    )
    reference_solution_high_res = sod_shock_tube_solution(
        config_high_res,
        registered_variables,
        params_high_res,
        helper_data_high_res,
    )

    # Plot the reference solution
    ax_density.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.density_index],
        label='Reference',
        color='black'
    )
    ax_velocity.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.velocity_index],
        label='Reference',
        color='black'
    )
    ax_pressure.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.pressure_index],
        label='Reference',
        color='black'
    )
    ax_density.set_xlabel('x')
    ax_density.set_ylabel('Density')
    ax_velocity.set_xlabel('x')
    ax_velocity.set_ylabel('Velocity')
    ax_pressure.set_xlabel('x')
    ax_pressure.set_ylabel('Pressure')
    ax_density.set_title('Density')
    ax_velocity.set_title('Velocity')
    ax_pressure.set_title('Pressure')

    for config in config_list:

        # setup the simulation
        registered_variables = get_registered_variables(config)
        helper_data = get_helper_data(config)

        # get the initial state
        initial_state, config, params = setup_sod_shock_tube(
            config,
            registered_variables,
            SimulationParams(
                C_cfl = 1.5 if config.solver_mode == FINITE_DIFFERENCE else 0.8,
            ),
            helper_data,
        )

        # run the simulation
        final_state = time_integration(
            initial_state,
            config,
            params,
            registered_variables
        )

        # get the reference solution
        true_final_state = sod_shock_tube_solution(
            config,
            registered_variables,
            params,
            helper_data,
        )

        # calculate the density, velocity and pressure error
        density_error = jnp.mean(jnp.abs(final_state[registered_variables.density_index] - true_final_state[registered_variables.density_index]))
        velocity_error = jnp.mean(jnp.abs(final_state[registered_variables.velocity_index] - true_final_state[registered_variables.velocity_index]))
        pressure_error = jnp.mean(jnp.abs(final_state[registered_variables.pressure_index] - true_final_state[registered_variables.pressure_index]))

        # assert that the errors are below the tolerance
        assert density_error < tol, f"Density error {density_error} exceeds tolerance {tol}"
        assert velocity_error < tol, f"Velocity error {velocity_error} exceeds tolerance {tol}"
        assert pressure_error < tol, f"Pressure error {pressure_error} exceeds tolerance {tol}"

        # plot the simulation result
        ax_density.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.density_index],
            label=config_to_string(config),
        )
        ax_velocity.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.velocity_index],
            label=config_to_string(config),
        )
        ax_pressure.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.pressure_index],
            label=config_to_string(config),
        )

    
    add_inset_box(ax_density, x1=0.62, x2=0.72, y1=0.20, y2=0.30, connect_loc1=1)

    handles, labels = ax_density.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig("figures/shock_tube1D_test.svg")

test_shock_tube1D()