"""
# Sod Shock Tube

The Sod shock tube (Sod 1978) is a classic 1D Riemann problem that simultaneously
exercises a fluid simulator's treatment of shocks, contact discontinuities, and
rarefaction waves. Two constant fluid states are separated by a diaphragm at
x = 0.5 inside a box of length 1.0:

    Left state  (x < 0.5): rho = 1.0,   u = 0.0, p = 1.0
    Right state (x > 0.5): rho = 0.125, u = 0.0, p = 0.1

Once the diaphragm is removed, the flow develops a leftward-moving rarefaction,
a rightward-moving contact discontinuity, and a rightward-moving shock. The
test is typically evaluated at t = 0.2 with gamma = 5/3.

## References

- Sod, G. A. (1978). "A survey of several finite difference methods for systems
  of nonlinear hyperbolic conservation laws". Journal of Computational Physics,
  27(1), 1-31.
- Toro, E. F. (2009). "Riemann Solvers and Numerical Methods for Fluid Dynamics",
  3rd ed., Springer.
"""

import jax.numpy as jnp
import numpy as np

from astronomix import CARTESIAN
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import OPEN_BOUNDARY, STATE_TYPE, BoundarySettings1D, SimulationConfig, finalize_config
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.test_setups.reference_solutions.riemann_solver import _exact_riemann_ideal_gas
from astronomix.variable_registry.registered_variables import RegisteredVariables

# Problem constants
_SHOCK_POS = 0.5
_GAMMA = 5/3
_BOX_SIZE = 1.0
_T_END = 0.2

_RHO_L, _U_L, _P_L = 1.0,   0.0, 1.0
_RHO_R, _U_R, _P_R = 0.125, 0.0, 0.1


def setup_sod_shock_tube(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
) -> tuple[STATE_TYPE, SimulationConfig, SimulationParams]:
    """
    Set up the Sod shock tube test.

    Enforces the geometry, box size and end time required by the standard
    problem. The number of cells, Riemann solver, and slope limiter are
    left untouched so that the caller can study their influence.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.

    Returns:
        state: Initial primitive state of the simulation.
        config: Updated simulation configuration (CARTESIAN geometry,
            box_size = 1.0).
        params: Updated simulation parameters (t_end = 0.2, gamma = 5/3).
    """
    config = config._replace(
        geometry = CARTESIAN,
        box_size = _BOX_SIZE,
        dimensionality = 1,
        boundary_settings = BoundarySettings1D(
            left_boundary = OPEN_BOUNDARY,
            right_boundary = OPEN_BOUNDARY,
        )
    )
    params = params._replace(t_end = _T_END, gamma = _GAMMA)

    r = helper_data.geometric_centers
    rho = jnp.where(r < _SHOCK_POS, _RHO_L, _RHO_R)
    u   = jnp.where(r < _SHOCK_POS, _U_L,   _U_R)
    p   = jnp.where(r < _SHOCK_POS, _P_L,   _P_R)

    state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u,
        gas_pressure = p,
    )

    config = finalize_config(config, state.shape)

    return state, config, params


def sod_shock_tube_solution(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
) -> STATE_TYPE:
    """
    Exact Riemann solution for the Sod shock tube, evaluated on the cell
    centers in ``helper_data`` at ``params.t_end``.

    Requires the ``exactpack`` package.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.

    Returns:
        state: Exact primitive state at t = params.t_end.
    """

    rho, u, p = _exact_riemann_ideal_gas(
        rho_L = _RHO_L, u_L = _U_L, p_L = _P_L,
        rho_R = _RHO_R, u_R = _U_R, p_R = _P_R,
        gamma = params.gamma,
        x = helper_data.geometric_centers,
        t = params.t_end,
        x0 = _SHOCK_POS,
    )

    return construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u,
        gas_pressure = p,
    )