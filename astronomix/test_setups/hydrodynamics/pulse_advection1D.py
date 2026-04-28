"""
# Gaussian Pulse Advection

A linear advection test in which a Gaussian density pulse is transported by a
uniform velocity field across a periodic domain. Under exact linear advection
the pulse retains its shape, so any deviation between the numerical and exact
solutions is purely numerical error. This makes the test well suited for
measuring the order of convergence of a hydrodynamics scheme.

The domain has length 1.0 with periodic boundaries. The initial state is

    rho(x, 0) = 1 + exp( -(x - x0)^2 / (2 * sigma^2) )
    u(x, 0)   = v_adv
    p(x, 0)   = p0

with x0 = 0.5, sigma = 0.0625, v_adv = 1.0, p0 = 10.0, and gamma = 1.4. The
test is typically run to t = 2.0, corresponding to two full traversals of
the periodic box, so the exact final density profile coincides with the
initial one.
"""

import jax.numpy as jnp

from astronomix import CARTESIAN
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    PERIODIC_BOUNDARY,
    STATE_TYPE,
    BoundarySettings1D,
    SimulationConfig,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# Problem constants
_BOX_SIZE = 1.0
_T_END = 2.0
_GAMMA = 1.4

_PULSE_CENTER = 0.5
_PULSE_WIDTH = 0.0625
_ADVECTION_VELOCITY = 1.0
_PRESSURE = 10.0

def _gaussian_pulse_density(r: jnp.ndarray, t: float) -> jnp.ndarray:
    """Exact density profile of the advected Gaussian pulse at time ``t``."""
    center = (_PULSE_CENTER + _ADVECTION_VELOCITY * t) % _BOX_SIZE
    distance = jnp.abs(r - center)
    return 1.0 + jnp.exp(-distance**2 / (2.0 * _PULSE_WIDTH**2))


def setup_gaussian_pulse_advection(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
) -> tuple[STATE_TYPE, SimulationConfig, SimulationParams]:
    """
    Set up the Gaussian pulse advection test.

    Enforces the geometry, box size, periodic boundaries, end time and gamma
    required by the standard problem. The number of cells, Riemann solver,
    slope limiter and CFL number are left untouched so that the caller can
    study their influence.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.

    Returns:
        state: Initial primitive state of the simulation.
        config: Updated simulation configuration (CARTESIAN geometry,
            box_size = 1.0, periodic boundaries).
        params: Updated simulation parameters (t_end = 2.0, gamma = 1.4).
    """
    config = config._replace(
        geometry = CARTESIAN,
        box_size = _BOX_SIZE,
        dimensionality = 1,
        boundary_settings = BoundarySettings1D(
            left_boundary = PERIODIC_BOUNDARY,
            right_boundary = PERIODIC_BOUNDARY,
        )
    )
    params = params._replace(t_end = _T_END, gamma = _GAMMA)

    r = helper_data.geometric_centers
    rho = _gaussian_pulse_density(r, t = 0.0)
    u   = jnp.full_like(r, _ADVECTION_VELOCITY)
    p   = jnp.full_like(r, _PRESSURE)

    state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u,
        gas_pressure = p,
    )

    config = finalize_config(config, state.shape)

    return state, config, params

def gaussian_pulse_advection_solution(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
) -> STATE_TYPE:
    """
    Exact solution for the Gaussian pulse advection test, evaluated on the
    cell centers in ``helper_data`` at ``params.t_end``.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.

    Returns:
        state: Exact primitive state at t = params.t_end.
    """
    r = helper_data.geometric_centers
    rho = _gaussian_pulse_density(r, t = params.t_end)
    u   = jnp.full_like(r, _ADVECTION_VELOCITY)
    p   = jnp.full_like(r, _PRESSURE)

    return construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = u,
        gas_pressure = p,
    )