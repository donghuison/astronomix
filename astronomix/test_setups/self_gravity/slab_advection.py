# Self-Gravitating Slab Advection (3D)

"""
Advection of self-gravitating slabs in equilibrium.

In the rest frame the slab is in static self-gravitating equilibrium with
a sinusoidal density perturbation along the direction k:

    rho(x) = rho_0 [1 + eps cos(k.x) + (eps^2 / 3) cos(2 k.x)]

The pressure is determined to satisfy hydrostatic equilibrium with
self-gravity (rho grad Phi + grad P = 0, laplacian Phi = 4 pi G rho),
yielding the perturbation expansion in eps:

    P(x) = p_0 + (4 pi G eps rho_0^2 / k^2) [
                (1 - eps^2 / 12)  cos(   k.x)
              + (eps   /  3)      cos( 2 k.x)
              + (eps^2 / 12)      cos( 3 k.x)
              + (eps^3 / 144)     cos( 4 k.x)
           ]

The equilibrium is then advected at a uniform velocity v parallel to k,
so the analytic state at any time t is the rest-frame configuration
evaluated at phase k.x - w t with w = k.v. After one period
T = 2 pi / w the state returns to the initial condition exactly.

Standard parameters: rho_0 = 1, p_0 = 6, eps = 0.3, gamma = 5/3,
4 pi G = 1, k = (2/3, 2/3, 2/3), v = (0.6, 0.6, 0.6). The cubic box has
side length L = 2 pi / min(k_i) = 3 pi so that exactly one wavelength
fits along each axis. The standard end time is t = T = 2 pi / (k.v).

"""

import jax.numpy as jnp

from astronomix import CARTESIAN
from astronomix.data_classes.simulation_helper_data import HelperData, get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    PERIODIC_BOUNDARY,
    STATE_TYPE,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import (
    RegisteredVariables,
    get_registered_variables,
)


# Problem constants
_K_VEC          = jnp.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])  # wavenumber vector
_V_VEC          = jnp.array([0.6, 0.6, 0.6])                    # advection velocity (parallel to k)
_BOX_LEN        = float(2.0 * jnp.pi * jnp.max(1.0 / _K_VEC))   # one wavelength per axis -> 3 pi
_BOX_SIZE       = (_BOX_LEN, _BOX_LEN, _BOX_LEN)
_N_PERIODS      = 1
_GAMMA          = 5.0 / 3.0

_RHO_0          = 1.0          # background density
_P_0            = 6.0          # background pressure
_EPS            = 0.3          # density perturbation amplitude
_FOUR_PI_G      = 1.0          # gravitational coupling, gives G = 1/(4 pi)
_G              = _FOUR_PI_G / (4.0 * jnp.pi)

# Derived: advection frequency w = k . v, period T = 2 pi / w.
_K_SQUARED      = float(jnp.sum(_K_VEC ** 2))
_OMEGA          = float(jnp.sum(_K_VEC * _V_VEC))
_PERIOD         = 2.0 * jnp.pi / _OMEGA
_T_END          = _N_PERIODS * _PERIOD


def _phase(X, Y, Z, t):
    """Wave phase ``k . x - w t`` at simulation-frame coordinates, with
    ``w = k . v`` because the equilibrium is rigidly translated along v."""
    return _K_VEC[0] * X + _K_VEC[1] * Y + _K_VEC[2] * Z - _OMEGA * t


def _slab_primitive_state(X, Y, Z, t):
    """
    Cell-centered primitive state of the advected self-gravitating slab.

    Returns ``(rho, v_x, v_y, v_z, p)`` at simulation-frame coordinates
    (X, Y, Z) and time ``t``. The velocity field is uniform; density and
    pressure carry the equilibrium perturbation expansion in ``eps``.
    """
    phi = _phase(X, Y, Z, t)
    cos1, cos2, cos3, cos4 = (
        jnp.cos(phi),
        jnp.cos(2.0 * phi),
        jnp.cos(3.0 * phi),
        jnp.cos(4.0 * phi),
    )

    rho = _RHO_0 * (1.0 + _EPS * cos1 + (_EPS ** 2) / 3.0 * cos2)

    v_x = jnp.full_like(X, _V_VEC[0])
    v_y = jnp.full_like(X, _V_VEC[1])
    v_z = jnp.full_like(X, _V_VEC[2])

    p = _P_0 + (_FOUR_PI_G * _EPS * _RHO_0 ** 2 / _K_SQUARED) * (
        (1.0 - (_EPS ** 2) / 12.0) * cos1
        + (_EPS / 3.0)              * cos2
        + ((_EPS ** 2) / 12.0)      * cos3
        + ((_EPS ** 3) / 144.0)     * cos4
    )

    return rho, v_x, v_y, v_z, p


def _generate_state(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    t: float,
) -> STATE_TYPE:
    """
    Generate the discrete primitive state for the advected slab at time
    ``t``. The hydrodynamic primitives are evaluated directly at the cell
    centers provided by ``helper_data``.
    """
    cell_centers = helper_data.geometric_centers
    Xc, Yc, Zc = cell_centers[..., 0], cell_centers[..., 1], cell_centers[..., 2]

    rho, v_x, v_y, v_z, p = _slab_primitive_state(Xc, Yc, Zc, t=t)

    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=v_x,
        velocity_y=v_y,
        velocity_z=v_z,
        gas_pressure=p,
    )


def setup_slab_advection(
    config: SimulationConfig,
    params: SimulationParams,
) -> tuple[STATE_TYPE, SimulationConfig, SimulationParams]:
    """
    Set up the 3D self-gravitating slab advection test.

    Enforces the geometry (3D Cartesian), cubic box of side length
    L = 2 pi / min(k_i) = 3 pi so that one wavelength fits along each
    axis, periodic boundaries on all faces, end time t = N_periods * T,
    gamma = 5/3, the gravitational constant G = 1/(4 pi), self-gravity
    enabled, and MHD disabled. The number of cells, solver mode,
    self-gravity version, Riemann solver, slope limiter and CFL number
    are left untouched. Because the box is cubic, the natural choice is
    num_cells = (N, N, N) for uniform grid spacing.

    Args:
        config: Simulation configuration.
        params: Simulation parameters.

    Returns:
        state: Initial primitive state of the simulation.
        config: Updated simulation configuration (CARTESIAN geometry,
            box_size = (3 pi, 3 pi, 3 pi), 3D, periodic boundaries,
            self-gravity enabled, MHD disabled).
        params: Updated simulation parameters (t_end = N_periods * T,
            gamma = 5/3, gravitational_constant = 1/(4 pi)).
    """
    config = config._replace(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=StaticFloatVector(*_BOX_SIZE),
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        mhd=False,
        self_gravity=True,
    )
    params = params._replace(
        t_end=_T_END,
        gamma=_GAMMA,
        gravitational_constant=_G,
    )

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    state = _generate_state(
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
        t=0.0,
    )

    config = finalize_config(config, state.shape)

    return state, config, params


def slab_advection_solution(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
) -> STATE_TYPE:
    """
    Exact slab advection state at t = ``params.t_end``, evaluated on the
    cell centers in ``helper_data``. Because the simulation runs for an
    integer number of full periods T = 2 pi / (k.v), the analytic state
    at t = t_end is identical to the initial condition.

    Note: the reference state is generated through the same code path as
    ``setup_slab_advection``, so that the convergence test measures only
    the evolution errors of the numerical scheme, free from grid
    initialization mismatches.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.

    Returns:
        state: Exact primitive state at t = params.t_end.
    """
    return _generate_state(
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
        t=params.t_end,
    )