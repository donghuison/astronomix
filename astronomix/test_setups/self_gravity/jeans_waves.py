# Jeans Linear Waves (3D)

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
_K_VEC          = jnp.array([2.0, 4.0, 4.0])              # wavenumber vector
_BOX_SIZE       = (float(2.0 * jnp.pi / _K_VEC[0]),
                   float(2.0 * jnp.pi / _K_VEC[1]),
                   float(2.0 * jnp.pi / _K_VEC[2]))       # exactly one wavelength per axis
_N_PERIODS      = 3
_GAMMA          = 5.0 / 3.0

_RHO_B          = 1.0          # background density
_C_S_SQUARED    = 1.0          # background sound speed squared
_FOUR_PI_G      = 1.0          # gravitational coupling, gives G = 1/(4π)
_G              = _FOUR_PI_G / (4.0 * jnp.pi)
_EPS            = 1e-6         # perturbation amplitude

# Dispersion relation: w = sqrt(c_s^2 k^2 - 4π G ρ_B); period T = 2π/w.
_K_SQUARED      = float(jnp.sum(_K_VEC ** 2))
_OMEGA          = float(jnp.sqrt(_C_S_SQUARED * _K_SQUARED - _FOUR_PI_G * _RHO_B))
_PERIOD         = 2.0 * jnp.pi / _OMEGA
_T_END          = _N_PERIODS * _PERIOD


def _phase(X, Y, Z, t):
    """Wave phase ``k . x - w t`` at simulation-frame coordinates."""
    return _K_VEC[0] * X + _K_VEC[1] * Y + _K_VEC[2] * Z - _OMEGA * t


def _wave_primitive_state(X, Y, Z, t):
    """
    Cell-centered primitive state of the Jeans linear wave.

    Returns ``(rho, v_x, v_y, v_z, p)`` at simulation-frame coordinates
    (X, Y, Z) and time ``t``. The velocity perturbation is parallel to
    ``k`` (longitudinal mode) with amplitude ``eps * w / k``.
    """
    s = jnp.sin(_phase(X, Y, Z, t))

    rho = _RHO_B + _RHO_B * _EPS * s
    v_x = _EPS * _OMEGA * _K_VEC[0] / _K_SQUARED * s
    v_y = _EPS * _OMEGA * _K_VEC[1] / _K_SQUARED * s
    v_z = _EPS * _OMEGA * _K_VEC[2] / _K_SQUARED * s
    p   = _C_S_SQUARED * _RHO_B / _GAMMA + _C_S_SQUARED * _RHO_B * _EPS * s

    return rho, v_x, v_y, v_z, p


def _generate_state(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    t: float,
) -> STATE_TYPE:
    """
    Generate the discrete primitive state for the Jeans wave at time ``t``.

    The hydrodynamic primitives are evaluated directly at the cell centers
    provided by ``helper_data``. No staggered initialization is required
    since the test is purely hydrodynamic with self-gravity.
    """
    cell_centers = helper_data.geometric_centers
    Xc, Yc, Zc = cell_centers[..., 0], cell_centers[..., 1], cell_centers[..., 2]

    rho, v_x, v_y, v_z, p = _wave_primitive_state(Xc, Yc, Zc, t=t)

    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=v_x,
        velocity_y=v_y,
        velocity_z=v_z,
        gas_pressure=p,
    )


def setup_jeans_wave(
    config: SimulationConfig,
    params: SimulationParams,
) -> tuple[STATE_TYPE, SimulationConfig, SimulationParams]:
    """
    Set up the 3D Jeans linear wave test.

    Enforces the geometry (3D Cartesian), box size (2π/k_x, 2π/k_y, 2π/k_z)
    so that exactly one wavelength fits along each axis, periodic
    boundaries on all faces, end time t = N_periods * T, gamma = 5/3, the
    gravitational constant G = 1/(4π), self-gravity enabled, and MHD
    disabled. The number of cells, solver mode, self-gravity version,
    Riemann solver, slope limiter and CFL number are left untouched. The
    user is responsible for choosing num_cells = (2N, N, N) so that the
    grid spacing is uniform across axes.

    Args:
        config: Simulation configuration.
        params: Simulation parameters.

    Returns:
        state: Initial primitive state of the simulation.
        config: Updated simulation configuration (CARTESIAN geometry,
            box_size = 2π/k per axis, 3D, periodic boundaries, self-gravity
            enabled, MHD disabled).
        params: Updated simulation parameters (t_end = N_periods * T,
            gamma = 5/3, gravitational_constant = 1/(4π)).
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


def jeans_wave_solution(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
) -> STATE_TYPE:
    """
    Exact Jeans linear wave state at t = ``params.t_end``, evaluated on
    the cell centers in ``helper_data``. Because the simulation runs for
    an integer number of full periods T = 2π/w, the analytic state at
    t = t_end is identical to the initial condition.

    Note: the reference state is generated through the same code path as
    ``setup_jeans_wave``, so that the convergence test measures only the
    evolution errors of the numerical scheme, free from grid
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