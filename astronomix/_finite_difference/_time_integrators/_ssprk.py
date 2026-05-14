"""
Strong Stability Preserving Runge-Kutta (SSPRK) time integrator.

See _magnetic_update/_constrained_transport.py for more details on the
Constrained Transport (CT) implementation following (Seo & Ryu 2023,
https://arxiv.org/abs/2304.04360).
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Union, Tuple

from astronomix._finite_difference._fluid_equations._enforce_positivity import (
    _enforce_positivity,
)
from astronomix._finite_difference._fluid_equations._equations import conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_x,
    _weno_flux_y,
    _weno_flux_z,
)

from astronomix._finite_difference._magnetic_update._constrained_transport import (
    constrained_transport_rhs,
    update_cell_center_fields,
)
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._physics_modules.run_physics_modules import _physics_sources
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import CONSERVATIVE_GAS_STATE, GHOST_CELLS, MAGNETIC_FIELD_ONLY, PALLAS, SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - optional backend
    pl = None

try:
    from jax.experimental.pallas import triton as pltriton
except Exception:  # pragma: no cover - optional backend
    pltriton = None



# -----------------------------------------------------------------------------
# Optional Pallas backend helpers for hydro-only flux divergence.
# -----------------------------------------------------------------------------


def _backend_name(config: SimulationConfig) -> str:
    backend = getattr(config, "backend", "NATIVE_JAX")
    name = getattr(backend, "name", None)
    if name is not None:
        return str(name).upper()
    value = getattr(backend, "value", None)
    if isinstance(value, str):
        return value.upper()
    return str(backend).upper()


def _backend_is_pallas(config: SimulationConfig) -> bool:
    return config.backend == PALLAS


def _default_pallas_block_shape(ndim: int) -> tuple[int, int, int]:
    if ndim == 1:
        return (128, 1, 1)
    if ndim == 2:
        return (16, 16, 1)
    return (4, 4, 8)


def _as_3tuple_block_shape(block_shape, ndim: int) -> tuple[int, int, int]:
    if block_shape is None:
        return _default_pallas_block_shape(ndim)
    if isinstance(block_shape, str):
        parts = tuple(int(part.strip()) for part in block_shape.split(",") if part.strip())
    else:
        parts = tuple(int(x) for x in block_shape)
    if len(parts) == 1:
        parts = (parts[0], 1, 1)
    elif len(parts) == 2:
        parts = (parts[0], parts[1], 1)
    elif len(parts) >= 3:
        parts = parts[:3]
    else:
        parts = _default_pallas_block_shape(ndim)
    if ndim == 1:
        return (parts[0], 1, 1)
    if ndim == 2:
        return (parts[0], parts[1], 1)
    return parts


def _pallas_compiler_params(config: SimulationConfig):
    use_triton = bool(getattr(config, "pallas_use_triton", True))
    if use_triton and pltriton is not None:
        return pltriton.CompilerParams(num_warps=int(getattr(config, "pallas_num_warps", 4)))
    return None


def _hydro_pallas_divergence_supported(state, config: SimulationConfig) -> bool:
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if bool(getattr(config, "mhd", False)):
        return False
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if state.ndim != ndim + 1:
        return False
    block_shape = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)
    for n, b in zip(state.shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _hydro_flux_divergence_pallas(dF_x, dF_y, dF_z, dtdx, config: SimulationConfig):
    """Pallas flux divergence for hydro RHS after WENO flux reconstruction.

    This replaces the full-array shifted flux differences in the hydro SSPRK RHS.
    MHD/CT does not call this helper.
    """
    ndim = int(config.dimensionality)
    nvars = int(dF_x.shape[0])
    spatial_shape = tuple(int(x) for x in dF_x.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)
    grid = (nx // bx, ny // by, nz // bz)

    if ndim == 1:
        block_shape = (nvars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        flux_spec = pl.BlockSpec(dF_x.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        flux_spec = pl.BlockSpec(dF_x.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        flux_spec = pl.BlockSpec(dF_x.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(fx_ref, fy_ref, fz_ref, dtdx_ref, rhs_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)

        if ndim == 1:
            ii = (bi * bx + jnp.arange(bx)) % nx
        elif ndim == 2:
            ii = (bi * bx + jnp.arange(bx)[:, None]) % nx
            jj = (bj * by + jnp.arange(by)[None, :]) % ny
        else:
            ii = (bi * bx + jnp.arange(bx)[:, None, None]) % nx
            jj = (bj * by + jnp.arange(by)[None, :, None]) % ny
            kk = (bk * bz + jnp.arange(bz)[None, None, :]) % nz

        def fx(var):
            if ndim == 1:
                return fx_ref[var, ii] - fx_ref[var, (ii - 1) % nx]
            if ndim == 2:
                return fx_ref[var, ii, jj] - fx_ref[var, (ii - 1) % nx, jj]
            return fx_ref[var, ii, jj, kk] - fx_ref[var, (ii - 1) % nx, jj, kk]

        def fy(var):
            if ndim == 1:
                return 0.0
            if ndim == 2:
                return fy_ref[var, ii, jj] - fy_ref[var, ii, (jj - 1) % ny]
            return fy_ref[var, ii, jj, kk] - fy_ref[var, ii, (jj - 1) % ny, kk]

        def fz(var):
            if ndim < 3:
                return 0.0
            return fz_ref[var, ii, jj, kk] - fz_ref[var, ii, jj, (kk - 1) % nz]

        for var in range(nvars):
            rhs_ref[var, ...] = -dtdx_ref[()] * (fx(var) + fy(var) + fz(var))

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    dummy_y = dF_x if dF_y is None else dF_y
    dummy_z = dF_x if dF_z is None else dF_z

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(dF_x.shape, dF_x.dtype),
        grid=grid,
        in_specs=[flux_spec, flux_spec, flux_spec, scalar_spec],
        out_specs=out_spec,
        interpret=bool(getattr(config, "pallas_interpret", False)),
        name="hydro_flux_divergence",
        **kwargs,
    )(dF_x, dummy_y, dummy_z, jnp.asarray(dtdx, dtype=dF_x.dtype))


@partial(jax.jit, static_argnames=["registered_variables", "config"], donate_argnames=["conserved_state", "bx_interface", "by_interface", "bz_interface"])
def _ssprk4_with_ct(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Integrates the MHD equations for one time step using a 5-stage, 4th-order
    Strong Stability Preserving Runge-Kutta (SSPRK) method
    with Constrained Transport (CT).
    """

    # for procceses with similar or smaller time scales as the hydrodynamics,
    # they should be included as source terms in the RK stages, otherwise
    # they could be handled outside

    def compute_rhs(current_q, bx, by, bz, k2_coeff):
        """
        Computes the right-hand side (RHS) of the MHD equations for a given stage.
        The `k2_coeff` scales the timestep `dt` for the current RK stage.
        """

        current_q = update_cell_center_fields(
            current_q, bx, by, bz, config, registered_variables
        )

        dt_tilde = k2_coeff * dt

        # in the future we might support
        # different grid spacings in each direction
        dtdx = dt_tilde / grid_spacing
        dtdy = dt_tilde / grid_spacing
        dtdz = dt_tilde / grid_spacing

        # Calculate fluxes based on the state of the current stage
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)

        if config.dimensionality == 1:
            dF_y = 0.0
            dF_z = 0.0

        if config.dimensionality == 2:
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            dF_z = 0.0

        if config.dimensionality == 3:
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)

        # Calculate RHS for interface magnetic fields using Constrained Transport
        rhs_bx, rhs_by, rhs_bz = constrained_transport_rhs(
            current_q, dF_x, dF_y, dF_z, dtdx, dtdy, dtdz, config, registered_variables
        )

        # Calculate RHS for conserved fluid variables
        if config.dimensionality == 1:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
            )
        elif config.dimensionality == 2:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
                + (dF_y - _shift(dF_y, 1, axis=2))
            )
        elif config.dimensionality == 3:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
                + (dF_y - _shift(dF_y, 1, axis=2))
                + (dF_z - _shift(dF_z, 1, axis=3))
            )

        if config.dimensionality == 1:
            density_fluxes = (dF_x[registered_variables.density_index],)
        elif config.dimensionality == 2:
            density_fluxes = (dF_x[registered_variables.density_index], dF_y[registered_variables.density_index])
        elif config.dimensionality == 3:
            density_fluxes = (dF_x[registered_variables.density_index], dF_y[registered_variables.density_index], dF_z[registered_variables.density_index])


        # Add physics source terms
        rhs_q += _physics_sources(
            current_q,
            density_fluxes,
            rhs_q[registered_variables.density_index], # drho
            dt_tilde,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
        )

        return rhs_q, rhs_bx, rhs_by, rhs_bz

    # define the SSPRK4 coefficients

    k1_1 = 1.0
    k2_1 = 0.39175222700392
    k3_1 = 0.0

    k1_2 = 0.44437049406734
    k2_2 = 0.36841059262959
    k3_2 = 0.55562950593266

    k1_3 = 0.62010185138540
    k2_3 = 0.25189177424738
    k3_3 = 0.37989814861460
    
    k1_4 = 0.17807995410773
    k2_4 = 0.54497475021237
    k3_4 = 0.82192004589227

    k1_5 = -2.081261929715610e-02
    k2_5 = 0.22600748319395
    k3_5 = 5.03580947213895e-01
    k4_5 = 0.51723167208978
    k5_5 = -6.518979800418380e-12

    final_factors = jnp.array([k1_5, 0.0, k4_5, k5_5, k3_5])
    k_rhs_s = jnp.array([k2_1, k2_2, k2_3, k2_4, k2_5])
    k_0_s = jnp.array([k1_1, k1_2, k1_3, k1_4, k1_5])
    k_curr_s = jnp.array([k3_1, k3_2, k3_3, k3_4, k3_5])

    # Store the initial state (t = n)
    q0 = conserved_state
    bx0, by0, bz0 = bx_interface, by_interface, bz_interface

    def ssprk_stage(stage_idx, carry):

        # unpack carry
        current_state, final_state = carry
        q_curr, bx_curr, by_curr, bz_curr = current_state
        q_final, bx_final, by_final, bz_final = final_state

        if config.enforce_positivity:
            q_curr = _enforce_positivity(
                q_curr,
                config,
                gamma,
                params.minimum_density,
                params.minimum_pressure,
                registered_variables,
            )

        if config.boundary_handling == GHOST_CELLS:
            q_curr = _boundary_handler(
                q_curr, config, registered_variables, params, CONSERVATIVE_GAS_STATE
            )
            b_curr = _boundary_handler(
                jnp.stack([bx_curr, by_curr, bz_curr], axis=0),
                config,
                registered_variables,
                params,
                MAGNETIC_FIELD_ONLY
            )
            bx_curr, by_curr, bz_curr = b_curr[0], b_curr[1], b_curr[2]

        k_rhs = k_rhs_s[stage_idx]
        k_0 = k_0_s[stage_idx]
        k_curr = k_curr_s[stage_idx]

        # update the current state
        rhs_q, rhs_bx, rhs_by, rhs_bz = compute_rhs(q_curr, bx_curr, by_curr, bz_curr, k_rhs)
        q_curr = k_0 * q0 + k_curr * q_curr + rhs_q
        bx_curr = k_0 * bx0 + k_curr * bx_curr + rhs_bx
        by_curr = k_0 * by0 + k_curr * by_curr + rhs_by
        bz_curr = k_0 * bz0 + k_curr * bz_curr + rhs_bz

        # update the final state
        final_factor = final_factors[stage_idx + 1]
        q_final += q_curr * final_factor
        bx_final += bx_curr * final_factor
        by_final += by_curr * final_factor
        bz_final += bz_curr * final_factor

        return (
            (q_curr, bx_curr, by_curr, bz_curr), 
            (q_final, bx_final, by_final, bz_final)
        )

    # one could also write out everything (which is what I originally had),
    # I used the fori_loop to possibly reduce the memory footprint
    (
        (q4, bx4, by4, bz4),
        (q_final, bx_final, by_final, bz_final)
    ) = jax.lax.fori_loop(0, 4, ssprk_stage, 
        (
            (q0, bx0, by0, bz0), 
            (final_factors[0] * q0, final_factors[0] * bx0, final_factors[0] * by0, final_factors[0] * bz0)
        )
    )

    # Final Stage (Stage 5)
    rhs_q4, rhs_bx4, rhs_by4, rhs_bz4 = compute_rhs(q4, bx4, by4, bz4, k2_5)
    q_final = q_final + rhs_q4
    bx_final = bx_final + rhs_bx4
    by_final = by_final + rhs_by4
    bz_final = bz_final + rhs_bz4

    # Update the cell-centered magnetic fields in the conserved state array
    # from the final interface magnetic fields.
    q_final = update_cell_center_fields(
        q_final, bx_final, by_final, bz_final, config, registered_variables
    )

    if config.enforce_positivity:
        q_final = _enforce_positivity(
            q_final,
            config,
            gamma,
            params.minimum_density,
            params.minimum_pressure,
            registered_variables,
        )
    
    return q_final, bx_final, by_final, bz_final


@partial(jax.jit, static_argnames=["registered_variables", "config"], donate_argnames=["conserved_state"])
def _ssprk4_hydro(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params, # Assuming SimulationParams type
    helper_data, # Assuming HelperData type
    config, # Assuming SimulationConfig type
    registered_variables: RegisteredVariables,
):
    """
    Integrates the Euler (hydrodynamics) equations for one time step using a 
    5-stage, 4th-order Strong Stability Preserving Runge-Kutta (SSPRK) method.
    """

    # for procceses with similar or smaller time scales as the hydrodynamics,
    # they should be included as source terms in the RK stages, otherwise
    # they could be handled outside

    def compute_rhs(current_q, k2_coeff):
        """
        Computes the right-hand side (RHS) of the hydro equations for a given stage.
        The `k2_coeff` scales the timestep `dt` for the current RK stage.
        """

        dt_tilde = k2_coeff * dt

        # in the future we might support
        # different grid spacings in each direction
        dtdx = dt_tilde / grid_spacing
        dtdy = dt_tilde / grid_spacing
        dtdz = dt_tilde / grid_spacing

        # Calculate fluxes based on the state of the current stage
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)

        if config.dimensionality >= 2:
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)

        if config.dimensionality == 3:
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)

        # Calculate RHS for conserved fluid variables.  In hydro/Pallas mode,
        # avoid materialising shifted flux arrays for the divergence.  The MHD/CT
        # integrator above keeps using the original native-JAX path.
        if _hydro_pallas_divergence_supported(current_q, config):
            rhs_q = _hydro_flux_divergence_pallas(
                dF_x,
                dF_y if config.dimensionality >= 2 else None,
                dF_z if config.dimensionality == 3 else None,
                dtdx,
                config,
            )
        elif config.dimensionality == 1:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
            )
        elif config.dimensionality == 2:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
                + (dF_y - _shift(dF_y, 1, axis=2))
            )
        elif config.dimensionality == 3:
            rhs_q = -dtdx * (
                (dF_x - _shift(dF_x, 1, axis=1))
                + (dF_y - _shift(dF_y, 1, axis=2))
                + (dF_z - _shift(dF_z, 1, axis=3))
            )

        if config.dimensionality == 1:
            density_fluxes = (dF_x[registered_variables.density_index],)
        elif config.dimensionality == 2:
            density_fluxes = (dF_x[registered_variables.density_index], dF_y[registered_variables.density_index])
        elif config.dimensionality == 3:
            density_fluxes = (dF_x[registered_variables.density_index], dF_y[registered_variables.density_index], dF_z[registered_variables.density_index])

        # Add physics source terms
        rhs_q += _physics_sources(
            current_q,
            density_fluxes,
            rhs_q[registered_variables.density_index], # drho
            dt_tilde,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
        )

        return rhs_q

    # define the SSPRK4 coefficients

    k1_1 = 1.0
    k2_1 = 0.39175222700392
    k3_1 = 0.0

    k1_2 = 0.44437049406734
    k2_2 = 0.36841059262959
    k3_2 = 0.55562950593266

    k1_3 = 0.62010185138540
    k2_3 = 0.25189177424738
    k3_3 = 0.37989814861460
    
    k1_4 = 0.17807995410773
    k2_4 = 0.54497475021237
    k3_4 = 0.82192004589227

    k1_5 = -2.081261929715610e-02
    k2_5 = 0.22600748319395
    k3_5 = 5.03580947213895e-01
    k4_5 = 0.51723167208978
    k5_5 = -6.518979800418380e-12

    final_factors = jnp.array([k1_5, 0.0, k4_5, k5_5, k3_5])
    k_rhs_s = jnp.array([k2_1, k2_2, k2_3, k2_4, k2_5])
    k_0_s = jnp.array([k1_1, k1_2, k1_3, k1_4, k1_5])
    k_curr_s = jnp.array([k3_1, k3_2, k3_3, k3_4, k3_5])

    # Store the initial state (t = n)
    q0 = conserved_state

    def ssprk_stage(stage_idx, carry):

        # unpack carry
        q_curr, q_final = carry

        if config.enforce_positivity:
            q_curr = _enforce_positivity(
                q_curr,
                config,
                gamma,
                params.minimum_density,
                params.minimum_pressure,
                registered_variables,
            )

        if config.boundary_handling == GHOST_CELLS:
            q_curr = _boundary_handler(
                q_curr, config, registered_variables, params, CONSERVATIVE_GAS_STATE
            )

        k_rhs = k_rhs_s[stage_idx]
        k_0 = k_0_s[stage_idx]
        k_curr = k_curr_s[stage_idx]

        # update the current state
        rhs_q = compute_rhs(q_curr, k_rhs)
        q_curr = k_0 * q0 + k_curr * q_curr + rhs_q

        # update the final state
        final_factor = final_factors[stage_idx + 1]
        q_final += q_curr * final_factor

        return (q_curr, q_final)

    q4, q_final = jax.lax.fori_loop(
        0, 4, ssprk_stage, (q0, final_factors[0] * q0)
    )

    # Final Stage (Stage 5)
    rhs_q4 = compute_rhs(q4, k2_5)
    q_final = q_final + rhs_q4

    if config.enforce_positivity:
        q_final = _enforce_positivity(
            q_final,
            config,
            gamma,
            params.minimum_density,
            params.minimum_pressure,
            registered_variables,
        )
    
    return q_final