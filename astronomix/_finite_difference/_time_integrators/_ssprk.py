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
    _hydro_pallas_flux_supported,
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
from astronomix.option_classes.simulation_config import CONSERVATIVE_GAS_STATE, GHOST_CELLS, MAGNETIC_FIELD_ONLY, PALLAS, SIMPLE_SOURCE_TERM, SimulationConfig
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


def _div_axis_pallas_shape_ok(state, config: SimulationConfig) -> bool:
    """Lightweight predicate used by callers (e.g. the MHD CT integrator) that
    want the per-axis divergence Pallas kernel but cannot rely on the
    full hydro-WENO support predicate (which excludes MHD on its WENO step).

    The divergence kernel itself is mhd-agnostic — it just walks every
    variable channel — so this only checks the spatial block-divisibility
    constraint required by ``pl.pallas_call``.
    """
    if pl is None:
        return False
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if state.ndim != ndim + 1:
        return False
    bx, by, bz = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)
    for n, b in zip(state.shape[1:], (bx, by, bz)[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _hydro_flux_div_axis_pallas(
    dF,
    dt_over_dx,
    config: SimulationConfig,
    *,
    axis: int,
    rhs_accumulator=None,
    scale_in: Union[float, jnp.ndarray] = 1.0,
):
    """Per-axis Pallas divergence kernel with optional in-place accumulation.

    Computes ``rhs_out = scale_in * (rhs_accumulator if provided else 0) +
    (-dt_over_dx) * (dF[..., i+1/2] - dF[..., (i+1/2)-1])`` along ``axis``.
    Calling it sequentially for each axis with ``rhs_accumulator=rhs_q`` lets
    XLA keep a single physical RHS buffer (via ``input_output_aliases``) across
    all three axes, eliminating both the chained ``rhs + ...`` adds and the
    transient buffers they would otherwise need.

    ``scale_in`` is folded into the kernel so the LSRK4 first-stage update
    ``dq = A[i] * dq + (-dt/dx) * div_0(F_0)`` can be done in place on the
    ``dq`` buffer without materialising a separate ``rhs`` register.

    This keeps the original 1-flux-per-cell WENO kernel (so peak compute is
    unchanged) while still consuming each ``dF_axis`` immediately after it is
    produced, instead of holding all three live for the original
    three-input divergence helper.
    """
    ndim = int(config.dimensionality)
    nvars = int(dF.shape[0])
    spatial_shape = tuple(int(x) for x in dF.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)
    grid = (nx // bx, ny // by, nz // bz)

    accumulate = rhs_accumulator is not None

    if ndim == 1:
        block_shape = (nvars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        flux_spec = pl.BlockSpec(dF.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        flux_spec = pl.BlockSpec(dF.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        flux_spec = pl.BlockSpec(dF.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(*refs):
        if accumulate:
            rhs_in_ref, f_ref, dtdx_ref, scale_in_ref, rhs_out_ref = refs
        else:
            f_ref, dtdx_ref, rhs_out_ref = refs
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

        dtdx = dtdx_ref[()]

        def flux_diff(var):
            if axis == 0:
                if ndim == 1:
                    return f_ref[var, ii] - f_ref[var, (ii - 1) % nx]
                if ndim == 2:
                    return f_ref[var, ii, jj] - f_ref[var, (ii - 1) % nx, jj]
                return f_ref[var, ii, jj, kk] - f_ref[var, (ii - 1) % nx, jj, kk]
            if axis == 1:
                if ndim == 2:
                    return f_ref[var, ii, jj] - f_ref[var, ii, (jj - 1) % ny]
                return f_ref[var, ii, jj, kk] - f_ref[var, ii, (jj - 1) % ny, kk]
            return f_ref[var, ii, jj, kk] - f_ref[var, ii, jj, (kk - 1) % nz]

        if accumulate:
            scale = scale_in_ref[()]
            for var in range(nvars):
                rhs_out_ref[var, ...] = scale * rhs_in_ref[var, ...] + (-dtdx) * flux_diff(var)
        else:
            for var in range(nvars):
                rhs_out_ref[var, ...] = -dtdx * flux_diff(var)

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    if accumulate:
        in_specs = [out_spec, flux_spec, scalar_spec, scalar_spec]
        kernel_args = (
            rhs_accumulator,
            dF,
            jnp.asarray(dt_over_dx, dtype=dF.dtype),
            jnp.asarray(scale_in, dtype=dF.dtype),
        )
        kwargs["input_output_aliases"] = {0: 0}
    else:
        in_specs = [flux_spec, scalar_spec]
        kernel_args = (dF, jnp.asarray(dt_over_dx, dtype=dF.dtype))

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(dF.shape, dF.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        interpret=bool(getattr(config, "pallas_interpret", False)),
        name=f"hydro_flux_div_axis_{axis}{'_acc' if accumulate else ''}",
        **kwargs,
    )(*kernel_args)


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

    # For the MHD/CT path the WENO kernel itself is still native (the Pallas
    # MHD WENO kernel is not yet written — see guide §4.1).  We can still pick
    # up the per-axis divergence accumulator + ``input_output_aliases``
    # memory win whenever the user selected the Pallas backend, since that
    # kernel is mhd-agnostic and just operates on whatever flux tensor it is
    # handed.
    use_pallas_div = (
        _backend_is_pallas(config) and pl is not None
        and _div_axis_pallas_shape_ok(conserved_state, config)
    )

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

        # Calculate fluxes based on the state of the current stage.  All three
        # dF arrays must be retained here because constrained_transport_rhs
        # needs them simultaneously to compute the edge-centered EMFs.  The
        # divergence step further down is the only place where we can save
        # memory — see below.
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

        # Calculate RHS for conserved fluid variables.  Under Pallas mode we
        # use the per-axis div+accumulator kernel so the single physical RHS
        # buffer is reused across axes (matches the hydro Pallas path).
        if use_pallas_div:
            rhs_q = _hydro_flux_div_axis_pallas(dF_x, dtdx, config, axis=0)
            if config.dimensionality >= 2:
                rhs_q = _hydro_flux_div_axis_pallas(
                    dF_y, dtdy, config, axis=1, rhs_accumulator=rhs_q
                )
            if config.dimensionality == 3:
                rhs_q = _hydro_flux_div_axis_pallas(
                    dF_z, dtdz, config, axis=2, rhs_accumulator=rhs_q
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


def _hydro_density_fluxes_needed(config) -> bool:
    """Whether any FD physics module actually consumes the per-axis density
    flux slices.  Only self-gravity variants other than SIMPLE_SOURCE_TERM do,
    so for typical setups (hydrodynamics only / wind / cooling without
    flux-coupled gravity) the standalone density flux arrays can be skipped
    and the fused Pallas WENO+divergence path is safe."""
    return bool(getattr(config, "self_gravity", False)) and (
        int(getattr(config, "self_gravity_version", SIMPLE_SOURCE_TERM)) != SIMPLE_SOURCE_TERM
    )


def _hydro_step_rhs(
    current_q,
    dt_tilde,
    *,
    params,
    config,
    registered_variables,
    gamma,
    grid_spacing,
    helper_data,
    density_fluxes_needed: bool,
):
    """RHS for one hydro WENO time-step stage (excluding RK coefficient logic).

    ``dt_tilde`` is the stage-effective step (``k * dt``).  Returns
    ``rhs_q = -dt_tilde * div(F(current_q)) + dt_tilde * S(current_q)``.

    Shared by the SSPRK4 and LSRK4 (low-storage) integrators below; the only
    integrator-specific code is the way ``dt_tilde`` is built and how each
    stage's update accumulates ``rhs_q`` back into the running state.
    """
    dtdx = dt_tilde / grid_spacing
    dtdy = dt_tilde / grid_spacing
    dtdz = dt_tilde / grid_spacing

    # Fused WENO + axis-flux-divergence: each axis is built and consumed one
    # at a time, so the full-state-sized ``dF_x/y/z`` temporaries that used
    # to dominate the peak memory footprint never coexist.  Falls back to the
    # explicit flux + divergence path when (a) Pallas is unavailable /
    # unsupported or (b) a physics module needs the standalone density flux
    # slices.
    use_fused_pallas = (
        _hydro_pallas_flux_supported(current_q, config)
        and not density_fluxes_needed
    )

    if use_fused_pallas:
        # Compute each axis flux with the standard (1-flux-per-cell) WENO
        # kernel, then immediately consume it via a per-axis divergence
        # kernel that accumulates into ``rhs_q`` in place (via the kernel's
        # ``input_output_aliases``).  This keeps WENO compute unchanged
        # relative to the original Pallas path while ensuring all three
        # ``dF`` temporaries never coexist and the rhs lives in a single
        # physical buffer across axes.
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)
        rhs_q = _hydro_flux_div_axis_pallas(dF_x, dtdx, config, axis=0)
        del dF_x

        if config.dimensionality >= 2:
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            rhs_q = _hydro_flux_div_axis_pallas(
                dF_y, dtdy, config, axis=1, rhs_accumulator=rhs_q
            )
            del dF_y

        if config.dimensionality == 3:
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)
            rhs_q = _hydro_flux_div_axis_pallas(
                dF_z, dtdz, config, axis=2, rhs_accumulator=rhs_q
            )
            del dF_z

        density_fluxes = None
    else:
        # Per-axis flux + divergence path.  Accumulate axis-by-axis rather
        # than holding all three flux arrays live simultaneously, so XLA
        # can reuse buffers between axes.
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)
        rhs_q = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        if density_fluxes_needed:
            density_fluxes = [dF_x[registered_variables.density_index]]
        else:
            density_fluxes = None
        del dF_x

        if config.dimensionality >= 2:
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            rhs_q = rhs_q - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            if density_fluxes_needed:
                density_fluxes.append(dF_y[registered_variables.density_index])
            del dF_y

        if config.dimensionality == 3:
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)
            rhs_q = rhs_q - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            if density_fluxes_needed:
                density_fluxes.append(dF_z[registered_variables.density_index])
            del dF_z

        if density_fluxes_needed:
            density_fluxes = tuple(density_fluxes)

    # Add physics source terms
    rhs_q += _physics_sources(
        current_q,
        density_fluxes,
        rhs_q[registered_variables.density_index],  # drho
        dt_tilde,
        gamma,
        config,
        params,
        helper_data,
        registered_variables,
    )

    return rhs_q


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

    Three-register Spiteri-Ruuth scheme: needs ``q0``, ``q_curr`` and
    ``q_final`` simultaneously.  For storage-constrained runs, the
    ``_lsrk4_hydro`` 2-register Carpenter-Kennedy LSRK4 is available below
    via ``time_integrator=RK4_LSRK``.
    """

    # for procceses with similar or smaller time scales as the hydrodynamics,
    # they should be included as source terms in the RK stages, otherwise
    # they could be handled outside

    density_fluxes_needed = _hydro_density_fluxes_needed(config)

    def compute_rhs(current_q, k2_coeff):
        return _hydro_step_rhs(
            current_q,
            k2_coeff * dt,
            params=params,
            config=config,
            registered_variables=registered_variables,
            gamma=gamma,
            grid_spacing=grid_spacing,
            helper_data=helper_data,
            density_fluxes_needed=density_fluxes_needed,
        )

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


# Carpenter & Kennedy (1994) 2N-storage 5-stage 4th-order low-storage RK
# coefficients ("RK4(5)").  See:
#   Carpenter, M.H. & Kennedy, C.A. (1994), "Fourth-order 2N-storage
#   Runge-Kutta schemes", NASA TM-109112.
# The scheme advances ``q`` and one auxiliary register ``dq``:
#     dq^{(0)} = 0
#     for i in 0..4:
#         dq^{(i+1)} = A[i] * dq^{(i)} + dt * L(q^{(i)})
#         q^{(i+1)}  = q^{(i)} + B[i] * dq^{(i+1)}
# with ``A[0] = 0`` so the first stage is a plain forward Euler micro-step.
_LSRK4_A = (
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
)
_LSRK4_B = (
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
)


@partial(jax.jit, static_argnames=["registered_variables", "config"], donate_argnames=["conserved_state"])
def _lsrk4_hydro(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params,
    helper_data,
    config,
    registered_variables: RegisteredVariables,
):
    """Carpenter-Kennedy 2N-storage, 5-stage, 4th-order low-storage RK4.

    The integrator carries two full-state registers (``q`` and ``dq``)
    instead of the three (``q0``, ``q_curr``, ``q_final``) required by the
    SSPRK4 Spiteri-Ruuth scheme above.  That saves one full conserved-state
    buffer at peak, which on the 128^3 Sedov benchmark cuts the per-device
    temp footprint by another ~50 MB on top of the WENO/divergence Pallas
    improvements.

    The trade-off is a smaller linear-stability CFL than SSPRK4 (the user
    should expect roughly half of the 1.5 that SSPRK4 tolerates with the
    5th-order WENO scheme); LSRK4 has no SSP property either, so very strong
    shocks may need a slightly tighter limiter / floor than SSPRK4 to avoid
    sporadic non-monotone overshoots.
    """

    density_fluxes_needed = _hydro_density_fluxes_needed(config)

    a_coeffs = jnp.asarray(_LSRK4_A, dtype=conserved_state.dtype)
    b_coeffs = jnp.asarray(_LSRK4_B, dtype=conserved_state.dtype)

    dtdx = dt / grid_spacing
    dtdy = dt / grid_spacing
    dtdz = dt / grid_spacing

    def stage(stage_idx, carry):
        q, dq = carry

        if config.enforce_positivity:
            q = _enforce_positivity(
                q,
                config,
                gamma,
                params.minimum_density,
                params.minimum_pressure,
                registered_variables,
            )

        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE
            )

        a_coef = a_coeffs[stage_idx]
        b_coef = b_coeffs[stage_idx]

        # Fused path: write the LSRK4 ``dq_new = A[i] * dq + dt * L(q)``
        # update directly into the ``dq`` buffer using the per-axis
        # divergence kernel's ``input_output_aliases``.  The
        # rhs/``L(q)``-sized scratch register is never materialised, which is
        # what gets us below the 3-buffer floor of the explicit
        # rhs-then-update path.
        use_fused_pallas = (
            _hydro_pallas_flux_supported(q, config)
            and not density_fluxes_needed
        )

        if use_fused_pallas:
            dF_x = _weno_flux_x(q, params, config, registered_variables)
            dq = _hydro_flux_div_axis_pallas(
                dF_x, dtdx, config, axis=0,
                rhs_accumulator=dq, scale_in=a_coef,
            )
            del dF_x

            if config.dimensionality >= 2:
                dF_y = _weno_flux_y(q, params, config, registered_variables)
                dq = _hydro_flux_div_axis_pallas(
                    dF_y, dtdy, config, axis=1,
                    rhs_accumulator=dq, scale_in=1.0,
                )
                del dF_y

            if config.dimensionality == 3:
                dF_z = _weno_flux_z(q, params, config, registered_variables)
                dq = _hydro_flux_div_axis_pallas(
                    dF_z, dtdz, config, axis=2,
                    rhs_accumulator=dq, scale_in=1.0,
                )
                del dF_z

            # Physics source terms.  Sedov-style hydro with no active modules
            # makes this a no-op (``_physics_sources`` returns zeros); for
            # active modules the dt-scaled source is added on top of the
            # already-scaled ``A[i] * dq + dt * L(q)`` value in ``dq``.
            sources = _physics_sources(
                q,
                None,
                dq[registered_variables.density_index],
                dt,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
            )
            if sources is not None:
                dq = dq + sources
        else:
            # Fallback: explicit ``rhs = dt * L(q)`` then ``dq = A * dq + rhs``.
            rhs = _hydro_step_rhs(
                q,
                dt,
                params=params,
                config=config,
                registered_variables=registered_variables,
                gamma=gamma,
                grid_spacing=grid_spacing,
                helper_data=helper_data,
                density_fluxes_needed=density_fluxes_needed,
            )
            dq = a_coef * dq + rhs

        q = q + b_coef * dq
        return (q, dq)

    q_final, _ = jax.lax.fori_loop(
        0, 5, stage, (conserved_state, jnp.zeros_like(conserved_state))
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

    return q_final