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
from astronomix._finite_difference._time_integrators._ssprk_pallas import (
    _div_axis_pallas_shape_ok,
    _hydro_flux_div_axis_pallas,
)
from astronomix._pallas_helpers import _backend_is_pallas, pl

from astronomix._finite_difference._magnetic_update._constrained_transport import (
    _constrained_transport_rhs_from_slices,
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

        # Axis-incremental flow: build each axis's full dF, extract the
        # two magnetic-flux slices CT needs (plus the density-flux slice
        # for any physics modules that consume it), consume dF for the
        # divergence step, then free dF.  CT runs at the end on the six
        # small single-channel slices instead of the three full 8-var dF
        # arrays — saves ~7/8 × 3 = 2.6× state-shape buffers at peak.
        my = registered_variables.magnetic_index.y
        mz = registered_variables.magnetic_index.z
        di = registered_variables.density_index

        # x-axis
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)
        By_flux_x = dF_x[my]
        Bz_flux_x = dF_x[mz]
        density_flux_x = dF_x[di]
        if use_pallas_div:
            rhs_q = _hydro_flux_div_axis_pallas(dF_x, dtdx, config, axis=0)
        else:
            rhs_q = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        del dF_x

        # y-axis
        if config.dimensionality >= 2:
            mx = registered_variables.magnetic_index.x
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            Bx_flux_y = dF_y[mx]
            Bz_flux_y = dF_y[mz]
            density_flux_y = dF_y[di]
            if use_pallas_div:
                rhs_q = _hydro_flux_div_axis_pallas(
                    dF_y, dtdy, config, axis=1, rhs_accumulator=rhs_q
                )
            else:
                rhs_q = rhs_q - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            del dF_y
        else:
            Bx_flux_y = 0.0
            Bz_flux_y = 0.0

        # z-axis
        if config.dimensionality == 3:
            mx = registered_variables.magnetic_index.x
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)
            Bx_flux_z = dF_z[mx]
            By_flux_z = dF_z[my]
            density_flux_z = dF_z[di]
            if use_pallas_div:
                rhs_q = _hydro_flux_div_axis_pallas(
                    dF_z, dtdz, config, axis=2, rhs_accumulator=rhs_q
                )
            else:
                rhs_q = rhs_q - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            del dF_z
        else:
            Bx_flux_z = 0.0
            By_flux_z = 0.0

        # CT now runs on the six single-channel B-flux slices only — the
        # three 8-var dF arrays have all been freed by this point.
        rhs_bx, rhs_by, rhs_bz = _constrained_transport_rhs_from_slices(
            current_q,
            By_flux_x, Bz_flux_x,
            Bx_flux_y, Bz_flux_y,
            Bx_flux_z, By_flux_z,
            dtdx, dtdy, dtdz,
            config, registered_variables,
        )

        if config.dimensionality == 1:
            density_fluxes = (density_flux_x,)
        elif config.dimensionality == 2:
            density_fluxes = (density_flux_x, density_flux_y)
        else:
            density_fluxes = (density_flux_x, density_flux_y, density_flux_z)


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
    return config.gravity and (
        config.self_gravity_version != SIMPLE_SOURCE_TERM
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


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
    donate_argnames=["conserved_state", "bx_interface", "by_interface", "bz_interface"],
)
def _lsrk4_with_ct(
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
    """Carpenter-Kennedy 2N-storage 5-stage 4th-order LSRK4 for MHD-CT.

    Mirrors ``_lsrk4_hydro`` but carries the four MHD register pairs that
    ``_ssprk4_with_ct``'s Spiteri-Ruuth scheme used as three-register triples:

      * ``(q, dq)`` for the conserved state (8 vars),
      * ``(bx, dbx)``, ``(by, dby)``, ``(bz, dbz)`` for the three interface
        magnetic-field components.

    Compared to the SSPRK4 carry ``(q0, q_curr, q_final)`` plus the three
    ``(bx0, bx_curr, bx_final)`` triples this saves one full conserved
    register plus three interface-B registers.

    Trade-off: linear-stability CFL drops from SSPRK4's ~1.5 to roughly 1.4
    and LSRK4 has no SSP property — same caveats as ``_lsrk4_hydro``.
    Selected via ``config.time_integrator == RK4_LSRK``.
    """

    use_pallas_div = (
        _backend_is_pallas(config) and pl is not None
        and _div_axis_pallas_shape_ok(conserved_state, config)
    )

    a_coeffs = jnp.asarray(_LSRK4_A, dtype=conserved_state.dtype)
    b_coeffs = jnp.asarray(_LSRK4_B, dtype=conserved_state.dtype)

    dtdx = dt / grid_spacing
    dtdy = dt / grid_spacing
    dtdz = dt / grid_spacing

    def compute_lqs(current_q, bx, by, bz, dq, a_coef):
        """Compute ``dq_new = a_coef * dq + dt * L_q`` (in-place via the
        Pallas div-axis accumulator when available) and the three
        interface-B ``dt * L_b{x,y,z}`` increments.

        The fused conserved-state path matches ``_lsrk4_hydro``: each
        axis's divergence kernel folds ``a_coef * dq + (-dt/dx) * div``
        directly into the ``dq`` register, so the LSRK4 update never
        materialises a separate ``rhs_q``.  When Pallas is unavailable
        we fall back to the explicit
        ``rhs_q`` → ``dq = a_coef * dq + rhs_q`` pattern.
        """
        current_q = update_cell_center_fields(
            current_q, bx, by, bz, config, registered_variables
        )

        # Axis-incremental flow — see the matching SSPRK4-with-CT path
        # above for the rationale.  Each axis's full dF is built, the
        # two magnetic-flux slices CT needs are extracted, dF is consumed
        # for the divergence step (folding ``a_coef * dq`` in for the
        # first axis), then freed.  CT runs on the six small slices only.
        my = registered_variables.magnetic_index.y
        mz = registered_variables.magnetic_index.z
        di = registered_variables.density_index

        # x-axis: fold the LSRK4 ``a_coef * dq + ...`` step into the
        # first axis's div kernel via ``scale_in`` so ``rhs_q`` is never
        # materialised; subsequent axes accumulate (scale_in = 1.0).  The
        # native fallback path keeps the explicit ``rhs_q`` register.
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)
        By_flux_x = dF_x[my]
        Bz_flux_x = dF_x[mz]
        density_flux_x = dF_x[di]
        if use_pallas_div:
            dq = _hydro_flux_div_axis_pallas(
                dF_x, dtdx, config, axis=0,
                rhs_accumulator=dq, scale_in=a_coef,
            )
            rhs_q_for_phys = None
        else:
            rhs_q_for_phys = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        del dF_x

        if config.dimensionality >= 2:
            mx = registered_variables.magnetic_index.x
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            Bx_flux_y = dF_y[mx]
            Bz_flux_y = dF_y[mz]
            density_flux_y = dF_y[di]
            if use_pallas_div:
                dq = _hydro_flux_div_axis_pallas(
                    dF_y, dtdy, config, axis=1, rhs_accumulator=dq,
                )
            else:
                rhs_q_for_phys = rhs_q_for_phys - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            del dF_y
        else:
            Bx_flux_y = 0.0
            Bz_flux_y = 0.0

        if config.dimensionality == 3:
            mx = registered_variables.magnetic_index.x
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)
            Bx_flux_z = dF_z[mx]
            By_flux_z = dF_z[my]
            density_flux_z = dF_z[di]
            if use_pallas_div:
                dq = _hydro_flux_div_axis_pallas(
                    dF_z, dtdz, config, axis=2, rhs_accumulator=dq,
                )
            else:
                rhs_q_for_phys = rhs_q_for_phys - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            del dF_z
        else:
            Bx_flux_z = 0.0
            By_flux_z = 0.0

        rhs_bx, rhs_by, rhs_bz = _constrained_transport_rhs_from_slices(
            current_q,
            By_flux_x, Bz_flux_x,
            Bx_flux_y, Bz_flux_y,
            Bx_flux_z, By_flux_z,
            dtdx, dtdy, dtdz,
            config, registered_variables,
        )

        if config.dimensionality == 1:
            density_fluxes = (density_flux_x,)
        elif config.dimensionality == 2:
            density_fluxes = (density_flux_x, density_flux_y)
        else:
            density_fluxes = (density_flux_x, density_flux_y, density_flux_z)

        # Physics source terms.  On the Pallas-fused path the divergence
        # has already been folded into ``dq``; we add ``dt * S`` on top.
        # On the native fallback we still have a standalone ``rhs_q_for_phys``
        # and fold the full LSRK4 update at the end.
        if use_pallas_div:
            sources = _physics_sources(
                current_q,
                density_fluxes,
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
            rhs_q_for_phys += _physics_sources(
                current_q,
                density_fluxes,
                rhs_q_for_phys[registered_variables.density_index],
                dt,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
            )
            dq = a_coef * dq + rhs_q_for_phys

        return dq, rhs_bx, rhs_by, rhs_bz

    def stage(stage_idx, carry):
        q, dq, bx, dbx, by, dby, bz, dbz = carry

        if config.enforce_positivity:
            q = _enforce_positivity(
                q, config, gamma,
                params.minimum_density, params.minimum_pressure,
                registered_variables,
            )

        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE,
            )
            b_curr = _boundary_handler(
                jnp.stack([bx, by, bz], axis=0),
                config, registered_variables, params, MAGNETIC_FIELD_ONLY,
            )
            bx, by, bz = b_curr[0], b_curr[1], b_curr[2]

        a_coef = a_coeffs[stage_idx]
        b_coef = b_coeffs[stage_idx]

        # ``compute_lqs`` returns the new ``dq`` already in LSRK4 form
        # (``a_coef * dq_old + dt * L_q``) so no separate
        # ``rhs_q + dq = a_coef * dq + rhs_q`` step is needed for the
        # conserved-state register.  The interface-B deltas are small so
        # they stay on the explicit LSRK4 update path.
        dq, rhs_bx, rhs_by, rhs_bz = compute_lqs(q, bx, by, bz, dq, a_coef)

        dbx = a_coef * dbx + rhs_bx
        dby = a_coef * dby + rhs_by
        dbz = a_coef * dbz + rhs_bz

        q = q + b_coef * dq
        bx = bx + b_coef * dbx
        by = by + b_coef * dby
        bz = bz + b_coef * dbz

        return (q, dq, bx, dbx, by, dby, bz, dbz)

    init = (
        conserved_state,
        jnp.zeros_like(conserved_state),
        bx_interface, jnp.zeros_like(bx_interface),
        by_interface, jnp.zeros_like(by_interface),
        bz_interface, jnp.zeros_like(bz_interface),
    )

    q_final, _, bx_final, _, by_final, _, bz_final, _ = jax.lax.fori_loop(
        0, 5, stage, init,
    )

    q_final = update_cell_center_fields(
        q_final, bx_final, by_final, bz_final, config, registered_variables,
    )

    if config.enforce_positivity:
        q_final = _enforce_positivity(
            q_final, config, gamma,
            params.minimum_density, params.minimum_pressure,
            registered_variables,
        )

    return q_final, bx_final, by_final, bz_final
