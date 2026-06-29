"""Deep-void first-order flux blending (FOFC-style robustness fix-up).

High-Mach turbulence carves deep voids whose density sits pinned at the floor.
At those voids the high-order WENO *characteristic* reconstruction overshoots
(the local Mach number is enormous and the characteristic fields are nearly
degenerate for an isothermal gas), which drives a fast momentum/velocity
blow-up that per-substage momentum-resting alone does not remove — the genuine
scheme-level marginal instability documented in
``tests/turbulence/HANDOFF_deepvoid_instability.md``.

The cure here is the standard one for high-order schemes at strong
rarefactions: blend the high-order interface flux toward a first-order *local
Lax-Friedrichs* (Rusanov) flux, but only in the immediate neighbourhood of a
void.  The blend weight ``theta`` ramps smoothly from 1 at the density floor to
0 at ``blend_factor * minimum_density``, so the scheme stays fully 5th-order
everywhere except where it is provably unsafe.  LLF is monotone/positivity-
friendly, so the blended flux cannot overshoot.

Hydro and (isothermal/ideal) MHD are both supported.  For MHD the Rusanov wave
speed is the fast magnetosonic speed (same formula as the CFL estimator), and
the blend is applied to the *full* interface flux before the transverse
magnetic-flux slices are handed to Constrained Transport.  This is CT-safe:
CT builds single-valued edge EMFs from whatever face fluxes it is given, so
div(B)=0 is preserved to machine precision regardless of the flux — blending
toward LLF merely adds a localised magnetic diffusivity at the void.  (The
normal-B flux is overwritten by CT's cell-centre update anyway.)

This operates purely on the assembled interface-flux array (the output of the
WENO kernel) — it does not touch the Pallas WENO kernel itself; it is a small
native-JAX post-process applied before the flux divergence.
"""

import jax.numpy as jnp

from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.option_classes.simulation_config import IDEAL_GAS


def _deepvoid_llf_blend(
    dF_weno,
    conserved_state,
    axis,
    params,
    config,
    registered_variables,
):
    """Blend the WENO interface flux ``dF_weno`` toward a first-order LLF flux
    near the density floor along ``axis`` (0/1/2).

    ``dF_weno[..., i]`` is the WENO flux at interface ``i+1/2``; the LLF flux is
    built from the same convention (cells ``i`` and ``i+1``) so the blended
    array feeds the existing ``-dt/dx * (F_{i+1/2} - F_{i-1/2})`` divergence
    unchanged.  Works for hydro and MHD, isothermal and ideal gas.
    """
    ndim = config.dimensionality
    di = registered_variables.density_index
    rhomin = params.minimum_density
    is_ideal = (config.equation_of_state == IDEAL_GAS)
    is_mhd = bool(config.mhd)

    if ndim == 1:
        mom_all = [registered_variables.velocity_index]
    else:
        mom_all = [
            registered_variables.velocity_index.x,
            registered_variables.velocity_index.y,
            registered_variables.velocity_index.z,
        ][:ndim]
    md = mom_all[axis]
    mom_others = [m for i, m in enumerate(mom_all) if i != axis]

    if is_mhd:
        B_all = [
            registered_variables.magnetic_index.x,
            registered_variables.magnetic_index.y,
            registered_variables.magnetic_index.z,
        ]
        Bd = B_all[axis]
        # transverse B components paired with their transverse momentum component
        B_others = [B_all[i] for i in range(3) if i != axis]

    def R(a):
        # right neighbour (i+1) for a single-variable (spatial-only) array
        return _shift(a, -1, axis=axis)

    def R_state(a):
        # right neighbour (i+1) for the full (n_vars, spatial...) state array
        return _shift(a, -1, axis=axis + 1)

    rhoL = jnp.maximum(conserved_state[di], rhomin)
    rhoR = jnp.maximum(R(conserved_state[di]), rhomin)
    mdL = conserved_state[md]
    mdR = R(conserved_state[md])
    vdL = mdL / rhoL
    vdR = mdR / rhoR

    # transverse velocities (needed by both momentum and induction fluxes)
    veL = [conserved_state[m] / rhoL for m in mom_others]
    veR = [R(conserved_state[m]) / rhoR for m in mom_others]

    if is_mhd:
        BdL = conserved_state[Bd]
        BdR = R(conserved_state[Bd])
        BeL = [conserved_state[b] for b in B_others]
        BeR = [R(conserved_state[b]) for b in B_others]
        b2L = BdL * BdL
        b2R = BdR * BdR
        for bl, br in zip(BeL, BeR):
            b2L = b2L + bl * bl
            b2R = b2R + br * br

    # gas pressure and sound speed^2
    if is_ideal:
        gamma = params.gamma
        EL = conserved_state[registered_variables.energy_index]
        ER = R(EL)
        keL = 0.5 * (mdL * mdL) / rhoL
        keR = 0.5 * (mdR * mdR) / rhoR
        for ve in veL:
            keL = keL + 0.5 * rhoL * ve * ve
        for ve in veR:
            keR = keR + 0.5 * rhoR * ve * ve
        if is_mhd:
            pL = jnp.maximum((gamma - 1.0) * (EL - keL - 0.5 * b2L), params.minimum_pressure)
            pR = jnp.maximum((gamma - 1.0) * (ER - keR - 0.5 * b2R), params.minimum_pressure)
        else:
            pL = jnp.maximum((gamma - 1.0) * (EL - keL), params.minimum_pressure)
            pR = jnp.maximum((gamma - 1.0) * (ER - keR), params.minimum_pressure)
        cs2L = gamma * pL / rhoL
        cs2R = gamma * pR / rhoR
    else:
        cs = params.isothermal_sound_speed
        cs2L = cs * cs
        cs2R = cs * cs
        pL = cs2L * rhoL
        pR = cs2R * rhoR

    # signal speed: |v_d| + fast magnetosonic speed (hydro: + sound speed)
    if is_mhd:
        def cfast(b2, rho, Bn, cs2):
            b2_over_rho = b2 / rho
            bn2_over_rho = (Bn * Bn) / rho
            disc = jnp.maximum((b2_over_rho + cs2) ** 2 - 4.0 * bn2_over_rho * cs2, 0.0)
            return jnp.sqrt(jnp.maximum(0.5 * (b2_over_rho + cs2 + jnp.sqrt(disc)), 0.0))
        cL = cfast(b2L, rhoL, BdL, cs2L)
        cR = cfast(b2R, rhoR, BdR, cs2R)
    else:
        cL = jnp.sqrt(cs2L)
        cR = jnp.sqrt(cs2R)

    alpha = jnp.maximum(jnp.abs(vdL) + cL, jnp.abs(vdR) + cR)

    qR = R_state(conserved_state)
    FL = jnp.zeros_like(conserved_state)
    FR = jnp.zeros_like(conserved_state)

    # density flux
    FL = FL.at[di].set(mdL)
    FR = FR.at[di].set(mdR)

    # normal momentum flux: m_d v_d + p (+ B^2/2 - B_d^2 for MHD)
    fmdL = mdL * vdL + pL
    fmdR = mdR * vdR + pR
    if is_mhd:
        fmdL = fmdL + 0.5 * b2L - BdL * BdL
        fmdR = fmdR + 0.5 * b2R - BdR * BdR
    FL = FL.at[md].set(fmdL)
    FR = FR.at[md].set(fmdR)

    # transverse momentum flux: m_d v_e (- B_d B_e for MHD)
    for k, m in enumerate(mom_others):
        feL = mdL * veL[k]
        feR = mdR * veR[k]
        if is_mhd:
            feL = feL - BdL * BeL[k]
            feR = feR - BdR * BeR[k]
        FL = FL.at[m].set(feL)
        FR = FR.at[m].set(feR)

    if is_mhd:
        # normal B flux is zero (CT overwrites the cell-centre normal field);
        # transverse induction flux: B_e v_d - B_d v_e
        FL = FL.at[Bd].set(jnp.zeros_like(BdL))
        FR = FR.at[Bd].set(jnp.zeros_like(BdR))
        for k, b in enumerate(B_others):
            FL = FL.at[b].set(BeL[k] * vdL - BdL * veL[k])
            FR = FR.at[b].set(BeR[k] * vdR - BdR * veR[k])

    if is_ideal:
        ei = registered_variables.energy_index
        if is_mhd:
            vdotBL = vdL * BdL
            vdotBR = vdR * BdR
            for k in range(len(mom_others)):
                vdotBL = vdotBL + veL[k] * BeL[k]
                vdotBR = vdotBR + veR[k] * BeR[k]
            FL = FL.at[ei].set((EL + pL + 0.5 * b2L) * vdL - BdL * vdotBL)
            FR = FR.at[ei].set((ER + pR + 0.5 * b2R) * vdR - BdR * vdotBR)
        else:
            FL = FL.at[ei].set((EL + pL) * vdL)
            FR = FR.at[ei].set((ER + pR) * vdR)

    F_llf = 0.5 * (FL + FR) - 0.5 * alpha * (qR - conserved_state)

    blend_thr = config.positivity_deepvoid_blend_factor * rhomin
    rho_face = jnp.minimum(rhoL, rhoR)
    theta = jnp.clip((blend_thr - rho_face) / (blend_thr - rhomin), 0.0, 1.0)
    theta = theta[None, ...]
    return dF_weno * (1.0 - theta) + F_llf * theta
