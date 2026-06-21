"""Positivity-preserving flux limiting for the finite-difference MHD WENO.

In hypersonic / low-beta flows the high-order WENO interface flux can over-deplete
a cell and drive its density to vacuum, after which ``v = momentum / rho`` becomes
non-finite (the diagnosed M~50 adiabatic blow-up). The Hu-Adams-Shu (JCP 2013)
positivity-preserving flux limiter — a Zalesak-style FCT limiter — blends each
interface flux toward the first-order Lax-Friedrichs flux,

    F_hat_{i+1/2} = F^LF_{i+1/2} + theta_{i+1/2} (F^WENO_{i+1/2} - F^LF_{i+1/2}),

choosing ``theta in [0, 1]`` per interface so the density updated with the limited
flux cannot fall below ``minimum_density``. The LF flux is positivity-preserving
under the CFL condition, so ``theta = 0`` is always a safe fallback; where the
WENO flux is already admissible ``theta = 1`` and full high-order accuracy is kept.

This operates on the cell-centred conserved MHD state inside the SSPRK/LSRK
stages, once per axis, and is gated by ``config.positivity_preserving_flux``.
"""

import jax.numpy as jnp

from astronomix._stencil_operations._stencil_operations import _shift
from astronomix._fluid_equations._fluxes_mhd import _mhd_flux_x
from astronomix._fluid_equations._eigen_mhd import _eigen_all_lambdas


def _swap_axis_components(q, registered_variables, axis):
    """Swap the x<->axis momentum and magnetic components so ``axis`` plays the
    role of the x-direction (mirrors the per-axis WENO component swap)."""
    mom = registered_variables.momentum_index
    mag = registered_variables.magnetic_index
    if axis == 1:
        (a, b), (c, d) = (mom.x, mom.y), (mag.x, mag.y)
    else:
        (a, b), (c, d) = (mom.x, mom.z), (mag.x, mag.z)
    qa, qb, qc, qd = q[a], q[b], q[c], q[d]
    q = q.at[a].set(qb).at[b].set(qa)
    q = q.at[c].set(qd).at[d].set(qc)
    return q


def _physical_flux_dir(conserved_state, axis, rhomin, pgmin, gamma, config,
                       registered_variables, internal_energy_density):
    """Cell-centred MHD physical flux in direction ``axis`` (original orientation)."""
    if axis == 0:
        return _mhd_flux_x(
            conserved_state, rhomin, pgmin, gamma, config, registered_variables,
            internal_energy_density=internal_energy_density,
        )
    perm = (0, 2, 1, 3) if axis == 1 else (0, 3, 2, 1)   # self-inverse
    gperm = (1, 0, 2) if axis == 1 else (2, 1, 0)
    qd = _swap_axis_components(jnp.transpose(conserved_state, perm), registered_variables, axis)
    gd = None if internal_energy_density is None else jnp.transpose(internal_energy_density, gperm)
    Fd = _mhd_flux_x(qd, rhomin, pgmin, gamma, config, registered_variables, internal_energy_density=gd)
    Fd = _swap_axis_components(Fd, registered_variables, axis)
    return jnp.transpose(Fd, perm)


def pp_limit_flux_axis(dF_weno, conserved_state, params, config,
                       registered_variables, dtdx, axis, internal_energy_density=None):
    """Return the positivity-preserving-limited interface flux for one axis.

    ``dF_weno[..., i, ...]`` is the WENO flux at interface ``i+1/2`` (the
    integrator's convention: ``rhs_i = -dtdx (dF[i] - dF[i-1])``).
    """
    va = axis + 1                       # flux axis in a (nvars, nx, ny, nz) array
    fa = axis                           # same flux axis in a scalar (nx, ny, nz) field
    rhomin = params.minimum_density
    pgmin = params.minimum_pressure
    gamma = params.gamma
    di = registered_variables.density_index

    # first-order Lax-Friedrichs interface flux F^LF_{i+1/2}
    F_cell = _physical_flux_dir(
        conserved_state, axis, rhomin, pgmin, gamma, config,
        registered_variables, internal_energy_density,
    )
    lambdas = _eigen_all_lambdas(
        conserved_state, rhomin, pgmin, gamma, registered_variables,
        internal_energy_density=internal_energy_density,
    )
    alpha = jnp.max(jnp.abs(lambdas), axis=0)            # per-cell max wave speed (scalar field)
    alpha_face = jnp.maximum(alpha, _shift(alpha, -1, axis=fa))
    U_R = _shift(conserved_state, -1, axis=va)
    F_R = _shift(F_cell, -1, axis=va)
    F_LF = 0.5 * (F_cell + F_R) - 0.5 * alpha_face[None, ...] * (U_R - conserved_state)

    # antidiffusive flux (high-order minus low-order)
    A = dF_weno - F_LF
    A_rho = A[di]                                        # scalar field

    # density updated with the (positivity-preserving) LF flux only
    F_LF_rho = F_LF[di]                                  # scalar field
    rho_LF_new = conserved_state[di] - dtdx * (F_LF_rho - _shift(F_LF_rho, 1, axis=fa))

    # Zalesak lower-bound limiter: cell i's admissible antidiffusive *outflow*
    # is the total antidiffusive mass it may lose before hitting the floor.
    P_minus = dtdx * (jnp.maximum(0.0, A_rho) + jnp.maximum(0.0, -_shift(A_rho, 1, axis=fa)))
    Q_minus = jnp.maximum(rho_LF_new - rhomin, 0.0)
    R_minus = jnp.where(P_minus > 1e-30, jnp.minimum(1.0, Q_minus / P_minus), 1.0)

    # theta at interface i+1/2: outflow direction selects which cell's budget
    # binds (A_rho >= 0 drains cell i; A_rho < 0 drains cell i+1).
    theta = jnp.where(A_rho >= 0.0, R_minus, _shift(R_minus, -1, axis=fa))

    return F_LF + theta[None, ...] * A
