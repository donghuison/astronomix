"""
Here we calculate weighted essentially non-oscillatory 
(WENO) fluxes for the MHD equations.

The idea of WENO is to find interface fluxes by interpolating
the cell centered fluxes using several stencils, and then
weighting the stencils based on their smoothness.

The reconstruction is done in characteristic variables to
better capture the underlying wave structure. At each interface,
we compute the eigenstructure (evaluated at the average of the
left and right states), and project all stencil
characteristic space.

Consider the interface at i + 1/2. Our vector of conserved
variables is q = (rho, rho*v_x, rho*v_y, rho*v_z, B_x, B_y, B_z, E)^T
with N_vars = 8 variables. In the eigenstructure of the MHD equations,
we have N_char = 7 characteristic waves.

We calculate the flux as follows:

1. We retrieve the eigenstructure given by the right 
   and left eigenvector matrices R_{i+1/2} \in R^{N_vars x N_char} 
   and L_{i+1/2} \in R^{N_char x N_vars}, as well 
   as the eigenvalues lambda at 
   q_{i+1/2} ~ 0.5 * (q_i + q_{i+1}).

2. In the stencil m = i - 2, ..., i + 2, we project the fluxes
   F_m and conserved variables q_m into characteristic space:
   F_s_m = L^s_{i+1/2} * F_m, q_s_m = L^s_{i+1/2} * q_m, where L^s_{i+1/2}
   is the s-th row of L so F_s_m and q_s_m are scalar fields. All
   fluxes and conserved variables in the stencil m = i - 2, ..., i + 2
   are projected using the same L^s_{i+1/2} at the interface i + 1/2.

3. We compute the differences ΔF_s_{m+1/2} = F_s_{m+1} - F_s_m and
   Δq_s_{m+1/2} = q_s_{m+1} - q_s_m for m = i - 2, ..., i + 1.

4. We use local Lax-Friedrichs flux splitting to split the fluxes
   into F_s^+ and F_s^- such that \partial_u F_s^+ only has non-negative
   eigenvalues, and \partial_u F_s^- only has non-positive eigenvalues.
   Both can then be properly upwinded with skewed stencils (for F_s^+ we
   use a left-biased stencil, for F_s^- we use a right-biased stencil, 
   see step 5).

   ΔF_s^+_{m+1/2} = 0.5 * (ΔF_s_{m+1/2} + alpha^s * Δq_s_{m+1/2}), m = i - 2, ..., i + 1
   ΔF_s^-_{m+1/2} = 0.5 * (ΔF_s_{m+1/2} - alpha^s * Δq_s_{m+1/2}), m = i - 1, ..., i + 2

   where alpha^s = max(|lambda^s_m|) over the 
   stencil m = i - 2, ..., i + 3.

5. We can compactly write the WENO flux reconstruction as:
   
   F_{i+1/2} = 1/12 * (-F_{i-1} + 7*F_i + 7*F_{i+1} - F_{i+2})
                +sum_{s = 1}^{N_char} [
                    -\phi(ΔF_s^+_{i-3/2}, ΔF_s^+_{i-1/2}, ΔF_s^+_{i+1/2}, ΔF_s^+_{i+3/2})
                    +\phi(ΔF_s^-_{i+5/2}, ΔF_s^-_{i+3/2}, ΔF_s^-_{i+1/2}, ΔF_s^-_{i-1/2})
                ] * R^s_{i+1/2}
    
    where R^s_{i+1/2} is the s-th column of R at the interface i + 1/2,
    and \phi is the WENO interpolant function given by:

    \phi(a, b, c, d) = 1/3 ω_0 (a - 2b + c) + 1/6 (ω_2 - 1/2) (b - 2c + d)

    with weight functions:
    
    ω_0 = α_0 / (α_0 + α_1 + α_2)
    ω_2 = α_2 / (α_0 + α_1 + α_2)

    α_0 = 1 / (ε + IS_0)^2
    α_1 = 6 / (ε + IS_1)^2
    α_2 = 3 / (ε + IS_2)^2

    and smoothness indicators:

    IS_0 = 13 (a - b)^2 + 3 (a - 3b)^2
    IS_1 = 13 (b - c)^2 + 3 (b + c)^2
    IS_2 = 13 (c - d)^2 + 3 (3c - d)^2

    ε is a small parameter to avoid 
    division by zero, here taken as 1e-8.

NOTE: I have seen formulations where the first part of the flux
is also calculated in characteristic space and then transformed back,
but I found that at single precision this introduces small perturbations
as RL is not exactly the identity matrix by finite precision effects.

For literature references, see:

 - High Order ENO and WENO Schemes for Computational Fluid Dynamics by Chi-Wang Shu (1997)
   (https://doi.org/10.1007/978-3-662-03882-6_5)

Concretely we implement the 5th-order WENO scheme as described in 

- HOW-MHD: A High-Order WENO-Based Magnetohydrodynamic Code with a High-Order 
  Constrained Transport Algorithm for Astrophysical Applications by Seo & Ryu 2023
  (https://arxiv.org/abs/2304.04360)
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Union

from jax import checkpoint

try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - optional backend
    pl = None

try:
    from jax.experimental.pallas import triton as pltriton
except Exception:  # pragma: no cover - optional backend
    pltriton = None

from astronomix._finite_difference._fluid_equations._eigen_hydro import _eigen_L_row_hydro, _eigen_R_col_hydro, _eigen_lambdas_hydro
from astronomix._finite_difference._fluid_equations._eigen_hydro_iso import _eigen_L_row_hydro_iso, _eigen_R_col_hydro_iso, _eigen_lambdas_hydro_iso
from astronomix._finite_difference._fluid_equations._eigen_mhd import _eigen_L_row, _eigen_R_col, _eigen_lambdas
from astronomix._finite_difference._fluid_equations._eigen_mhd_iso import _eigen_L_row_iso, _eigen_R_col_iso, _eigen_lambdas_iso
from astronomix._finite_difference._fluid_equations._fluxes import _euler_flux_isothermal_x, _mhd_flux_isothermal_x, _mhd_flux_x
from astronomix._fluid_equations._equations import primitive_state_from_conserved
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.option_classes.simulation_config import IDEAL_GAS, ISOTHERMAL, PALLAS, SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_x_native(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    WENO flux reconstruction.
    """

    # if jax.config.jax_enable_x64:
    #     epsilon = 1e-8
    # else:
    #     epsilon = 1e-6
    epsilon = 1e-7

    # only used in the IDEAL_GAS case
    rhomin = params.minimum_density
    pgmin = params.minimum_pressure
    gamma = params.gamma

    # only used in the ISOTHERMAL case
    isothermal_sound_speed = params.isothermal_sound_speed

    if config.equation_of_state == IDEAL_GAS:
        # retrieve the center fluxes
        if config.mhd:
            F = _mhd_flux_x(
                conserved_state,
                rhomin,
                pgmin,
                gamma,
                config,
                registered_variables
            )
        else:
            F =  _euler_flux(
                primitive_state_from_conserved(
                    conserved_state, gamma, config, registered_variables
                ),
                gamma, config, registered_variables, 1
            )
    elif config.equation_of_state == ISOTHERMAL:
        if config.mhd:
            F = _mhd_flux_isothermal_x(
                conserved_state,
                rhomin,
                isothermal_sound_speed,
                config,
                registered_variables,
            )
        else:
            F = _euler_flux_isothermal_x(
                conserved_state,
                rhomin,
                isothermal_sound_speed,
                config,
                registered_variables,
            )

        
    # with this we can already compute the first part of the flux
    F_interface = 1/12 * (
        -_shift(F, 1, axis=1) + 7 * F + 7 * _shift(F, -1, axis=1) - _shift(F, -2, axis=1)
    )

    def mode_flux(mode, F_current):

        # get eigenstructure for this mode
        if config.equation_of_state == IDEAL_GAS:
            if config.mhd:
                lambdas_center = _eigen_lambdas(conserved_state, rhomin, pgmin, gamma, registered_variables, mode)
                L_row = _eigen_L_row(conserved_state, rhomin, pgmin, gamma, registered_variables, mode)
            else:
                lambdas_center = _eigen_lambdas_hydro(conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode)
                L_row = _eigen_L_row_hydro(conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode)
        elif config.equation_of_state == ISOTHERMAL:
            if config.mhd:
                lambdas_center = _eigen_lambdas_iso(conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode)
                L_row = _eigen_L_row_iso(conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode)
            else:
                lambdas_center = _eigen_lambdas_hydro_iso(conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode)
                L_row = _eigen_L_row_hydro_iso(conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode)

        F0 = _shift(F,  2, axis=1)   # shape (N_vars, Nx, Ny, Nz) — i-2 at target i
        F1 = _shift(F,  1, axis=1)   # i-1
        F2 = F                         # i
        F3 = _shift(F, -1, axis=1)   # i+1
        F4 = _shift(F, -2, axis=1)   # i+2
        F5 = _shift(F, -3, axis=1)   # i+3

        if config.dimensionality == 3:
            s0 = jnp.einsum('nxyz,nxyz->xyz', L_row, F0)
            s1 = jnp.einsum('nxyz,nxyz->xyz', L_row, F1)
            s2 = jnp.einsum('nxyz,nxyz->xyz', L_row, F2)
            s3 = jnp.einsum('nxyz,nxyz->xyz', L_row, F3)
            s4 = jnp.einsum('nxyz,nxyz->xyz', L_row, F4)
            s5 = jnp.einsum('nxyz,nxyz->xyz', L_row, F5)

            q0 = jnp.einsum('nxyz,nxyz->xyz', L_row, _shift(conserved_state, 2, axis=1))
            q1 = jnp.einsum('nxyz,nxyz->xyz', L_row, _shift(conserved_state, 1, axis=1))
            q2 = jnp.einsum('nxyz,nxyz->xyz', L_row, conserved_state)
            q3 = jnp.einsum('nxyz,nxyz->xyz', L_row, _shift(conserved_state, -1, axis=1))
            q4 = jnp.einsum('nxyz,nxyz->xyz', L_row, _shift(conserved_state, -2, axis=1))
            q5 = jnp.einsum('nxyz,nxyz->xyz', L_row, _shift(conserved_state, -3, axis=1))
        elif config.dimensionality == 2:
            s0 = jnp.einsum('nxy,nxy->xy', L_row, F0)
            s1 = jnp.einsum('nxy,nxy->xy', L_row, F1)
            s2 = jnp.einsum('nxy,nxy->xy', L_row, F2)
            s3 = jnp.einsum('nxy,nxy->xy', L_row, F3)
            s4 = jnp.einsum('nxy,nxy->xy', L_row, F4)
            s5 = jnp.einsum('nxy,nxy->xy', L_row, F5)

            q0 = jnp.einsum('nxy,nxy->xy', L_row, _shift(conserved_state, 2, axis=1))
            q1 = jnp.einsum('nxy,nxy->xy', L_row, _shift(conserved_state, 1, axis=1))
            q2 = jnp.einsum('nxy,nxy->xy', L_row, conserved_state)
            q3 = jnp.einsum('nxy,nxy->xy', L_row, _shift(conserved_state, -1, axis=1))
            q4 = jnp.einsum('nxy,nxy->xy', L_row, _shift(conserved_state, -2, axis=1))
            q5 = jnp.einsum('nxy,nxy->xy', L_row, _shift(conserved_state, -3, axis=1))
        else:
            s0 = jnp.einsum('nx,nx->x', L_row, F0)
            s1 = jnp.einsum('nx,nx->x', L_row, F1)
            s2 = jnp.einsum('nx,nx->x', L_row, F2)
            s3 = jnp.einsum('nx,nx->x', L_row, F3)
            s4 = jnp.einsum('nx,nx->x', L_row, F4)
            s5 = jnp.einsum('nx,nx->x', L_row, F5)

            q0 = jnp.einsum('nx,nx->x', L_row, _shift(conserved_state, 2, axis=1))
            q1 = jnp.einsum('nx,nx->x', L_row, _shift(conserved_state, 1, axis=1))
            q2 = jnp.einsum('nx,nx->x', L_row, conserved_state)
            q3 = jnp.einsum('nx,nx->x', L_row, _shift(conserved_state, -1, axis=1))
            q4 = jnp.einsum('nx,nx->x', L_row, _shift(conserved_state, -2, axis=1))
            q5 = jnp.einsum('nx,nx->x', L_row, _shift(conserved_state, -3, axis=1))

        # dFsk identical to original: d0 = s1 - s0, d1 = s2 - s1, ...
        d0 = s1 - s0
        d1 = s2 - s1
        d2 = s3 - s2
        d3 = s4 - s3
        d4 = s5 - s4

        dq0 = q1 - q0
        dq1 = q2 - q1
        dq2 = q3 - q2
        dq3 = q4 - q3
        dq4 = q5 - q4

        # compute amx over the same stencil (take abs then max over the six entries)
        lam0 = _shift(lambdas_center,  2, axis=0)
        lam1 = _shift(lambdas_center,  1, axis=0)
        lam2 = lambdas_center
        lam3 = _shift(lambdas_center, -1, axis=0)
        lam4 = _shift(lambdas_center, -2, axis=0)
        lam5 = _shift(lambdas_center, -3, axis=0)
        lam_stack = jnp.stack([lam0, lam1, lam2, lam3, lam4, lam5], axis=0)
        amx = jnp.max(jnp.abs(lam_stack), axis=0)

        # Now use the exact same definitions as original for aterm/bterm/cterm/dterm
        aterm_p = 0.5 * (d0 + amx * dq0)
        bterm_p = 0.5 * (d1 + amx * dq1)
        cterm_p = 0.5 * (d2 + amx * dq2)
        dterm_p = 0.5 * (d3 + amx * dq3)

        IS0_p = 13.0 * (aterm_p - bterm_p)**2 + 3.0 * (aterm_p - 3.0*bterm_p)**2
        IS1_p = 13.0 * (bterm_p - cterm_p)**2 + 3.0 * (bterm_p + cterm_p)**2
        IS2_p = 13.0 * (cterm_p - dterm_p)**2 + 3.0 * (3.0*cterm_p - dterm_p)**2

        alpha0_p = 1.0 / (epsilon + IS0_p)**2
        alpha1_p = 6.0 / (epsilon + IS1_p)**2
        alpha2_p = 3.0 / (epsilon + IS2_p)**2

        alpha_sum_p = alpha0_p + alpha1_p + alpha2_p
        alpha_sum_p = jnp.maximum(alpha_sum_p, 1e-14)  # prevent division by zero

        omega0_p = alpha0_p / alpha_sum_p
        omega2_p = alpha2_p / alpha_sum_p

        second = (omega0_p * (aterm_p - 2.0*bterm_p + cterm_p) / 3.0
                  + (omega2_p - 0.5) * (bterm_p - 2.0*cterm_p + dterm_p) / 6.0)

        # Backward WENO similarly with the matching stencil differences:
        aterm_m = 0.5 * (d4 - amx * dq4)   # corresponds to original dFsk[4] etc.
        bterm_m = 0.5 * (d3 - amx * dq3)
        cterm_m = 0.5 * (d2 - amx * dq2)
        dterm_m = 0.5 * (d1 - amx * dq1)

        IS0_m = 13.0 * (aterm_m - bterm_m)**2 + 3.0 * (aterm_m - 3.0*bterm_m)**2
        IS1_m = 13.0 * (bterm_m - cterm_m)**2 + 3.0 * (bterm_m + cterm_m)**2
        IS2_m = 13.0 * (cterm_m - dterm_m)**2 + 3.0 * (3.0*cterm_m - dterm_m)**2

        alpha0_m = 1.0 / (epsilon + IS0_m)**2
        alpha1_m = 6.0 / (epsilon + IS1_m)**2
        alpha2_m = 3.0 / (epsilon + IS2_m)**2

        alpha_sum_m = alpha0_m + alpha1_m + alpha2_m
        alpha_sum_m = jnp.maximum(alpha_sum_m, 1e-14)  # prevent division by zero

        omega0_m = alpha0_m / alpha_sum_m
        omega2_m = alpha2_m / alpha_sum_m

        third = (omega0_m * (aterm_m - 2.0*bterm_m + cterm_m) / 3.0
                 + (omega2_m - 0.5) * (bterm_m - 2.0*cterm_m + dterm_m) / 6.0)

        Fs = -second + third

        # transform back and add to current flux
        if config.equation_of_state == IDEAL_GAS:
            if config.mhd:
                R_col = _eigen_R_col(conserved_state, rhomin, pgmin, gamma, registered_variables, mode)
            else:
                R_col = _eigen_R_col_hydro(conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode)
        elif config.equation_of_state == ISOTHERMAL:
            if config.mhd:
                R_col = _eigen_R_col_iso(conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode)
            else:
                R_col = _eigen_R_col_hydro_iso(conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode)

        if config.dimensionality == 3:
            dF = jnp.einsum('nxyz,xyz->nxyz', R_col, Fs)
        elif config.dimensionality == 2:
            dF = jnp.einsum('nxy,xy->nxy', R_col, Fs)
        else:
            dF = jnp.einsum('nx,x->nx', R_col, Fs)
        return F_current + dF
    
    if config.mhd:
        num_modes = 7
    else:
        num_modes = config.dimensionality + 2

    if config.equation_of_state == ISOTHERMAL:
        num_modes -= 1
    
    # I went for the for loop instead of one einsum
    # because of memory considerations (the full projection
    # matrix does not need to be materialized)
    # But probably this (as I originally had it)
    # would be faster (?).
    return jax.lax.fori_loop(
        0, num_modes,
        mode_flux,
        F_interface
    )

@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_y_native(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    # Transpose to make y the "x" direction
    if config.dimensionality == 2:
        qy = jnp.transpose(conserved_state, (0, 2, 1))
    elif config.dimensionality == 3:
        qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
    
    # Swap components
    momentum_x = qy[registered_variables.momentum_index.x]
    momentum_y = qy[registered_variables.momentum_index.y]

    if config.mhd:
        B_x = qy[registered_variables.magnetic_index.x]
        B_y = qy[registered_variables.magnetic_index.y]
    
    qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
    qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)

    if config.mhd:
        qy = qy.at[registered_variables.magnetic_index.x].set(B_y)
        qy = qy.at[registered_variables.magnetic_index.y].set(B_x)
    
    Fy = _weno_flux_x_native(qy, params, config, registered_variables)
    
    # Transpose back
    if config.dimensionality == 2:
        Fy = jnp.transpose(Fy, (0, 2, 1))
    elif config.dimensionality == 3:
        Fy = jnp.transpose(Fy, (0, 2, 1, 3))
    
    # Swap components back
    Fmomentum_x = Fy[registered_variables.momentum_index.x]
    Fmomentum_y = Fy[registered_variables.momentum_index.y]

    if config.mhd:
        FB_x = Fy[registered_variables.magnetic_index.x]
        FB_y = Fy[registered_variables.magnetic_index.y]
    
    Fy = Fy.at[registered_variables.momentum_index.x].set(Fmomentum_y)
    Fy = Fy.at[registered_variables.momentum_index.y].set(Fmomentum_x)

    if config.mhd:
        Fy = Fy.at[registered_variables.magnetic_index.x].set(FB_y)
        Fy = Fy.at[registered_variables.magnetic_index.y].set(FB_x)
    
    return Fy


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_z_native(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    # Transpose to make z the "x" direction
    qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
    
    # Swap components
    momentum_x = qz[registered_variables.momentum_index.x]
    momentum_z = qz[registered_variables.momentum_index.z]

    if config.mhd:
        B_x = qz[registered_variables.magnetic_index.x]
        B_z = qz[registered_variables.magnetic_index.z]
    
    qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
    qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)

    if config.mhd:
        qz = qz.at[registered_variables.magnetic_index.x].set(B_z)
        qz = qz.at[registered_variables.magnetic_index.z].set(B_x)
    
    Fz = _weno_flux_x_native(qz, params, config, registered_variables)
    
    # Transpose back
    Fz = jnp.transpose(Fz, (0, 3, 2, 1))
    
    # Swap components back
    Fmomentum_x = Fz[registered_variables.momentum_index.x]
    Fmomentum_z = Fz[registered_variables.momentum_index.z]

    if config.mhd:
        FB_x = Fz[registered_variables.magnetic_index.x]
        FB_z = Fz[registered_variables.magnetic_index.z]
    
    Fz = Fz.at[registered_variables.momentum_index.x].set(Fmomentum_z)
    Fz = Fz.at[registered_variables.momentum_index.z].set(Fmomentum_x)

    if config.mhd:
        Fz = Fz.at[registered_variables.magnetic_index.x].set(FB_z)
        Fz = Fz.at[registered_variables.magnetic_index.z].set(FB_x)
    
    return Fz

# -----------------------------------------------------------------------------
# Optional Pallas backend for hydrodynamic WENO fluxes.
# -----------------------------------------------------------------------------


def _backend_name(config: SimulationConfig) -> str:
    """Return a robust string representation of config.backend.

    This intentionally does not import PALLAS/NATIVE_JAX constants.  It works with
    string constants, enum values, or small dataclass-like constant objects whose
    ``name`` or ``value`` carries the backend name.
    """
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


def _hydro_pallas_flux_supported(conserved_state, config: SimulationConfig) -> bool:
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if bool(getattr(config, "mhd", False)):
        return False
    # The first implementation is for ideal-gas Euler.  Isothermal hydro and MHD
    # intentionally fall back to the existing native-JAX implementation.
    if config.equation_of_state != IDEAL_GAS:
        return False
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if conserved_state.ndim != ndim + 1:
        return False
    block_shape = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)
    spatial_shape = conserved_state.shape[1:]
    for n, b in zip(spatial_shape, block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _hydro_indices_for_axis(config: SimulationConfig, registered_variables: RegisteredVariables, axis: int):
    """Return local Euler component indices for a flux normal to ``axis``.

    The returned order is the local characteristic order used by the Euler
    eigenvectors: density, normal momentum, first transverse momentum, optional
    second transverse momentum, energy.  The indices themselves refer to the
    original conserved-state component axis.
    """
    density_index = int(registered_variables.density_index)
    energy_index = int(registered_variables.energy_index)
    ndim = int(config.dimensionality)

    if ndim == 1:
        momentum_x = int(registered_variables.momentum_index)
        return (density_index, momentum_x, energy_index)

    mx = int(registered_variables.momentum_index.x)
    my = int(registered_variables.momentum_index.y)
    if ndim == 2:
        if axis == 0:
            return (density_index, mx, my, energy_index)
        return (density_index, my, mx, energy_index)

    mz = int(registered_variables.momentum_index.z)
    if axis == 0:
        return (density_index, mx, my, mz, energy_index)
    if axis == 1:
        return (density_index, my, mx, mz, energy_index)
    return (density_index, mz, my, mx, energy_index)


def _weno_flux_hydro_pallas(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Pallas implementation of the ideal-gas hydrodynamic WENO flux.

    It computes the same characteristic WENO interface flux as
    ``_weno_flux_x_native`` but evaluates one spatial block at a time and avoids
    materialising shifted full-state arrays inside the reconstruction.  The
    hydro eigenvectors and characteristic projections are evaluated locally in
    the Pallas program, so no full-domain ``L_row``/``R_col`` eigen-projection
    arrays are formed.  The MHD path never calls this function.
    """
    if not _hydro_pallas_flux_supported(conserved_state, config):
        if axis == 0:
            return _weno_flux_x_native(conserved_state, params, config, registered_variables)
        if axis == 1:
            return _weno_flux_y_native(conserved_state, params, config, registered_variables)
        return _weno_flux_z_native(conserved_state, params, config, registered_variables)

    ndim = int(config.dimensionality)
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _hydro_indices_for_axis(config, registered_variables, axis)
    ncomp = len(local_indices)
    num_modes = ndim + 2
    epsilon = 1e-7
    tiny = 1e-14

    # Output block specs keep the conserved-variable axis complete and block only
    # the spatial dimensions.
    if ndim == 1:
        block_shape = (nvars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, gamma_ref, rhomin_ref, pgmin_ref, flux_out_ref):
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

        gamma = gamma_ref[()]
        gm1 = gamma - 1.0
        rhomin = rhomin_ref[()]
        pgmin = pgmin_ref[()]

        def q_at(var_index: int, offset: int):
            if ndim == 1:
                return q_ref[var_index, (ii + offset) % nx]
            if ndim == 2:
                if axis == 0:
                    return q_ref[var_index, (ii + offset) % nx, jj]
                return q_ref[var_index, ii, (jj + offset) % ny]
            if axis == 0:
                return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        def q_local(offset: int):
            return tuple(q_at(idx, offset) for idx in local_indices)

        def primitive_from_q(q):
            rho = q[0]
            mn = q[1]
            if ncomp == 3:
                mt1 = 0.0
                mt2 = 0.0
                energy = q[2]
            elif ncomp == 4:
                mt1 = q[2]
                mt2 = 0.0
                energy = q[3]
            else:
                mt1 = q[2]
                mt2 = q[3]
                energy = q[4]

            inv_rho = 1.0 / rho
            vn = mn * inv_rho
            vt1 = mt1 * inv_rho
            vt2 = mt2 * inv_rho
            v2 = vn * vn + vt1 * vt1 + vt2 * vt2
            pressure = gm1 * (energy - 0.5 * rho * v2)
            return rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure

        def floored_cell(q):
            rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q)
            troubled = (rho < rhomin) | (pressure < pgmin)
            rho_f = jnp.where(troubled, jnp.maximum(rho, rhomin), rho)
            pressure_f = jnp.where(troubled, jnp.maximum(pressure, pgmin), pressure)
            energy_f = jnp.where(troubled, pressure_f / gm1 + 0.5 * rho_f * v2, energy)
            specific_enthalpy = (energy_f + pressure_f) / rho_f
            sound_speed = jnp.sqrt(jnp.maximum(gamma * jnp.abs(pressure_f / rho_f), 1e-12))
            return rho_f, mn, mt1, mt2, energy_f, vn, vt1, vt2, v2, pressure_f, specific_enthalpy, sound_speed

        def flux_from_q(q):
            rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q)
            if ncomp == 3:
                return (mn, mn * vn + pressure, (energy + pressure) * vn)
            if ncomp == 4:
                return (mn, mn * vn + pressure, mt1 * vn, (energy + pressure) * vn)
            return (mn, mn * vn + pressure, mt1 * vn, mt2 * vn, (energy + pressure) * vn)

        qm2 = q_local(-2)
        qm1 = q_local(-1)
        q0 = q_local(0)
        qp1 = q_local(1)
        qp2 = q_local(2)
        qp3 = q_local(3)
        q_stencil = (qm2, qm1, q0, qp1, qp2, qp3)
        f_stencil = tuple(flux_from_q(q) for q in q_stencil)

        # Compute floored primitive/eigenvalue data once for the six cells used
        # by the local Lax-Friedrichs alpha.  The earlier version recomputed this
        # data inside every characteristic mode; keeping it local here avoids both
        # global eigenvalue arrays and repeated per-mode work.
        floored_stencil = tuple(floored_cell(q) for q in q_stencil)

        # Interface eigenvector building blocks at i + 1/2, following
        # _eigenvector_building_blocks in _eigen_hydro.py.
        cell_l = floored_stencil[2]
        cell_r = floored_stencil[3]
        rho_i, mn_i, mt1_i, mt2_i, energy_i, vn_i, vt1_i, vt2_i, v2_i, p_i, h_i, c_i = cell_l
        rho_j, mn_j, mt1_j, mt2_j, energy_j, vn_j, vt1_j, vt2_j, v2_j, p_j, h_j, c_j = cell_r
        rho_face = jnp.maximum(0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)), rhomin)
        vn_face = 0.5 * (mn_i + mn_j) / rho_face
        vt1_face = 0.5 * (mt1_i + mt1_j) / rho_face
        vt2_face = 0.5 * (mt2_i + mt2_j) / rho_face
        h_face = 0.5 * (h_i + h_j)
        v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
        c2_face = gm1 * (h_face - 0.5 * v2_face)
        c_face = jnp.sqrt(jnp.maximum(c2_face, 1e-12))
        inv_c2 = jnp.where(c2_face > 0.0, 1.0 / c2_face, 0.0)

        def left_project(mode: int, values):
            """Project one local vector onto one Euler left eigenvector.

            This is the local Pallas replacement for materialising
            ``_eigen_L_row_hydro(..., mode)`` followed by a full-array einsum.
            ``values`` is either a local conserved-state vector or a local flux
            vector in the normal/tangential component order used by this axis.
            """
            if mode == 0:
                acc = (0.5 * gm1 * v2_face + vn_face * c_face) * values[0]
                acc = acc - (gm1 * vn_face + c_face) * values[1]
                if ncomp == 3:
                    acc = acc + gm1 * values[2]
                elif ncomp == 4:
                    acc = acc - gm1 * vt1_face * values[2] + gm1 * values[3]
                else:
                    acc = (
                        acc
                        - gm1 * vt1_face * values[2]
                        - gm1 * vt2_face * values[3]
                        + gm1 * values[4]
                    )
                return 0.5 * inv_c2 * acc

            if mode == 1:
                acc = (c2_face - 0.5 * gm1 * v2_face) * values[0]
                acc = acc + gm1 * vn_face * values[1]
                if ncomp == 3:
                    acc = acc - gm1 * values[2]
                elif ncomp == 4:
                    acc = acc + gm1 * vt1_face * values[2] - gm1 * values[3]
                else:
                    acc = (
                        acc
                        + gm1 * vt1_face * values[2]
                        + gm1 * vt2_face * values[3]
                        - gm1 * values[4]
                    )
                return inv_c2 * acc

            if mode == 2 and ncomp >= 4:
                return -vt1_face * values[0] + values[2]

            if mode == 3 and ncomp == 5:
                return -vt2_face * values[0] + values[3]

            # Right acoustic wave.
            acc = (0.5 * gm1 * v2_face - vn_face * c_face) * values[0]
            acc = acc - (gm1 * vn_face - c_face) * values[1]
            if ncomp == 3:
                acc = acc + gm1 * values[2]
            elif ncomp == 4:
                acc = acc - gm1 * vt1_face * values[2] + gm1 * values[3]
            else:
                acc = (
                    acc
                    - gm1 * vt1_face * values[2]
                    - gm1 * vt2_face * values[3]
                    + gm1 * values[4]
                )
            return 0.5 * inv_c2 * acc

        def add_right_correction(flux_acc, mode: int, Fs):
            """Add Fs times one local Euler right eigenvector to flux_acc.

            This is the local Pallas replacement for materialising
            ``_eigen_R_col_hydro(..., mode)`` followed by an outer-product style
            einsum.  Returning a Python list keeps the component axis static and
            avoids building small dense eigenvector arrays inside the kernel.
            """
            if mode == 0:
                if ncomp == 3:
                    R = (1.0, vn_face - c_face, h_face - vn_face * c_face)
                elif ncomp == 4:
                    R = (1.0, vn_face - c_face, vt1_face, h_face - vn_face * c_face)
                else:
                    R = (1.0, vn_face - c_face, vt1_face, vt2_face, h_face - vn_face * c_face)
            elif mode == 1:
                if ncomp == 3:
                    R = (1.0, vn_face, 0.5 * v2_face)
                elif ncomp == 4:
                    R = (1.0, vn_face, vt1_face, 0.5 * v2_face)
                else:
                    R = (1.0, vn_face, vt1_face, vt2_face, 0.5 * v2_face)
            elif mode == 2 and ncomp >= 4:
                if ncomp == 4:
                    R = (0.0, 0.0, 1.0, vt1_face)
                else:
                    R = (0.0, 0.0, 1.0, 0.0, vt1_face)
            elif mode == 3 and ncomp == 5:
                R = (0.0, 0.0, 0.0, 1.0, vt2_face)
            else:
                if ncomp == 3:
                    R = (1.0, vn_face + c_face, h_face + vn_face * c_face)
                elif ncomp == 4:
                    R = (1.0, vn_face + c_face, vt1_face, h_face + vn_face * c_face)
                else:
                    R = (1.0, vn_face + c_face, vt1_face, vt2_face, h_face + vn_face * c_face)
            return [flux_acc[slot] + R[slot] * Fs for slot in range(ncomp)]

        def lambda_from_floored_cell(cell, mode: int):
            vn = cell[5]
            c = cell[11]
            if mode == 0:
                return vn - c
            if mode == num_modes - 1:
                return vn + c
            return vn

        def alpha_for_mode(mode: int):
            amx = jnp.abs(lambda_from_floored_cell(floored_stencil[0], mode))
            for k in range(1, 6):
                amx = jnp.maximum(
                    amx,
                    jnp.abs(lambda_from_floored_cell(floored_stencil[k], mode)),
                )
            return amx

        flux_acc = [
            (-f_stencil[1][slot] + 7.0 * f_stencil[2][slot] + 7.0 * f_stencil[3][slot] - f_stencil[4][slot]) / 12.0
            for slot in range(ncomp)
        ]

        for mode in range(num_modes):
            s = tuple(left_project(mode, f_stencil[k]) for k in range(6))
            qproj = tuple(left_project(mode, q_stencil[k]) for k in range(6))

            d0 = s[1] - s[0]
            d1 = s[2] - s[1]
            d2 = s[3] - s[2]
            d3 = s[4] - s[3]
            d4 = s[5] - s[4]

            dq0 = qproj[1] - qproj[0]
            dq1 = qproj[2] - qproj[1]
            dq2 = qproj[3] - qproj[2]
            dq3 = qproj[4] - qproj[3]
            dq4 = qproj[5] - qproj[4]

            amx = alpha_for_mode(mode)

            aterm_p = 0.5 * (d0 + amx * dq0)
            bterm_p = 0.5 * (d1 + amx * dq1)
            cterm_p = 0.5 * (d2 + amx * dq2)
            dterm_p = 0.5 * (d3 + amx * dq3)

            IS0_p = 13.0 * (aterm_p - bterm_p) ** 2 + 3.0 * (aterm_p - 3.0 * bterm_p) ** 2
            IS1_p = 13.0 * (bterm_p - cterm_p) ** 2 + 3.0 * (bterm_p + cterm_p) ** 2
            IS2_p = 13.0 * (cterm_p - dterm_p) ** 2 + 3.0 * (3.0 * cterm_p - dterm_p) ** 2
            alpha0_p = 1.0 / (epsilon + IS0_p) ** 2
            alpha1_p = 6.0 / (epsilon + IS1_p) ** 2
            alpha2_p = 3.0 / (epsilon + IS2_p) ** 2
            alpha_sum_p = jnp.maximum(alpha0_p + alpha1_p + alpha2_p, tiny)
            omega0_p = alpha0_p / alpha_sum_p
            omega2_p = alpha2_p / alpha_sum_p
            second = (
                omega0_p * (aterm_p - 2.0 * bterm_p + cterm_p) / 3.0
                + (omega2_p - 0.5) * (bterm_p - 2.0 * cterm_p + dterm_p) / 6.0
            )

            aterm_m = 0.5 * (d4 - amx * dq4)
            bterm_m = 0.5 * (d3 - amx * dq3)
            cterm_m = 0.5 * (d2 - amx * dq2)
            dterm_m = 0.5 * (d1 - amx * dq1)

            IS0_m = 13.0 * (aterm_m - bterm_m) ** 2 + 3.0 * (aterm_m - 3.0 * bterm_m) ** 2
            IS1_m = 13.0 * (bterm_m - cterm_m) ** 2 + 3.0 * (bterm_m + cterm_m) ** 2
            IS2_m = 13.0 * (cterm_m - dterm_m) ** 2 + 3.0 * (3.0 * cterm_m - dterm_m) ** 2
            alpha0_m = 1.0 / (epsilon + IS0_m) ** 2
            alpha1_m = 6.0 / (epsilon + IS1_m) ** 2
            alpha2_m = 3.0 / (epsilon + IS2_m) ** 2
            alpha_sum_m = jnp.maximum(alpha0_m + alpha1_m + alpha2_m, tiny)
            omega0_m = alpha0_m / alpha_sum_m
            omega2_m = alpha2_m / alpha_sum_m
            third = (
                omega0_m * (aterm_m - 2.0 * bterm_m + cterm_m) / 3.0
                + (omega2_m - 0.5) * (bterm_m - 2.0 * cterm_m + dterm_m) / 6.0
            )

            Fs = -second + third
            flux_acc = add_right_correction(flux_acc, mode, Fs)

        # Set every output component.  Hydro should fill all components, but the
        # explicit zeroing makes failures obvious if a future registry adds fields.
        zero = flux_acc[0] * 0.0
        for var in range(nvars):
            flux_out_ref[var, ...] = zero
        for slot, var in enumerate(local_indices):
            flux_out_ref[var, ...] = flux_acc[slot]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[in_state_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=bool(getattr(config, "pallas_interpret", False)),
        name=f"hydro_weno_flux_axis_{axis}",
        **kwargs,
    )(
        conserved_state,
        jnp.asarray(params.gamma, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_pressure, dtype=conserved_state.dtype),
    )


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_x(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    if _hydro_pallas_flux_supported(conserved_state, config):
        return _weno_flux_hydro_pallas(
            conserved_state, params, config, registered_variables, axis=0
        )
    return _weno_flux_x_native(conserved_state, params, config, registered_variables)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_y(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    if _hydro_pallas_flux_supported(conserved_state, config) and int(config.dimensionality) >= 2:
        return _weno_flux_hydro_pallas(
            conserved_state, params, config, registered_variables, axis=1
        )
    return _weno_flux_y_native(conserved_state, params, config, registered_variables)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_z(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    if _hydro_pallas_flux_supported(conserved_state, config) and int(config.dimensionality) == 3:
        return _weno_flux_hydro_pallas(
            conserved_state, params, config, registered_variables, axis=2
        )
    return _weno_flux_z_native(conserved_state, params, config, registered_variables)
