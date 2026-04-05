from functools import partial

import jax
import jax.numpy as jnp   
from astronomix._fluid_equations._equations import conserved_state_from_primitive, primitive_state_from_conserved
from astronomix._stencil_operations._stencil_operations import _shift, _stencil_add

# the fv one is preliminary, in the future I want an all source
# term scheme
@partial(jax.jit, static_argnames=('config', 'registered_variables'))
def fv_viscosity_update(primitive_state, params, config, registered_variables, dt):

    mu = params.viscosity
    dx = config.grid_spacing
    ndim = config.dimensionality

    rho = primitive_state[registered_variables.density_index]
    v = primitive_state[1:ndim + 1]                            # (ndim, *spatial)

    # Cell-center velocity gradient (2nd-order centered, for tangential terms)
    grad_v_cc = jnp.stack([
        (_shift(v, -1, axis=j + 1) - _shift(v, 1, axis=j + 1)) / (2.0 * dx)
        for j in range(ndim)
    ], axis=1)                                                 # (ndim_i, ndim_j, *spatial)

    div_v_cc = jnp.trace(grad_v_cc, axis1=0, axis2=1)         # (*spatial)

    mom_src = jnp.zeros_like(v)                                # (ndim, *spatial)
    energy_src = jnp.zeros_like(rho)                           # (*spatial)

    for j in range(ndim):
        ax = j + 1  # array axis (0 = component, 1..3 = spatial)

        # ── right face i+1/2 along direction j ────────────────────

        # normal derivative: compact, 2nd-order exact at face
        dv_dxj = (_shift(v, -1, axis=ax) - v) / dx            # (ndim, *spatial)

        # tangential derivatives: average cell-center values to face
        # grad_v_cc[j][i] = ∂v_j/∂x_i  →  need this for all i
        dvj_dxi = 0.5 * (grad_v_cc[j] + _shift(grad_v_cc[j], -1, axis=ax))

        # ∇·v at face
        div_v = 0.5 * (div_v_cc + _shift(div_v_cc, -1, axis=ax))

        # δ_{ij} with broadcasting shape (ndim, 1, 1, ...)
        d_ij = jnp.zeros(ndim).at[j].set(1.0)
        d_ij = d_ij.reshape((-1,) + (1,) * rho.ndim)

        # τ_{ij} at face for all i:
        # τ_{ij} = μ (∂v_i/∂x_j + ∂v_j/∂x_i − ⅔ δ_{ij} ∇·v)
        tau_face = mu * (dv_dxj + dvj_dxi - (2.0 / 3.0) * d_ij * div_v)

        # velocity at face (for energy flux)
        v_face = 0.5 * (v + _shift(v, -1, axis=ax))

        # viscous energy flux:  Σ_i v_i τ_{ij}
        e_flux = jnp.sum(v_face * tau_face, axis=0)           # (*spatial)

        # ── conservative divergence: (F_{i+1/2} − F_{i-1/2}) / dx ──
        mom_src    += (tau_face - _shift(tau_face, 1, axis=ax)) / dx
        energy_src += (e_flux   - _shift(e_flux,   1, axis=ax)) / dx

    S_visc = jnp.zeros_like(primitive_state)
    S_visc = S_visc.at[1:ndim + 1].set(mom_src)
    S_visc = S_visc.at[registered_variables.energy_index].set(energy_src)

    # add the source term to the conserved state with an Euler step
    # (this is a bit hacky and I would prefer a more proper solution in 
    # the future)

    conserved_state = conserved_state_from_primitive(primitive_state, params.gamma, config, registered_variables)
    conserved_state += S_visc * dt
    primitive_state = primitive_state_from_conserved(conserved_state, params.gamma, config, registered_variables)
    return primitive_state

@partial(jax.jit, static_argnames=('config', 'registered_variables'))
def fd_viscosity_source(primitive_state, params, config, registered_variables):

    mu = params.viscosity # the dynamic viscosity
    dx = config.grid_spacing
    ndim = config.dimensionality

    rho = primitive_state[registered_variables.density_index]
    v = primitive_state[1:ndim + 1]                            # (ndim, *spatial)

    def _d1(field, ax):
        return _stencil_add(
            field, indices=(3, 2, 1, -1, -2, -3),
            factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0), axis=ax,
        ) / (60.0 * dx)

    # velocity gradient tensor  G_{ij} = ∂v_i/∂x_j   (ndim, ndim, *spatial)
    grad_v = jnp.stack([_d1(v, j + 1) for j in range(ndim)], axis=1)

    # stress tensor  τ_{ij} = μ (G_{ij} + G_{ji} − ⅔ δ_{ij} ∇·v)
    div_v = jnp.trace(grad_v, axis1=0, axis2=1)               # (*spatial)
    delta = jnp.eye(ndim)[(slice(None), slice(None)) + (None,) * rho.ndim]
    tau = mu * (grad_v + grad_v.swapaxes(0, 1)
                - (2.0 / 3.0) * delta * div_v)                 # (ndim, ndim, *spatial)

    # momentum source  (∇·τ)_i = Σ_j ∂τ_{ij}/∂x_j            (ndim, *spatial)
    div_tau = sum(_d1(tau[:, j], j + 1) for j in range(ndim))

    # energy source  Σ_j ∂/∂x_j (Σ_i v_i τ_{ij})
    v_dot_tau = jnp.einsum('i...,ij...->j...', v, tau)        # (ndim, *spatial)
    energy_src = sum(_d1(v_dot_tau[j], j) for j in range(ndim))

    S_visc = jnp.zeros_like(primitive_state)
    S_visc = S_visc.at[1:ndim + 1].set(div_tau)
    S_visc = S_visc.at[registered_variables.energy_index].set(energy_src)

    return S_visc