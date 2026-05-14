"""
Helpers for computing Jacobians.
"""

import jax
import jax.numpy as jnp

from astronomix._fluid_equations._equations import conserved_state_from_primitive
from astronomix.analysis_helpers._exposed_rhs import _exposed_rhs
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping import time_integration


def single_xmode_rhs_jacobian2D(
    primitive_state_unperturbed,
    config,
    params,
    registered_variables,
    helper_data,
    wavelength,
    assembly_batch_size=4,
):
    """
    Return the dense +kx Fourier-block Jacobian of the simulator RHS.

    The full semi-discrete RHS is

        dU/dt = R(U),

    where U is the conserved state on the full 2D grid. The full Jacobian dR/dU
    has size

        (nvar * Nx * Ny) x (nvar * Nx * Ny),

    which is usually too large to form explicitly.

    If the unperturbed state U0 depends only on y, the x-direction is
    translation-invariant and different streamwise Fourier modes decouple in
    the linearized problem. We restrict the perturbation to one Fourier mode,

        dU(x, y) = qhat(y) exp(i kx x),

    with

        qhat(y) in C^(nvar * Ny).

    The returned matrix represents

        L_k qhat = P_k J[ qhat(y) exp(i kx x) ],

    where J = dR/dU evaluated at U0 and P_k projects back onto the +kx Fourier
    mode.

    A complex Fourier amplitude is just a compact representation of sine and
    cosine perturbations:

        Re[qhat exp(i kx x)]
        = Re(qhat) cos(kx x) - Im(qhat) sin(kx x).

    Since the simulator RHS is real-valued, the complex tangent action is built
    from two real JVPs:

        J Re[qhat exp(i kx x)] + i J Im[qhat exp(i kx x)].

    Unlike a direct jacfwd over the real-packed vector [Re(qhat), Im(qhat)],
    this function assembles the complex Jacobian column-by-column in small
    batches. This avoids materializing all tangent directions at once.

    Args:
        primitive_state_unperturbed:
            Primitive base state. It is converted to conserved variables before
            linearization.
        config:
            Simulation config. The x-boundary should be periodic.
        params:
            Simulation parameters.
        registered_variables:
            Variable registry.
        helper_data:
            Grid/helper data containing cell centers.
        wavelength:
            Physical wavelength of the x-Fourier perturbation. Must be
            commensurate with the x-domain length.
        assembly_batch_size:
            Number of Jacobian columns to compute per JVP batch. Lower this if
            GPU memory is still tight.

    Returns:
        J_complex:
            Complex array of shape (nvar * Ny, nvar * Ny), representing the raw
            +kx Fourier-block tangent operator of the implemented simulator RHS.
    """

    config = finalize_config(config, primitive_state_unperturbed.shape)

    cons0 = conserved_state_from_primitive(
        primitive_state_unperturbed,
        params.gamma,
        config,
        registered_variables,
    )

    real_dtype = cons0.dtype
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64

    nvar, Nx, Ny = cons0.shape
    n = nvar * Ny

    Lx = float(config.box_size.x)
    m_float = Lx / float(wavelength)
    m = int(round(m_float))

    if abs(m_float - m) > 1e-12:
        raise ValueError(
            f"wavelength={wavelength} is not commensurate with Lx={Lx}. "
            f"Got Lx / wavelength = {m_float}, not an integer."
        )

    kx = 2.0 * jnp.pi * m / Lx

    X = helper_data.geometric_centers[:, :, 0]
    phase = jnp.exp(1j * kx * X).astype(complex_dtype)
    phase_conj = jnp.conj(phase)

    def rhs(q):
        return _exposed_rhs(q, params, config, registered_variables)

    def rhs_jvp(dq_real):
        _, Jdq = jax.jvp(rhs, (cons0,), (dq_real,))
        return Jdq

    def apply_one_real_column(qhat_flat_real):
        """
        Apply L_k to one real basis vector qhat.

        Because L_k is complex-linear, applying it to all real basis vectors
        gives all columns of the complex matrix L_k = A + iB.
        """
        qhat = qhat_flat_real.reshape(nvar, Ny).astype(complex_dtype)

        dq_complex = qhat[:, None, :] * phase[None, :, :]

        Jdq_complex = (
            rhs_jvp(jnp.real(dq_complex).astype(real_dtype))
            + 1j * rhs_jvp(jnp.imag(dq_complex).astype(real_dtype))
        )

        Jhat = jnp.mean(
            Jdq_complex * phase_conj[None, :, :],
            axis=1,
        )

        return Jhat.reshape(-1)

    apply_batch = jax.jit(jax.vmap(apply_one_real_column))

    column_blocks = []

    for start in range(0, n, assembly_batch_size):
        stop = min(start + assembly_batch_size, n)

        cols = jnp.arange(start, stop)
        basis_batch = jax.nn.one_hot(cols, n, dtype=real_dtype)

        # Shape: (batch, n). Each row is one output column.
        out_batch = apply_batch(basis_batch)

        # Store as matrix columns.
        column_blocks.append(jnp.swapaxes(out_batch, 0, 1))

    J_complex = jnp.concatenate(column_blocks, axis=1)

    return J_complex

def single_xmode_jacobian2Dt(
    primitive_state_unperturbed,
    config,
    params,
    registered_variables,
    helper_data,
    wavelength,
    assembly_batch_size=4,
):
    """
    Return the dense +kx Fourier-block Jacobian of the time-integrated simulator.
    This might be useful for bringing out modes with large Re(\lambda).
    """
    config = finalize_config(config, primitive_state_unperturbed.shape)

    real_dtype = primitive_state_unperturbed.dtype
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64

    nvar, Nx, Ny = primitive_state_unperturbed.shape
    n = nvar * Ny

    Lx = float(config.box_size.x)
    m_float = Lx / float(wavelength)
    m = int(round(m_float))

    if abs(m_float - m) > 1e-12:
        raise ValueError(
            f"wavelength={wavelength} is not commensurate with Lx={Lx}. "
            f"Got Lx / wavelength = {m_float}, not an integer."
        )

    kx = 2.0 * jnp.pi * m / Lx

    X = helper_data.geometric_centers[:, :, 0]
    phase = jnp.exp(1j * kx * X).astype(complex_dtype)
    phase_conj = jnp.conj(phase)

    def rhs(q):
        return time_integration(q, config, params, registered_variables)

    def rhs_jvp(dq_real):
        _, Jdq = jax.jvp(rhs, (primitive_state_unperturbed,), (dq_real,))
        return Jdq

    def apply_one_real_column(qhat_flat_real):
        """
        Apply L_k to one real basis vector qhat.

        Because L_k is complex-linear, applying it to all real basis vectors
        gives all columns of the complex matrix L_k = A + iB.
        """
        qhat = qhat_flat_real.reshape(nvar, Ny).astype(complex_dtype)

        dq_complex = qhat[:, None, :] * phase[None, :, :]

        Jdq_complex = (
            rhs_jvp(jnp.real(dq_complex).astype(real_dtype))
            + 1j * rhs_jvp(jnp.imag(dq_complex).astype(real_dtype))
        )

        Jhat = jnp.mean(
            Jdq_complex * phase_conj[None, :, :],
            axis=1,
        )

        return Jhat.reshape(-1)

    apply_batch = jax.jit(jax.vmap(apply_one_real_column))

    column_blocks = []

    for start in range(0, n, assembly_batch_size):

        print(f"Iteration {start // assembly_batch_size} / {(n + assembly_batch_size - 1) // assembly_batch_size}")
        
        stop = min(start + assembly_batch_size, n)

        cols = jnp.arange(start, stop)
        basis_batch = jax.nn.one_hot(cols, n, dtype=real_dtype)

        # Shape: (batch, n). Each row is one output column.
        out_batch = apply_batch(basis_batch)

        # Store as matrix columns.
        column_blocks.append(jnp.swapaxes(out_batch, 0, 1))

    J_complex = jnp.concatenate(column_blocks, axis=1)

    return J_complex
