"""
Convenience functions for operations that combine multiple elements
of an array based on some stencil, e.g. b_i <- a_{i + 1} + a_{i - 1}.
Allows for code "closer to the math".
"""

# general
import jax
import jax.numpy as jnp
from functools import partial

# typechecking
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from typing import Tuple, Union

from astronomix.option_classes.simulation_config import OPEN_BOUNDARY, PERIODIC_BOUNDARY


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["shift", "axis"])
def custom_roll(input_array: jnp.ndarray, shift: int, axis: int) -> jnp.ndarray:
    i = (-shift) % input_array.shape[axis]
    return jax.lax.concatenate(
        [
            jax.lax.slice_in_dim(input_array, i, input_array.shape[axis], axis=axis),
            jax.lax.slice_in_dim(input_array, 0, i, axis=axis),
        ],
        dimension=axis,
    )

@partial(jax.jit, static_argnames=["shift", "axis"])
def custom_shift_open(input_array: jnp.ndarray, shift: int, axis: int) -> jnp.ndarray:
    size = input_array.shape[axis]

    if shift == 0:
        return input_array

    if shift > 0:
        # shift right → duplicate left boundary
        sliced = jax.lax.slice_in_dim(input_array, 0, size - shift, axis=axis)
        pad_value = jax.lax.slice_in_dim(input_array, 0, 1, axis=axis)
        pad = jnp.repeat(pad_value, shift, axis=axis)
        return jax.lax.concatenate([pad, sliced], dimension=axis)

    else:
        # shift left → duplicate right boundary
        shift = -shift
        sliced = jax.lax.slice_in_dim(input_array, shift, size, axis=axis)
        pad_value = jax.lax.slice_in_dim(input_array, size - 1, size, axis=axis)
        pad = jnp.repeat(pad_value, shift, axis=axis)
        return jax.lax.concatenate([sliced, pad], dimension=axis)

def _shift(input_array: jnp.ndarray, shift: int, axis: int, boundaries: int) -> jnp.ndarray:
    if boundaries == PERIODIC_BOUNDARY:
        return custom_roll(input_array, shift, axis)
    elif boundaries == OPEN_BOUNDARY:
        return custom_shift_open(input_array, shift, axis)
    else:
        raise ValueError(f"Invalid boundaries: {boundaries}. Supported boundaries are PERIODIC_BOUNDARY and OPEN_BOUNDARY.")


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["indices", "axis"])
def _stencil_add(
    input_array: jnp.ndarray,
    indices: Tuple[int, ...],
    factors: Tuple[Union[float, Float[Array, ""]], ...],
    axis: int,
) -> jnp.ndarray:
    """
    Combines elements of an array additively
        output_i <- sum_j factors_j * input_array_{i + indices_j}

    Args:
        input_array: The array to operate on.
        indices: output_i <- sum_j factors_j * input_array_{i + indices_j}
        factors: output_i <- sum_j factors_j * input_array_{i + indices_j}
        axis: The axis along which to operate.

    Returns:
        output_i <- sum_j factors_j * input_array_{i + indices_j}
    """

    output = sum(
        factor * custom_roll(input_array, -index, axis=axis)
        for factor, index in zip(factors, indices)
    )

    return output
