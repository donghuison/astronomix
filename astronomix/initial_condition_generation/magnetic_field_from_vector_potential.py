import jax.numpy as jnp
from typing import Callable, Tuple

from astronomix.option_classes.simulation_config import (
    SimulationConfig,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
)
from astronomix._finite_difference._maths._differencing import finite_difference_int6
from astronomix._finite_difference._maths._interpolate import interp_face_to_center

def setup_magnetic_fields_from_vector_potential(
    config: SimulationConfig,
    vector_potential_func: Callable,
    *args,
    **kwargs
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculates the cell-centered and face-centered magnetic fields from a given 
    vector potential function, correctly applying discrete curl operations 
    depending on the solver mode (FINITE_DIFFERENCE or FINITE_VOLUME) to ensure
    a divergence-free magnetic field.

    Note: Uniform background fields cannot be correctly generated from a periodic 
    vector potential. Add background fields to the returned arrays after calling.

    Args:
        config: The simulation configuration object.
        vector_potential_func: A callable `f(X, Y, Z, *args, **kwargs)` that returns 
                               a tuple (A_x, A_y, A_z) evaluated at the given coordinates.
        *args, **kwargs: Additional arguments to pass to `vector_potential_func` (e.g. time t).

    Returns:
        B_x, B_y, B_z: Cell-centered magnetic field components.
        bxb, byb, bzb: Face-centered (interface) magnetic field components.
    """
    _XAXIS, _YAXIS, _ZAXIS = 0, 1, 2

    # Extract dimensions and resolution. 
    # Handles both uniform 1D scalar config and 3D tuple config automatically.
    try:
        Lx, Ly, Lz = config.box_size
    except TypeError:
        Lx = Ly = Lz = config.box_size

    try:
        Nx, Ny, Nz = config.num_cells
    except TypeError:
        Nx = Ny = Nz = config.num_cells

    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

    if config.solver_mode == FINITE_DIFFERENCE:
        # Build the staggered edge grids for the vector potential components
        x_l = jnp.linspace(dx, Lx, Nx, endpoint=True)
        y_l = jnp.linspace(dy, Ly, Ny, endpoint=True)
        z_l = jnp.linspace(dz, Lz, Nz, endpoint=True)
        
        x_c = jnp.linspace(dx/2, Lx + dx/2, Nx, endpoint=False)
        y_c = jnp.linspace(dy/2, Ly + dy/2, Ny, endpoint=False)
        z_c = jnp.linspace(dz/2, Lz + dz/2, Nz, endpoint=False)

        # A_x lives on yz-edges: (i, j+1/2, k+1/2) -> (x_c, y_l, z_l)
        Xax, Yax, Zax = jnp.meshgrid(x_c, y_l, z_l, indexing="ij")
        # A_y lives on zx-edges: (i+1/2, j, k+1/2) -> (x_l, y_c, z_l)
        Xay, Yay, Zay = jnp.meshgrid(x_l, y_c, z_l, indexing="ij")
        # A_z lives on xy-edges: (i+1/2, j+1/2, k) -> (x_l, y_l, z_c)
        Xaz, Yaz, Zaz = jnp.meshgrid(x_l, y_l, z_c, indexing="ij")

        # Evaluate the given vector potential function
        A_x, _, _ = vector_potential_func(Xax, Yax, Zax, *args, **kwargs)
        _, A_y, _ = vector_potential_func(Xay, Yay, Zay, *args, **kwargs)
        _, _, A_z = vector_potential_func(Xaz, Yaz, Zaz, *args, **kwargs)

        # Apply discrete curl -> face-centered B-field
        bxb = (1.0 / dy) * finite_difference_int6(A_z, _YAXIS) \
            - (1.0 / dz) * finite_difference_int6(A_y, _ZAXIS)
        byb = (1.0 / dz) * finite_difference_int6(A_x, _ZAXIS) \
            - (1.0 / dx) * finite_difference_int6(A_z, _XAXIS)
        bzb = (1.0 / dx) * finite_difference_int6(A_y, _XAXIS) \
            - (1.0 / dy) * finite_difference_int6(A_x, _YAXIS)

        # Interpolate faces to centers -> cell-centered B-field
        B_x = interp_face_to_center(bxb, _XAXIS)
        B_y = interp_face_to_center(byb, _YAXIS)
        B_z = interp_face_to_center(bzb, _ZAXIS)

    elif config.solver_mode == FINITE_VOLUME:
        # Build cell-centered coordinates
        x_c = jnp.linspace(dx/2, Lx + dx/2, Nx, endpoint=False)
        y_c = jnp.linspace(dy/2, Ly + dy/2, Ny, endpoint=False)
        z_c = jnp.linspace(dz/2, Lz + dz/2, Nz, endpoint=False)
        Xc, Yc, Zc = jnp.meshgrid(x_c, y_c, z_c, indexing="ij")

        # Evaluate the given vector potential function
        A_x_c, A_y_c, A_z_c = vector_potential_func(Xc, Yc, Zc, *args, **kwargs)

        # Central difference helper for FV mode div(B)=0 guarantee
        def central_diff(f, axis, d):
            return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (2 * d)

        # Apply discrete central difference curl -> cell-centered B-field
        B_x = central_diff(A_z_c, _YAXIS, dy) - central_diff(A_y_c, _ZAXIS, dz)
        B_y = central_diff(A_x_c, _ZAXIS, dz) - central_diff(A_z_c, _XAXIS, dx)
        B_z = central_diff(A_y_c, _XAXIS, dx) - central_diff(A_x_c, _YAXIS, dy)

        # For FV, interface fields simply mirror the cell centers natively
        bxb, byb, bzb = B_x, B_y, B_z
    else:
        raise ValueError(f"Unsupported solver_mode: {config.solver_mode}")

    return B_x, B_y, B_z, bxb, byb, bzb