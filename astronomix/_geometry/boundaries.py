from functools import partial
import jax
import jax.numpy as jnp
 
from astronomix.option_classes.simulation_config import (
    GAS_STATE,
    MAGNETIC_FIELD_ONLY,
    MHD_JET_BOUNDARY,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    STATE_TYPE,
    VELOCITY_ONLY,
    SimulationConfig,
)
 
 
# -----------------------------------------------------------------------------
# Indexing helper
# -----------------------------------------------------------------------------
 
def _axis_slice(axis: int, start, stop, ndim: int) -> tuple:
    """Build an indexing tuple of the form ``[:, ..., slice(start, stop), ..., :]``
    that slices only the requested ``axis``. ``axis`` and ``ndim`` are static at
    trace time, so this constructs a plain Python tuple of slices."""
    return (
        (slice(None),) * axis
        + (slice(start, stop),)
        + (slice(None),) * (ndim - axis - 1)
    )
 
 
# -----------------------------------------------------------------------------
# Open boundaries — broadcast the first/last interior cell into all ghost cells
# -----------------------------------------------------------------------------
 
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _open_left_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    """All left ghost cells ← first interior cell (via length-1 broadcast)."""
    ndim = primitive_state.ndim
    src = _axis_slice(axis, num_ghost_cells, num_ghost_cells + 1, ndim)
    dst = _axis_slice(axis, 0, num_ghost_cells, ndim)
    return primitive_state.at[dst].set(primitive_state[src])
 
 
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _open_right_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    """All right ghost cells ← last interior cell (via length-1 broadcast)."""
    ndim = primitive_state.ndim
    src = _axis_slice(axis, -num_ghost_cells - 1, -num_ghost_cells, ndim)
    dst = _axis_slice(axis, -num_ghost_cells, None, ndim)
    return primitive_state.at[dst].set(primitive_state[src])
 
 
# -----------------------------------------------------------------------------
# Periodic boundaries — wrap interior cells to the opposite ghost region
# -----------------------------------------------------------------------------
 
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _periodic_boundaries(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    """Wrap both ghost regions with a single scatter per side."""
    ndim = primitive_state.ndim
    ng = num_ghost_cells
 
    # Left ghosts ← last ``ng`` interior cells  (state[..., :ng] = state[..., -2*ng:-ng])
    left_dst = _axis_slice(axis, 0, ng, ndim)
    left_src = _axis_slice(axis, -2 * ng, -ng, ndim)
    primitive_state = primitive_state.at[left_dst].set(primitive_state[left_src])
 
    # Right ghosts ← first ``ng`` interior cells  (state[..., -ng:] = state[..., ng:2*ng])
    right_dst = _axis_slice(axis, -ng, None, ndim)
    right_src = _axis_slice(axis, ng, 2 * ng, ndim)
    primitive_state = primitive_state.at[right_dst].set(primitive_state[right_src])
 
    return primitive_state
 
 
# -----------------------------------------------------------------------------
# Reflective boundaries — mirror interior block, negate normal velocity
# -----------------------------------------------------------------------------
 
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _reflective_left_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    """Mirror the first ``num_ghost_cells`` interior cells into the left ghost
    region and negate the velocity component normal to the boundary
    (``var_index == axis`` by convention)."""
    ndim = primitive_state.ndim
    ng = num_ghost_cells
 
    # Read the interior block and reverse it along the spatial axis.
    src = _axis_slice(axis, ng, 2 * ng, ndim)
    block = jnp.flip(primitive_state[src], axis=axis)
 
    # Negate the normal-velocity component on the (small) mirrored block before
    # scattering — cheaper than a second scatter on the full array.
    block = block.at[axis].multiply(-1.0)
 
    dst = _axis_slice(axis, 0, ng, ndim)
    return primitive_state.at[dst].set(block)
 
 
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _reflective_right_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    """Mirror image of ``_reflective_left_boundary`` for the right side."""
    ndim = primitive_state.ndim
    ng = num_ghost_cells
 
    src = _axis_slice(axis, -2 * ng, -ng, ndim)
    block = jnp.flip(primitive_state[src], axis=axis)
    block = block.at[axis].multiply(-1.0)
 
    dst = _axis_slice(axis, -ng, None, ndim)
    return primitive_state.at[dst].set(block)
 
 
# -----------------------------------------------------------------------------
# MHD jet injection boundary (2D, y-left only)
# -----------------------------------------------------------------------------
 
@partial(
    jax.jit,
    static_argnames=[
        "axis",
        "num_ghost_cells",
        "grid_spacing",
        "num_cells",
        "type_handled",
    ],
)
def _jet_left_boundary(
    primitive_state: STATE_TYPE,
    num_ghost_cells: int,
    axis: int,
    grid_spacing: float,
    num_cells: int,
    type_handled: int,
) -> STATE_TYPE:
    # Start from an open boundary (single broadcast; no loop).
    primitive_state = _open_left_boundary(primitive_state, num_ghost_cells, axis)
 
    half_inj_width = 0.025
    half_inj_cell_num = int(half_inj_width / grid_spacing)
 
    B0 = 200**0.5
    to_set_gas_state = jnp.array([5 / 3, 800.0, 0.0, 1.0])
    to_set_velocity = jnp.array([800.0, 0.0, 0.0])
    to_set_magnetic_field = jnp.array([B0, 0.0, 0.0])
 
    jet_lo = num_cells // 2 - half_inj_cell_num
    jet_hi = num_cells // 2 + half_inj_cell_num
 
    if type_handled == GAS_STATE:
        primitive_state = primitive_state.at[
            :, 0:num_ghost_cells, jet_lo:jet_hi
        ].set(to_set_gas_state[:, None, None])
    elif type_handled == VELOCITY_ONLY:
        primitive_state = primitive_state.at[
            :, 0:num_ghost_cells, jet_lo:jet_hi
        ].set(to_set_velocity[:, None, None])
    elif type_handled == MAGNETIC_FIELD_ONLY:
        primitive_state = primitive_state.at[
            :, 0:num_ghost_cells, jet_lo:jet_hi
        ].set(to_set_magnetic_field[:, None, None])
 
    return primitive_state
 
 
# -----------------------------------------------------------------------------
# Per-axis dispatch helpers (branches on static config values only — no loops)
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["num_ghost_cells", "bs", "axis"])
def _apply_axis_bcs(
    primitive_state: STATE_TYPE, num_ghost_cells: int, bs, axis: int
) -> STATE_TYPE:
    """Apply left/right/periodic BCs along a single spatial axis. All branches
    are on static (``SimulationConfig``) fields and resolve at trace time."""
    if bs.left_boundary == OPEN_BOUNDARY:
        primitive_state = _open_left_boundary(primitive_state, num_ghost_cells, axis=axis)
    elif bs.left_boundary == REFLECTIVE_BOUNDARY:
        primitive_state = _reflective_left_boundary(primitive_state, num_ghost_cells, axis=axis)
 
    if bs.right_boundary == OPEN_BOUNDARY:
        primitive_state = _open_right_boundary(primitive_state, num_ghost_cells, axis=axis)
    elif bs.right_boundary == REFLECTIVE_BOUNDARY:
        primitive_state = _reflective_right_boundary(primitive_state, num_ghost_cells, axis=axis)
 
    if (
        bs.left_boundary == PERIODIC_BOUNDARY
        and bs.right_boundary == PERIODIC_BOUNDARY
    ):
        primitive_state = _periodic_boundaries(primitive_state, num_ghost_cells, axis=axis)
 
    return primitive_state
 
 
@partial(jax.jit, static_argnames=["num_ghost_cells", "bs"])
def _apply_axis_bcs_1d(
    primitive_state: STATE_TYPE, num_ghost_cells: int, bs
) -> STATE_TYPE:
    """1D has a flat ``boundary_settings`` (no x/y/z split). The general
    reflective functions handle ``axis=1`` correctly — in 1D the normal-velocity
    variable index happens to equal ``axis``, which is exactly the convention
    those helpers already use."""
    if bs.left_boundary == OPEN_BOUNDARY:
        primitive_state = _open_left_boundary(primitive_state, num_ghost_cells, axis=1)
    elif bs.left_boundary == REFLECTIVE_BOUNDARY:
        primitive_state = _reflective_left_boundary(primitive_state, num_ghost_cells, axis=1)
 
    if bs.right_boundary == OPEN_BOUNDARY:
        primitive_state = _open_right_boundary(primitive_state, num_ghost_cells, axis=1)
    elif bs.right_boundary == REFLECTIVE_BOUNDARY:
        primitive_state = _reflective_right_boundary(primitive_state, num_ghost_cells, axis=1)
 
    if (
        bs.left_boundary == PERIODIC_BOUNDARY
        and bs.right_boundary == PERIODIC_BOUNDARY
    ):
        primitive_state = _periodic_boundaries(primitive_state, num_ghost_cells, axis=1)
 
    return primitive_state
 
 
# -----------------------------------------------------------------------------
# Top-level boundary handler
# -----------------------------------------------------------------------------
 
@partial(jax.jit, static_argnames=["config", "type_handled"])
def _boundary_handler(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    type_handled: int = GAS_STATE,
) -> STATE_TYPE:
    """Apply all boundary conditions to the primitive state."""
    ng = config.num_ghost_cells
 
    if config.dimensionality == 1:
        return _apply_axis_bcs_1d(primitive_state, ng, config.boundary_settings)
 
    # 2D / 3D: dispatch per axis. Each call is a single traced branch.
    primitive_state = _apply_axis_bcs(
        primitive_state, ng, config.boundary_settings.x, axis=1
    )
    primitive_state = _apply_axis_bcs(
        primitive_state, ng, config.boundary_settings.y, axis=2
    )
    if config.dimensionality == 3:
        primitive_state = _apply_axis_bcs(
            primitive_state, ng, config.boundary_settings.z, axis=3
        )
 
    # MHD jet injection (2D only, y-left).
    if (
        config.dimensionality == 2
        and config.boundary_settings.y.left_boundary == MHD_JET_BOUNDARY
    ):
        primitive_state = _jet_left_boundary(
            primitive_state,
            ng,
            axis=2,
            grid_spacing=config.grid_spacing,
            num_cells=config.num_cells.y,
            type_handled=type_handled,
        )
 
    return primitive_state