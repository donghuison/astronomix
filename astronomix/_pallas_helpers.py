"""Shared Pallas-backend utilities used across the FD and FV paths.

This module is the single place that:
- imports Pallas / Triton (and exposes ``pl is None`` if unavailable),
- normalises ``config.pallas_block_shape`` to a 3-tuple,
- builds Triton ``CompilerParams`` from config knobs,
- exposes the ``backend == PALLAS`` predicate.

Every Pallas kernel module under ``astronomix`` should import from here so
new knobs / fallbacks only need to be added once.
"""

from astronomix.option_classes.simulation_config import PALLAS, SimulationConfig

try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - Pallas optional
    pl = None

try:
    from jax.experimental.pallas import triton as pltriton
except Exception:  # pragma: no cover - Triton GPU backend optional
    pltriton = None


def _backend_is_pallas(config: SimulationConfig) -> bool:
    return config.backend == PALLAS


def _default_pallas_block_shape(ndim: int) -> tuple[int, int, int]:
    if ndim == 1:
        return (128, 1, 1)
    if ndim == 2:
        return (16, 16, 1)
    return (4, 4, 8)


def _as_3tuple_block_shape(block_shape, ndim: int) -> tuple[int, int, int]:
    """Normalise whatever the user supplied (None / str / tuple) to
    ``(bx, by, bz)`` with the inactive dims forced to 1.  Pallas grid
    construction depends on this tuple being canonical."""
    if block_shape is None:
        return _default_pallas_block_shape(ndim)
    if isinstance(block_shape, str):
        parts = tuple(int(p.strip()) for p in block_shape.split(",") if p.strip())
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
    """Return Triton ``CompilerParams`` (or None if the Triton backend is
    not available / the user opted out via ``pallas_use_triton=False``)."""
    use_triton = config.pallas_use_triton
    if use_triton and pltriton is not None:
        return pltriton.CompilerParams(
            num_warps=config.pallas_num_warps,
        )
    return None
