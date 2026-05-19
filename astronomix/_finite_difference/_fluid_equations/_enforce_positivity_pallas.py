"""Pallas backend for ``_enforce_positivity``.

Per-cell pointwise op (no halo) with ``input_output_aliases={0: 0}`` so
the floored state is written back into the input buffer.  Removes one
full-state intermediate per stage on the FD time-integrator hot path.

Mirrors ``_enforce_positivity.py``; the developer never touches this
file by hand.  See ``pallas_backend_implementation_guide.md`` §4.4 for
the pallasify recipe.
"""

import jax
import jax.numpy as jnp

from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _backend_is_pallas,
    _pallas_call_sharded,
    _pallas_compiler_params,
    pl,
)
from astronomix.option_classes.simulation_config import IDEAL_GAS, ISOTHERMAL, SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables


def _enforce_positivity_pallas_supported(state, config: SimulationConfig) -> bool:
    """Pure pointwise op — no halo, no axis branches.  Supported for
    1/2/3D, IDEAL_GAS and ISOTHERMAL, with or without MHD, as long as
    the spatial block divides the spatial dims."""
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if config.equation_of_state not in (IDEAL_GAS, ISOTHERMAL):
        return False
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if state.ndim != ndim + 1:
        return False
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    for n, b in zip(state.shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _enforce_positivity_pallas(
    conserved_state,
    config: SimulationConfig,
    gamma,
    minimum_density,
    minimum_pressure,
    registered_variables: RegisteredVariables,
):
    """In-place Pallas positivity floor.  Same arithmetic as the native
    ``_enforce_positivity``; the input buffer is donated via the
    Pallas-side ``input_output_aliases={0: 0}`` so XLA reuses it for
    the output (one full-state buffer saved per call).
    """
    assert _enforce_positivity_pallas_supported(conserved_state, config)

    # Multi-GPU: pure pointwise op, so halo=0.  Still route through the
    # shard_map wrapper so XLA does NOT all-gather the state before this
    # opaque pallas_call — the wrapper just runs the kernel locally on
    # each shard.
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)

    def _local(state_local):
        return _enforce_positivity_pallas_local(
            state_local,
            config,
            gamma,
            minimum_density,
            minimum_pressure,
            registered_variables,
        )

    return _pallas_call_sharded(
        _local,
        state_inputs=(conserved_state,),
        halo=(0,) * ndim,
        block_shape=block_shape[:ndim],
    )


def _enforce_positivity_pallas_local(
    conserved_state,
    config: SimulationConfig,
    gamma,
    minimum_density,
    minimum_pressure,
    registered_variables: RegisteredVariables,
):
    """Single-shard kernel build.  Called either directly (single device)
    or once per device from inside ``shard_map`` (multi-device).  The
    pallas_call's ``out_shape`` and ``grid`` are recomputed from the
    *local* ``conserved_state.shape`` so this works for both."""

    is_mhd = config.mhd
    is_ideal = (config.equation_of_state == IDEAL_GAS)
    ndim = int(config.dimensionality)
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx_blk, by_blk, bz_blk = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx_blk, ny // by_blk, nz // bz_blk)

    DENSITY = int(registered_variables.density_index)
    if ndim == 1:
        MX = int(registered_variables.momentum_index)
    else:
        MX = int(registered_variables.momentum_index.x)
    if ndim >= 2:
        MY = int(registered_variables.momentum_index.y)
    if ndim == 3:
        MZ = int(registered_variables.momentum_index.z)
    if is_ideal:
        E = int(registered_variables.energy_index)
    if is_mhd:
        BX = int(registered_variables.magnetic_index.x)
        BY = int(registered_variables.magnetic_index.y)
        BZ = int(registered_variables.magnetic_index.z)

    if ndim == 1:
        block_shape = (nvars, bx_blk)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx_blk, by_blk)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx_blk, by_blk, bz_blk)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_in_ref, gamma_ref, rhomin_ref, pmin_ref, q_out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        if ndim == 1:
            ii = (bi * bx_blk + jnp.arange(bx_blk)) % nx
        elif ndim == 2:
            ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None]) % nx
            jj = (bj * by_blk + jnp.arange(by_blk)[None, :]) % ny
        else:
            ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
            jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
            kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        gamma = gamma_ref[()]
        gm1 = gamma - 1.0
        rhomin = rhomin_ref[()]
        pmin = pmin_ref[()]

        def read(var):
            if ndim == 1:
                return q_in_ref[var, ii]
            if ndim == 2:
                return q_in_ref[var, ii, jj]
            return q_in_ref[var, ii, jj, kk]

        rho = read(DENSITY)
        rho_floored = jnp.maximum(rho, rhomin)

        if is_ideal:
            mx = read(MX)
            v2 = (mx * mx) / (rho_floored * rho_floored)
            if ndim >= 2:
                my = read(MY)
                v2 = v2 + (my * my) / (rho_floored * rho_floored)
            if ndim == 3:
                mz = read(MZ)
                v2 = v2 + (mz * mz) / (rho_floored * rho_floored)
            energy = read(E)
            if is_mhd:
                bxv = read(BX)
                byv = read(BY)
                bzv = read(BZ)
                b2 = bxv * bxv + byv * byv + bzv * bzv
                pressure = gm1 * (energy - 0.5 * rho_floored * v2 - 0.5 * b2)
            else:
                pressure = gm1 * (energy - 0.5 * rho_floored * v2)
            pressure_floored = jnp.maximum(pressure, pmin)
            if is_mhd:
                energy_floored = pressure_floored / gm1 + 0.5 * rho_floored * v2 + 0.5 * b2
            else:
                energy_floored = pressure_floored / gm1 + 0.5 * rho_floored * v2

        # Pass-through every variable, then overwrite the ones we touched.
        for var in range(nvars):
            if var == DENSITY:
                q_out_ref[var, ...] = rho_floored
            elif is_ideal and var == E:
                q_out_ref[var, ...] = energy_floored
            else:
                q_out_ref[var, ...] = read(var)

    kwargs = {"input_output_aliases": {0: 0}}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[in_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name="enforce_positivity",
        **kwargs,
    )(
        conserved_state,
        jnp.asarray(gamma, dtype=conserved_state.dtype),
        jnp.asarray(minimum_density, dtype=conserved_state.dtype),
        jnp.asarray(minimum_pressure, dtype=conserved_state.dtype),
    )
