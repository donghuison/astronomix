---
name: pallasify
description: Translate a native-JAX numerical kernel in this astronomix codebase into the matching Pallas backend kernel, following the conventions in `pallas_backend_implementation_guide.md`. Use when the user asks to "pallasify", "port to Pallas", "compile to Pallas", "regenerate the Pallas backend for X", or has just modified a `*_native` function and wants the `*_pallas` sibling refreshed. Pure code-translation: native JAX in, Pallas kernel out — the developer does NOT touch `_pallas` files by hand.
---

# pallasify — Compile native JAX to a Pallas backend kernel

This skill is the developer-loop mechanism for the astronomix Pallas
backend: **a developer writes / tweaks a native JAX function, runs this
skill, and the matching Pallas kernel in `_pallas` files is regenerated
to match**.  The Pallas side is treated as compiled output that mirrors
the native side — the user should not edit Pallas files by hand.

The full design rationale and the patterns this skill follows are in
`pallas_backend_implementation_guide.md` at the repo root.  Read it
before doing anything non-trivial; everything below assumes you have.

---

## When you are invoked

The user has either:

1. **Modified an existing native function** (e.g. `_weno_flux_x_native`,
   `_evolve_state_along_axis`, a Riemann solver, a reconstruction step)
   and wants its `_pallas` sibling refreshed; or
2. **Asked for a new native function to be pallasified**.

You will need to:

1. Identify the native function (the user names it; if ambiguous, ask).
2. Locate its `_pallas` module sibling (same-directory `_*_pallas.py`,
   or the top-level FV `_pallas_evolve.py`).  If none exists yet, create
   one — `astronomix/<package>/<subpackage>/_<kernel>_pallas.py`.
3. Either generate a new Pallas kernel from scratch or rewrite an
   existing one so its math, control flow and dispatch logic match the
   native version exactly.
4. Wire the dispatch in the *native* file at the bottom (lazy import
   pattern to avoid circular imports — see step 6).
5. Validate against the matching native version on the cheapest
   relevant test (`tests/pallas/sedov3D.py`,
   `pytests/mhd/alfven_wave3D.py`, or a small standalone smoke test).
   PALLAS and NATIVE must match to single-precision rounding.

---

## File layout to use

```
astronomix/
  _pallas_helpers.py                     ← block-shape, compiler-params, pl, pltriton
  _finite_difference/
    _interface_fluxes/
      _weno.py                           ← native + dispatcher (your edits go here)
      _weno_pallas.py                    ← generated; do not hand-edit
    _time_integrators/
      _ssprk.py                          ← native + integrator + dispatcher
      _ssprk_pallas.py                   ← generated; do not hand-edit
  _finite_volume/
    _state_evolution/
      evolve_state.py                    ← native + dispatcher
      _pallas_evolve.py                  ← generated; do not hand-edit
```

A `_pallas` module must:
- Import shared helpers from `astronomix._pallas_helpers`
  (`_as_3tuple_block_shape`, `_backend_is_pallas`,
  `_pallas_compiler_params`, `pl`, `pltriton`).
- **Not** import from its native sibling at module load — use lazy
  imports inside the function body if a native fallback is needed.  The
  native module imports from the Pallas module at the **bottom** of
  its file.
- Expose a `_<flavour>_pallas_supported(state, config)` predicate that
  is a plain Python function and gates the Pallas kernel.
- Expose a `_<flavour>_indices_for_axis(config, registered_variables,
  axis)` if the algorithm uses characteristic projection / per-axis
  component permutation.
- Expose the actual kernel (`_*_pallas`) which:
  - takes the same arguments as the native function plus `axis` (if
    per-axis) and any accumulator buffers;
  - asserts/short-circuits on `_supported(...) is False`;
  - uses `pl.BlockSpec`, `pl.program_id`, modular indexing for the
    stencil reads, and `input_output_aliases={0:0}` for any
    accumulator buffer that is reused across axes.

---

## Decide first: pallasify, rewrite, or leave alone

Not every native op should become a Pallas kernel. Before opening the
recipe, classify the target:

- **Heavy stencil with characteristic projection** (WENO flux, FV
  reconstruct+Riemann, CT stages) → pallasify with the recipe below.
- **Pure pointwise op** (positivity floor, EOS conversion, source
  term) → pallasify, but use the *pointwise leaf-op* shape in §4b: no
  halo, `input_output_aliases={0:0}`, pass-through + selective
  overwrite. Cheap to compile, removes one full-state intermediate per
  call.
- **Op whose cost is a temp buffer, not arithmetic** (e.g. the MHD CFL
  estimator allocating a full-state 7-eigenvalue array) → **do not
  pallasify**. Rewrite it in pure JAX without the intermediate. The
  hot example here is `_cfl_time_step_fd_mhd_fast` —
  `max(|v_d| + c_fast_d)` per cell, no Pallas needed.
- **Multi-stage native pipeline with halo > ~6 if fused** (e.g.
  CT: modified flux → edge EMF → curl → cell-center update) → split
  into one Pallas kernel per stage with bounded halo each. A single
  fused kernel with halo ~8 made Triton lowering hang for >5 min;
  three kernels with halo ≤2 each compile in <1 s apiece.
- **Optional / opt-in Pallas variants**: if the pallasified version
  saves <5% runtime at production scale but costs >5 s extra
  compile (the CT case at production-N), expose a `config.<flag>`
  and have `_supported(...)` short-circuit on `if not config.<flag>:
  return False`. Predicate machinery is the right place to make a
  Pallas backend "available but off by default", not just to gate on
  hardware capability. See `pallas_ct` in `simulation_config.py`.

If the decision is "leave alone", stop here. Document the reasoning in
a one-line comment in the dispatcher so the next pass doesn't re-open
the question.

---

## Translation recipe

For each native function being pallasified:

### 1. Read the native body and classify each operation

Walk the native body top-to-bottom and tag each statement:

- **stencil read**: `_shift(x, k, axis=a)` or `jnp.roll(x, k, axis=a)`
  — becomes `q_ref[var, (ii + k) % nx, jj, kk]` (or analogous for the
  active axis) inside the kernel.  `axis` in the kernel is the *normal*
  direction; for off-axis variants, do an axis-aware
  `local_indices` permutation (see `_mhd_indices_for_axis` for the 8-var
  MHD example) rather than physical transpose+swap.
- **pointwise op**: `+`, `*`, `jnp.maximum`, `jnp.where`, `jnp.sqrt`,
  etc. — copy 1:1 into the kernel.
- **eos/eigenstructure call** (`primitive_state_from_conserved`,
  `_eigen_L_row_*`, `_eigen_R_col_*`, `_eigen_lambdas_*`,
  `_calculate_limited_gradients`, `_riemann_solver`, …) — **inline** the
  body as a kernel-local closure.  These calls cannot stay as function
  calls inside a Pallas kernel; their bodies must be expanded so every
  intermediate is per-tile compute.  `jax.lax.switch(mode, [f0, …, fk])`
  becomes `if mode == 0: … elif mode == 1: …` (Python-time dispatch,
  compiled away).
- **whole-state operation** (`F = _euler_flux(state, ...)`,
  `flux_minus_shift`, …) — pull the per-cell expression out of the
  whole-state function and replicate it per-tile.  If the function is
  used elsewhere, leave the native function alone — just hand-mirror
  its body inside the kernel.
- **boundary handler call** — for periodic boundaries the kernel's
  modular indexing handles them for free; the dispatcher only calls the
  native boundary handler ahead of the Pallas kernel when
  `config.boundary_handling == GHOST_CELLS`.

### 2. Build the kernel skeleton

Start from this template (3-D variant; 1-D / 2-D drop the trailing
indices analogously — see `_weno_flux_hydro_pallas` for the
multi-dim form):

```python
def _flavour_pallas_supported(state, config) -> bool:
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    # …add flavour-specific gates: equation_of_state, mhd, ndim, dtype…
    if jax.config.jax_enable_x64 and not bool(getattr(config, "pallas_interpret", False)):
        return False  # Triton-x64 caveat — see guide §4
    block_shape = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)
    for n, b in zip(state.shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _flavour_indices_for_axis(config, registered_variables, axis):
    """Local component order for axis-aware kernels."""
    …  # axis=0 → (density, p_normal=mx, …); axis=1 → swap x/y; axis=2 → swap x/z.


def _flavour_pallas_kernel(state, …, *, axis):
    if not _flavour_pallas_supported(state, config):
        from astronomix.<…>._<…> import _flavour_native_x, _flavour_native_y, _flavour_native_z  # lazy
        return [_flavour_native_x, _flavour_native_y, _flavour_native_z][axis](state, …)

    ndim = int(config.dimensionality)
    nvars = int(state.shape[0])
    spatial_shape = tuple(int(x) for x in state.shape[1:])
    nx, ny, nz = (spatial_shape + (1, 1))[:3]
    bx, by, bz = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _flavour_indices_for_axis(config, registered_variables, axis)
    ncomp = len(local_indices)
    # …flavour constants (num_modes, epsilon, tiny, b_eps if MHD)…

    if ndim == 3:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_state_spec = pl.BlockSpec(state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    # …1-D / 2-D variants…
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, *scalar_refs, out_ref):
        bi = pl.program_id(0); bj = pl.program_id(1); bk = pl.program_id(2)
        # Modular index arrays — periodic BC for free.
        ii = (bi*bx + jnp.arange(bx)[:, None, None]) % nx
        jj = (bj*by + jnp.arange(by)[None, :, None]) % ny
        kk = (bk*bz + jnp.arange(bz)[None, None, :]) % nz
        # Scalars: ref[()] once at the top.
        gamma = gamma_ref[()]
        …

        def q_at(var_index, offset):
            if axis == 0: return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1: return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        # Inline what the native function does — closures for primitive_from_q,
        # floored_cell, flux_from_q, left_project, add_right_correction, …
        …

        for var in range(nvars):
            out_ref[var, …] = …  # write every conserved slot

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(state.shape, state.dtype),
        grid=grid,
        in_specs=[in_state_spec, scalar_spec, …],
        out_specs=out_spec,
        interpret=bool(getattr(config, "pallas_interpret", False)),
        name=f"flavour_axis_{axis}",
        **({"compiler_params": _pallas_compiler_params(config)} if _pallas_compiler_params(config) else {}),
    )(state, jnp.asarray(scalar_value, dtype=state.dtype), …)
```

### 3. Translate the native math

For each kernel-local closure that mirrors a native helper:

- `_shift(arr, k, axis=a)` reads → `q_at(var, k)` with the appropriate
  axis-conditional indexing.
- `state[index]` → take `index` from `local_indices` so the same kernel
  body works for any axis.
- `jnp.einsum('nxyz,nxyz->xyz', L_row, F)` → `sum_i L[i] * F[i]`
  per-tile, written as a Python sum over the 5–8 components.
- `jax.lax.switch(mode, [col_0, …, col_k])` →
  `if mode == 0: R = (…) elif mode == 1: R = (…) …` (Python-static).
- `jax.lax.cond(pred, t, f)` → `jnp.where(pred, t_value, f_value)` if
  you can predicate both sides; otherwise expand to a `jnp.where`
  multi-line equivalent.
- Whole-state allocations like `S = jnp.zeros_like(state)` followed by
  scattered writes → just emit zeros into `out_ref[var, …] = 0.0` at
  the top of the kernel and overwrite per local index.

### 4. Hook up `input_output_aliases` for any accumulator

Any kernel that gets called once per axis with a running buffer (rhs,
dq, conservative_change…) should expose an `accumulator=None` kwarg.
When provided:

- Put the accumulator first in the input list of `pl.pallas_call`.
- Add `kwargs["input_output_aliases"] = {0: 0}` so XLA reuses one
  physical buffer across calls.
- Inside the kernel, read `accumulator_in_ref[var, …]` and write
  `out_ref[var, …] = scale * accumulator_in_ref[…] + new_contrib`.

`_hydro_flux_div_axis_pallas` in `_ssprk_pallas.py` is the canonical
example; `_fv_evolve_axis_pallas` shows the same trick on the FV side.

### 4b. Pointwise leaf-op shape (no halo, in-place)

For per-cell ops with no spatial dependence (positivity floor, EOS
conversion, source-term application), use this simpler shape rather
than the stencil+accumulator pattern in §4:

- Single `in_spec` for the conserved state, single `out_spec`, plus
  scalar specs for typed parameters (`gamma`, `minimum_density`, …).
- `kwargs["input_output_aliases"] = {0: 0}` so XLA reuses the input
  buffer for the output — one full-state buffer saved per call,
  every stage.
- Inside the kernel, close `ii, jj, kk` over the program ID and write a
  small `read(var)` helper. Then **pass every variable through, and
  selectively overwrite the ones the op touches**:
  ```python
  for var in range(nvars):
      if var == DENSITY:
          out_ref[var, ...] = rho_floored
      elif is_ideal and var == E:
          out_ref[var, ...] = energy_floored
      else:
          out_ref[var, ...] = read(var)
  ```
- Same `_as_3tuple_block_shape` / `_pallas_compiler_params` plumbing
  as the stencil case. The predicate is shorter — no per-axis halo
  check, just `state.ndim == ndim + 1` and block-divisibility.

`_enforce_positivity_pallas` in
`_finite_difference/_fluid_equations/` is the canonical example;
copy its skeleton for new leaf ops.

### 4c. Multi-stage pipeline: split, don't fuse

If a native pipeline computes A → B → C → D where each arrow is a
stencil of radius r, fusing the whole pipeline into one Pallas kernel
makes the effective halo `3r`. With `r=2` (typical WENO/CT spacing)
that pushes halo to 6–8, and Triton's lowering pass grinds to a halt
trying to schedule the closure graph (observed: >5 min stuck on a
single kernel).

Split at every natural intermediate. For CT we now have **three**
bounded-halo Pallas kernels in `_constrained_transport_pallas.py`
(`_ct_modified_flux_pallas`, `_ct_edge_emf_pallas`,
`_ct_curl_pallas`); each has halo ≤ 2 and compiles in well under a
second. The intermediates between kernels are real allocations, but
they're each 1/4 the size of the full state, and the alternative was
a kernel that never finished compiling.

Heuristic: target halo ≤ 4 per kernel. If your native pipeline has
more than two consecutive stencil stages, split.

### 5. Wire the dispatcher in the native file

At the **bottom** of the native file (after all native function
definitions, before the public dispatchers), add:

```python
from astronomix.<package>.<subpackage>._<flavour>_pallas import (  # noqa: E402
    _flavour_pallas_supported,
    _flavour_pallas_kernel,
)
```

and update the dispatcher to route through the Pallas kernel when the
predicate accepts:

```python
@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _flavour_flux_x(state, params, config, registered_variables):
    if _flavour_pallas_supported(state, config):
        return _flavour_pallas_kernel(state, params, config, registered_variables, axis=0)
    return _flavour_native_x(state, params, config, registered_variables)
```

The bottom-of-file import position is important: it lets the
`_pallas` module do a *lazy* import of native fallbacks from the
native file without tripping a circular import (the native names are
already bound in this module's globals by the time `_pallas` is
loaded).

### 6. Validate

Pick the cheapest meaningful regression test for the flavour:

- **FD hydro WENO** → `tests/pallas/sedov3D.py` (Pallas mode, 128³).
- **FD MHD WENO** → `pytests/mhd/alfven_wave3D.py` at N=8 or N=16,
  PALLAS vs NATIVE.
- **FV hydro** → a small periodic-box smoke test (32³ density wave is
  fine).
- **Anything else** → handcraft a 16³ or 32³ smoke test that exercises
  the path.

For *optional* Pallas variants (anything gated by a config flag like
`pallas_ct`) run an A/B/C harness — native baseline, flag off, flag
on — and report compile / warm runtime / temp memory / L1 diff vs
native side-by-side. The standalone scaffold in
`/tmp/compare_pallas_ct.py` is the working template: build three
configs from one base, run each twice (cold+warm) so compile time
falls out of the diff, then print compile/warm/iters/µs-per-iter and
L1 diff vs the baseline. Always reinstall (`python -m pip install .`)
before running — predicates are evaluated at module load and
monkey-patching them in a notebook is silently ineffective.

Acceptance criteria:

- `max|PALLAS − NATIVE|` matches to single-precision rounding (~1e-5
  relative, often much better).  For trivially smooth setups
  (uniform flow with tiny perturbation) demand machine-epsilon match.
- 5th-order or expected convergence rate preserved on a multi-N sweep
  if the native version is high-order.
- Memory analysis (`compiled.memory_analysis()`) shows the expected
  reduction (typically 30–60 %) and no regressions.

### 7. Update the guide if the flavour is new

If you added a new `_pallas` module / kernel that didn't exist before,
add a short subsection in `pallas_backend_implementation_guide.md`
§4 with the headline numbers and any known limitations.

---

## Known limitations / things to never do

- **Don't import `_*_pallas.py` symbols from the native file at the top
  of the file** — that re-introduces the circular import the
  bottom-of-file pattern fixes.  Always import Pallas symbols at the
  end of the native module.
- **Don't put `jax.lax.switch` / `jax.lax.cond` inside a Pallas
  kernel for compile-time selection** — Python `if` is what you want
  there.  `jax.lax.switch` should only show up if the decision genuinely
  needs to be runtime-dynamic (which is rare in these kernels).
- **Don't materialise a whole-state JAX array inside a Pallas
  kernel** — `jnp.zeros_like(conserved_state)`,
  `.at[idx].set(...)` chains on full arrays inside the kernel defeat
  the whole point.  Per-tile compute only.
- **Watch out for Python-float literals in `jnp.where` arms.** A bare
  `1.0` / `-1.0` / `1e-20` in the false-arm of `jnp.where` enters the
  Triton lowering as f32 regardless of the surrounding tile dtype, and
  trips a `('f64','f32')` assertion in `_truediv_lowering_rule` when
  the kernel is later run in x64.  Always derive typed scalars from
  an already-typed kernel input (e.g. `gamma`):
  ```python
  zero_typed = gamma - gamma
  one_typed = zero_typed + 1.0
  neg_one_typed = zero_typed - 1.0
  inv_sqrt_two_typed = zero_typed + (1.0 / 2.0 ** 0.5)
  ```
  Then `jnp.where(Bn >= 0.0, one_typed, neg_one_typed)` is x64-safe.
  Same trick for `b_eps` / `sqrt` floors: pass them as scalar kernel
  args with `jnp.asarray(value, dtype=state.dtype)` rather than using
  bare Python floats inside the kernel.  See
  `pallas_backend_implementation_guide.md` §4.4 for the full
  diagnosis.
- **Don't change the native function's signature when pallasifying** —
  the Pallas kernel mirrors the signature so the dispatcher in the
  native file can call either path interchangeably.
- **Don't fuse a multi-stage pipeline into one Pallas kernel if the
  effective halo exceeds ~4–6.** Triton's lowering pass scales badly
  with closure graph depth × halo. Split at intermediates (see §4c)
  instead — three sub-second kernels beat one that never finishes.
- **Don't pallasify just because it's possible.** If the runtime gain
  on a production-scale benchmark is <5% and the compile-time hit is
  >5 s, either gate it behind a config flag (`pallas_ct` is the
  precedent) or skip the port and rewrite the native op to avoid the
  intermediate buffer that motivated it (`_cfl_time_step_fd_mhd_fast`
  is the precedent — same goal, pure JAX, zero compile cost).

---

## After you finish

Report to the user:

- which native function was pallasified,
- the path to the new / updated `_pallas` module,
- the validation test you ran and the `max|PALLAS − NATIVE|` it
  produced,
- the memory / runtime delta on that test,
- anything you had to leave on the native fallback (and why — usually
  an unsupported limiter or Riemann solver, or an x64 gate).

The user should never need to open the generated `_pallas` files; if
they do, that's a sign this skill missed a translation pattern and
should be improved.
