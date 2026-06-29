# Hand-off: field-level-inference figure + Pallas MHD WENO reverse adjoint

**Worktree:** `/export/home/lstorcks/agent-home/astronomix-refactor-port` (branch
`refactor`, pinned — do **not** checkout/pull/commit; the main node works here too).
**Node:** H200 box, shared GPUs. **Cluster rule: never run compute on CPU**, and never
grab a GPU another user holds — use `autocvd` (waits for a fully-free GPU).

This hand-off has two independent parts. **Part A** is a quick GPU run that produces a
paper figure. **Part B** is the real engineering task: give the MHD WENO Pallas kernel a
native reverse-mode adjoint so differentiable runs (inverse problems) scale past ~64³ —
the lever that makes the **256³** field-level-inference run feasible.

---

## Background: the field-level-inference experiment

`tests/field_level_inference/image_optim.py` reconstructs an **initial velocity field**
of a 3D **MHD** turbulence box (periodic, FINITE_DIFFERENCE, WENO) so that, after
evolving with turbulence OFF for `t_end = crossing_time/2` (BACKWARDS adjoint), the
**z-projected density** `sum(rho, axis=2)` matches a target image (the Astronomix logo).
Loss = MSE(projection, target). Optimizer = Adam, 150 steps. The result is saved to
`best_state.npy` (shape `(11, 64, 64, 64)` — a full 64³ MHD primitive state).

Units: code_length=3 pc, code_mass=100 M_sun, code_velocity=100 km/s. `n_h=2 cm^-3`,
`p_0=3e4 K/cm^3·k_B`, `B_0=13.5 µG/√µ0`, gamma=5/3, C_cfl=0.8. Forcing is **seeded**
(`config.random_seed`), so the pipeline is deterministic.

---

## Part A — produce the 64³ MHD 4-panel figure (volume + side screen)

**Goal figure:** one row, four panels, each a **volumetric rendering** of the 3D density
cube (pyvista `add_volume`, viridis, sigmoid opacity, composite, iso camera — the same
style as `image_optim.py`'s `plot_slices`) **with the line-of-sight projection shown as a
"screen" plane on the SIDE of the cube**, plus faint rays cube→screen. The four states:

1. initial optimized state (t=0)
2. shortly before the loss time (t = 0.85·t_end)
3. the target / loss time (t = t_end)  ← projection should look like the logo
4. shortly after (t = 1.15·t_end)

**Design is already validated** (placeholder render): `panels_test_side.png` (screen on
the side) and `panels_test.png` (screen below — rejected). These used the existing
`best_state` density replicated 4×, so all four panels look identical; **real data will
show the evolution** (turbulent → logo → dispersing).

**Scripts (ready):**
- `make_panel_snaps.py` — loads `best_state.npy`, evolves it (turbulence OFF) and writes
  `panel_snaps.npz` (4 density cubes + times + target). It recovers `t_end` **without the
  expensive forced run** by scanning evolution times for the best logo match (the velocity
  was optimized so the logo appears exactly at `t_end`). Defaults to **GPU** (`autocvd`)
  + Pallas forward. **Do not pass `--device cpu`** (cluster rule).
- `make_panels_fig.py` — loads `panel_snaps.npz`, renders the 1×4 figure. Reorients so the
  loss/projection axis (z) becomes the scene's side axis (`np.moveaxis(...,proj_axis,0)`),
  so the screen sits to the side. Global color scales across panels.

**Env split (important — two different envs):**
- **Sim** (`make_panel_snaps.py`): needs `astronomix` + jax. Use **`astx`**
  (`/export/home/lstorcks/.local/share/mamba/envs/astx/bin/python`, jax 0.10.2) **OR**
  `jf1uids` (jax 0.6.2). For pure-forward MHD either works; `astx` is the project default.
- **Render** (`make_panels_fig.py`): needs **pyvista + vtk**, only in **`jf1uids`**
  (`/export/home/lstorcks/.local/share/mamba/envs/jf1uids/bin/python`, pyvista 0.47,
  vtk 9.6). Offscreen EGL prints warnings but works (falls back, produces real images).

**Run (from `tests/field_level_inference/`, with `PYTHONPATH=<worktree root>`):**
```bash
# 1) snapshots on a free GPU (astx env):
PYTHONPATH=/export/home/lstorcks/agent-home/astronomix-refactor-port \
  .../envs/astx/bin/python make_panel_snaps.py --device gpu --out panel_snaps.npz
# 2) render (jf1uids env, has pyvista):
PYTHONPATH=/export/home/lstorcks/agent-home/astronomix-refactor-port \
  .../envs/jf1uids/bin/python make_panels_fig.py --snaps panel_snaps.npz --out panels.png
```
NOTE: `panel_snaps.npz` currently on disk is a **placeholder** (fake, from the layout
test) — overwrite it with the real run above. Tunables: `--f-pre/--f-post` (panel times),
`--gap` (cube↔screen distance), `--screen-cmap` (default magma), `--cmap` (volume).

Sanity check after the real run: panel 3's projection (`panel_snaps.npz["target_image"]`
vs the at-`t_end` cube's z-sum) should resemble the logo, matching `overview.png`'s
bottom-right.

---

## Part B — Pallas MHD WENO reverse adjoint (enables 256³)

### Why
256³ field-level inference is infeasible today because **MHD reverse-mode AD runs on the
native-JAX backward**, which is memory-bound (the "64³ ceiling"). The fix is a native
**Pallas** reverse kernel for the MHD WENO flux, exactly analogous to the **already-done
hydro** one, which measured **~17× faster + ~10× less memory** on the backward.

### Current dispatch (the precise state to change)
`astronomix/_finite_difference/_interface_fluxes/_weno.py`, `_weno_flux_axis_dispatch`
(~line 507):
- **hydro** (`_hydro_pallas_flux_supported`, line 533): when
  `differentiation_mode == BACKWARDS`, routes the backward through
  **`pallas_vjp_call(...)` → `_weno_flux_hydro_pallas_vjp_local`** (the explicit Pallas
  adjoint). ✅ fast on-GPU backward.
- **MHD** (`_mhd_pallas_flux_supported`, line 555) and **MHD-iso** (line 565): use
  **`diffable_pallas_call`** = `jax.custom_jvp` (Pallas primal, **native** tangent). Under
  reverse mode the tangent is transposed in native JAX ⇒ **native-cost backward**. ← this
  is what to replace with a `pallas_vjp_call` branch once the MHD adjoint kernel exists.

### The hydro model to replicate (all in `_weno_pallas.py`, 3039 lines)
- `_weno_hydro_flux_from_window(q_stencil, gamma, rhomin, pgmin, ncomp, num_modes)`
  (line ~185) — the **pure** per-interface WENO flux from the 6-cell window (single source
  of truth; forward kernel + adjoint both build on it).
- `_weno_hydro_flux_from_window_adjoint(...)` (line ~437) — the **fully explicit,
  hand-derived** elementwise reverse pass (NOT in-kernel `jax.vjp` — that miscompiled and
  was slow). Mirrors the forward: per-cell adjoints (primitive/flux/floored),
  `left_project_adj`, `weno_recon_adj`, face/`amx`-argmax adjoints. **This is the template.**
- `_weno_flux_hydro_pallas_local` (line ~860) — forward Pallas kernel (the `kernel(q_ref,
  ...)` at ~904).
- `_weno_flux_hydro_pallas_vjp_local` (line ~1225) — the **adjoint Pallas kernel**: each
  block gathers its 6-cell window, runs the explicit window-adjoint against the tile's flux
  cotangent, scatters. Single-device.
- `pallas_vjp_call(state, aux, *, pallas_forward, pallas_backward)` in `_pallas_helpers.py`
  — the `jax.custom_vjp` boundary (reverse-mode only; differentiates **state only**, params
  threaded as `aux` with zero cotangent — custom_vjp cannot close over traced values).

### MHD specifics (what's different from hydro)
- MHD WENO uses 8 variables and **`num_modes = 7`** (vs hydro `dim+2`); see `_weno.py`
  ~line 354. Eigen-system lives in `astronomix/_fluid_equations/_eigen_mhd.py` (1016 lines:
  `_eigen_L_row`, `_eigen_R_col`, `_eigen_lambdas`) and the **isothermal** variant
  `_eigen_mhd_iso.py` (768 lines). The MHD eigenvectors are far more involved than hydro
  (`_eigen_hydro.py`, 546 lines) — fast/slow/Alfvén modes, normalization edge cases. The
  hand-derived adjoint of the MHD `left_project`/`R_col` is the bulk of the work.
- Forward Pallas kernels already exist: `_weno_flux_mhd_pallas` and
  `_weno_flux_mhd_iso_pallas`. You need the **window function** factored out (like
  `_weno_hydro_flux_from_window`) if not already, then its explicit adjoint, then the
  adjoint Pallas kernel `_weno_flux_mhd_pallas_vjp_local`, then wire a `BACKWARDS` branch
  into `_weno_flux_axis_dispatch` for the MHD predicate.
- Decide scope: do ideal-gas MHD first (matches the experiment, which is adiabatic MHD),
  then iso-MHD.

### Validation (proven workflow from the hydro effort)
1. **Interpret mode first** (`pallas_interpret=True`, CPU lowering — *this is allowed, it's
   not a CPU compute run, it's the Pallas correctness oracle*): the window adjoint must be
   **bit-exact (~1e-15)** vs `jax.vjp` of the forward window, for the MHD component count.
   Model test: `pytests/pallas/_weno_window_adjoint_check.py` (hydro) — write an MHD analog.
2. **FD ground truth** confirms the native gradient is right.
3. **GPU/Triton**: the prior hydro saga hit a **3D axis-1 Triton miscompile on jaxlib
   0.6.2** that was **fixed by jax ≥ ~0.8** (verified 0.9.2). **`astx` is jax 0.10.2 → use
   it; do NOT validate the GPU adjoint on `jf1uids` (0.6.2)**, which still has the bug.
4. End-to-end: `jax.grad` through `time_integration` (MHD, BACKWARDS) should match the
   native backward to ~1e-7 (genuine FP-order diff — it's the exact transpose of the
   *Pallas* forward).
5. Gotcha from hydro: the Lax-Friedrichs `amx = max_k|λ_k|` fold needs the
   `prev_gets = amxs[k-1] > absl[k]` (strict) tie-break to match `lax.max`'s
   cotangent-to-second-operand rule. MHD has the same structure.

### Config knobs / pitfalls
- Pallas backend: `backend=PALLAS, pallas_block_shape=(4,4,8), pallas_use_triton=True,
  pallas_interpret=False`. Periodic BCs + N divisible by 8 → `(4,4,8)` OK (256 fine).
  Non-periodic FD would need `(4,4,4)` (N+12 divisible) — not relevant here (periodic).
- The Pallas reverse path is **single-device** and **state-only** (no d/dparams) — fine for
  this experiment (it differentiates the initial velocity, i.e. state).
- Relevant memories: `pallas-weno-native-vjp` (the full hydro-adjoint saga + the jax-version
  fix), `differentiable-runs-guidance`, `pallas-backend-fast-fd-runs`,
  `pallas-block-shape-ghost-cells`.

### Suggested sequence
1. Factor `_weno_mhd_flux_from_window` (pure) if not present; confirm forward kernel can use
   it (validate-vs-native guard stays green).
2. Hand-derive `_weno_mhd_flux_from_window_adjoint`; validate **bit-exact in interpret mode**
   (new `pytests/pallas/_weno_mhd_window_adjoint_check.py`).
3. Build `_weno_flux_mhd_pallas_vjp_local` (gather→adjoint→scatter), mirror the hydro kernel.
4. Add the `BACKWARDS` branch to `_weno_flux_axis_dispatch` for MHD; validate grad through
   `time_integration` on `astx` (jax 0.10.2), all 3 axes, 3D.
5. Re-run the field-level inference at 64³ MHD to confirm parity, then scale to **256³**
   (re-optimize; tune `num_checkpoints` to fit H200 memory) and regenerate the Part-A figure
   at 256³.

---

## Quick reference
- Worktree root: `/export/home/lstorcks/agent-home/astronomix-refactor-port` (branch `refactor`)
- Sim env (jax 0.10.2): `/export/home/lstorcks/.local/share/mamba/envs/astx/bin/python`
- Render env (pyvista): `/export/home/lstorcks/.local/share/mamba/envs/jf1uids/bin/python`
- Always `PYTHONPATH=<worktree root>` so `import astronomix` resolves to this worktree.
- GPUs via `autocvd` (the installed one uses `eval $(autocvd -q)`, **not** `autocvd -- cmd`).
- Figure scripts: `tests/field_level_inference/make_panel_snaps.py`, `make_panels_fig.py`
- Design refs: `panels_test_side.png` (chosen), `panels_test.png`
- Dispatch to edit: `astronomix/_finite_difference/_interface_fluxes/_weno.py:507`
- Hydro adjoint template: `astronomix/_finite_difference/_interface_fluxes/_weno_pallas.py`
  (`_weno_hydro_flux_from_window_adjoint` ~437, `_weno_flux_hydro_pallas_vjp_local` ~1225)
- MHD eigen-system: `astronomix/_fluid_equations/_eigen_mhd.py`, `_eigen_mhd_iso.py`
