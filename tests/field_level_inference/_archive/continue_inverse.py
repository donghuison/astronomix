#!/usr/bin/env python
"""Continue the field-level inference from a saved state (no turbulence re-gen).

Loads a full MHD state (its turbulent background + current optimised velocity),
keeps the background fixed, and runs more Adam steps on the velocity against the
logo loss at a fixed t_end.  Used to push the 128³ reconstruction further along
the slow convergence tail.  PALLAS BACKWARDS + on-disk compile cache.  Saves the
best state periodically (recoverable).  Run in astx on a GPU.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--state", type=str, required=True)
ap.add_argument("--resolution", type=int, default=128)
ap.add_argument("--t-end", type=float, required=True)
ap.add_argument("--num-steps", type=int, default=25)
ap.add_argument("--num-checkpoints", type=int, default=20)
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--out", type=str, default=None)
args = ap.parse_args()

from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
import numpy as np
import jax
_CACHE = os.path.expanduser("~/.cache/astronomix_jax_cache")
jax.config.update("jax_compilation_cache_dir", _CACHE)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 30)
import jax.numpy as jnp
from PIL import Image

from astronomix import (SimulationConfig, SimulationParams, get_registered_variables,
                        time_integration, CodeUnits)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    PALLAS, BACKWARDS, finalize_config,
)
from astronomix.option_classes.simulation_config import finalize_config as _fc  # noqa: F401
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig
from astropy import units as u
import astropy.constants as c

CU = CodeUnits(3 * u.parsec, 100 * u.M_sun, 100 * u.km / u.s)
RHO0 = (2 * c.m_p / u.cm**3).to(CU.code_density).value
P0 = (3e4 * u.K / u.cm**3 * c.k_B).to(CU.code_pressure).value


def load_image(path, N):
    img = jnp.array(Image.open(path).convert("L")); img = 1.0 - img / 255.0
    h, w = img.shape; ph, pw = (-h) % N, (-w) % N
    img = jnp.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    hp, wp = img.shape
    r = img.reshape(N, hp // N, N, wp // N).mean(axis=(1, 3))
    return 1.0 + (r - 0.01) / 0.99 * 0.01


def logo_corr(proj, tgt):
    pz = proj - jnp.mean(proj); tz = tgt - jnp.mean(tgt)
    return float(jnp.sum(pz * tz) / jnp.sqrt(jnp.sum(pz * pz) * jnp.sum(tz * tz)))


def main():
    N = args.resolution
    out = args.out or args.state
    cfg = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        enforce_positivity=True, donate_state=False, dimensionality=3,
        box_size=1.0, num_cells=N, differentiation_mode=BACKWARDS,
        num_checkpoints=args.num_checkpoints,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=False),
        boundary_settings=BoundarySettings(
            *[BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY) for _ in range(3)]),
        backend=PALLAS, pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True, pallas_interpret=False)
    rv = get_registered_variables(cfg)
    params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5 / 3,
                              minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0,
                              t_end=args.t_end)

    base = jnp.asarray(np.load(args.state))   # full state: bg + current velocity
    cfg = finalize_config(cfg, base.shape)
    di, vi = rv.density_index, rv.velocity_index
    tgt = load_image("logo.png", N); tgt = tgt / jnp.sum(tgt) * jnp.sum(base[di])
    print(f"=== continue N={N} t_end={args.t_end} steps={args.num_steps} "
          f"from {args.state} ===", flush=True)

    def loss(v):
        s = base.at[vi.x].set(v[0]).at[vi.y].set(v[1]).at[vi.z].set(v[2])
        proj = jnp.sum(time_integration(s, cfg, params, rv)[di], axis=2)
        return jnp.mean((proj - tgt) ** 2)

    def proj_of(v):
        s = base.at[vi.x].set(v[0]).at[vi.y].set(v[1]).at[vi.z].set(v[2])
        return jnp.sum(time_integration(s, cfg, params, rv)[di], axis=2)

    v = jnp.stack([base[vi.x], base[vi.y], base[vi.z]])
    vg = jax.jit(jax.value_and_grad(loss))
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, args.lr
    m = jnp.zeros_like(v); vv = jnp.zeros_like(v)

    @jax.jit
    def step(v, m, vv, g, t):
        m = b1 * m + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * g * g
        v = v - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(vv / (1 - b2 ** t)) + eps)
        return v, m, vv

    def save(vbest, tag):
        s = base.at[vi.x].set(vbest[0]).at[vi.y].set(vbest[1]).at[vi.z].set(vbest[2])
        np.save(out, np.asarray(s))
        print(f"   [saved {out} @ {tag}]", flush=True)

    l0 = float(loss(v))
    print(f"start: L={l0:.4e}  corr={logo_corr(proj_of(v), tgt):.4f}", flush=True)
    best = (l0, v)
    for s in range(args.num_steps):
        ts = time.time()
        l, g = vg(v); l = float(l)
        if l < best[0]:
            best = (l, v)
        v, m, vv = step(v, m, vv, g, float(s + 1))
        if s < 2 or s % 5 == 0 or s == args.num_steps - 1:
            print(f"   step {s:3d}: L={l:.4e}  ({time.time()-ts:.1f}s)", flush=True)
        if s > 0 and s % 5 == 0:
            save(best[1], f"step {s}")
    save(best[1], "final")
    print(f"DONE  bestL={best[0]:.4e}  corr={logo_corr(proj_of(best[1]), tgt):.4f}", flush=True)


if __name__ == "__main__":
    main()
