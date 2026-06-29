#!/usr/bin/env python
"""Coarse-to-fine (cascadic) multiresolution field-level inference.

Generate the turbulent background once at the TOP resolution, down-filter it
spectrally (clean ideal low-pass, NOT a box filter -- a box average multiplies
by a sinc, attenuating low-k and aliasing high-k) to each ladder level, then
optimise the initial velocity coarse-to-fine:

    for N in levels:                       # e.g. 32 -> 64 -> 128 -> 256
        theta_N = prolong(theta_{prev})    # warm-start, add zero high-k modes
        optimise theta_N against L_N for K_N steps   # N-res PALLAS backward
    best_state = top-res state with velocity = prolong(theta_last)

Each level uses its OWN (correct) gradient; the coarse levels are cheap and do
most of the work, so the expensive fine levels need only a few steps (or zero =
forward-eval only).  Schedule "N:steps,..." sets steps per level; steps=0 means
prolong + evaluate only (no backward at that level).  PALLAS BACKWARDS + on-disk
compile cache.  Run in astx on a GPU.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--schedule", type=str, default="32:40,64:15,128:6,256:0",
                help="comma list of <res>:<num_steps>; steps=0 = eval-only (no backward)")
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--num-checkpoints", type=int, default=10)
ap.add_argument("--gen-res", type=int, default=0,
                help="resolution to GENERATE the turbulent background at (0 = top level). "
                     "Generating at the top is prohibitive at 256³ (the gen runs a long "
                     "physical time); generate at e.g. 128 and spectrally prolong the "
                     "background up to higher eval-only levels.")
ap.add_argument("--out", type=str, default=None)
args = ap.parse_args()

SCHEDULE = [(int(a.split(":")[0]), int(a.split(":")[1])) for a in args.schedule.split(",")]
LEVELS = [n for n, _ in SCHEDULE]
N_TOP = LEVELS[-1]
GEN_RES = args.gen_res or N_TOP

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
                        construct_primitive_state, time_integration, CodeUnits)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    PALLAS, BACKWARDS, FORWARDS, finalize_config,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astropy import units as u
import astropy.constants as c

from _spectral_ops import prolong, restrict

CU = CodeUnits(3 * u.parsec, 100 * u.M_sun, 100 * u.km / u.s)
RHO0 = (2 * c.m_p / u.cm**3).to(CU.code_density).value
P0 = (3e4 * u.K / u.cm**3 * c.k_B).to(CU.code_pressure).value
B0 = (13.5 * u.microgauss / c.mu0**0.5).to(CU.code_magnetic_field).value
DEDT = (4.3e34 * u.erg / u.s).to(CU.code_energy / CU.code_time).value
GAMMA = 5 / 3
PARAMS = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=GAMMA,
                          minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0,
                          turbulent_forcing_params=TurbulentForcingParams(energy_injection_rate=DEDT))


def make_cfg(N, forcing, backward):
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        enforce_positivity=True, donate_state=False, dimensionality=3,
        box_size=1.0, num_cells=N,
        differentiation_mode=BACKWARDS if backward else FORWARDS,
        num_checkpoints=args.num_checkpoints,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=forcing),
        boundary_settings=BoundarySettings(
            *[BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY) for _ in range(3)]),
        backend=PALLAS, pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True, pallas_interpret=False)


def build_state(cfg, rv, rho, v, p, B):
    bxb, byb, bzb = initialize_interface_fields(B[0], B[1], B[2])
    return construct_primitive_state(
        config=cfg, registered_variables=rv, density=rho,
        velocity_x=v[0], velocity_y=v[1], velocity_z=v[2], gas_pressure=p,
        magnetic_field_x=B[0], magnetic_field_y=B[1], magnetic_field_z=B[2],
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb)


def load_image(path, N):
    img = jnp.array(Image.open(path).convert("L"))
    img = 1.0 - img / 255.0
    h, w = img.shape
    ph, pw = (-h) % N, (-w) % N
    img = jnp.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    hp, wp = img.shape
    r = img.reshape(N, hp // N, N, wp // N).mean(axis=(1, 3))
    return 1.0 + (r - 0.01) / 0.99 * 0.01


def _resample(field, N):
    """Spectral restrict (down) or prolong (up) the last-3-axes to size N."""
    n = field.shape[-1]
    if N == n:
        return field
    return restrict(field, N) if N < n else prolong(field, N)


def fields_at(turb, rv, N):
    """Resample the gen-res turbulent primitive fields to level N (restrict if
    N <= gen-res, prolong if N > gen-res)."""
    di, pi, vi, mi = rv.density_index, rv.pressure_index, rv.velocity_index, rv.magnetic_index
    rho = _resample(turb[di][None], N)[0]
    p = _resample(turb[pi][None], N)[0]
    v = _resample(jnp.stack([turb[vi.x], turb[vi.y], turb[vi.z]]), N)
    B = _resample(jnp.stack([turb[mi.x], turb[mi.y], turb[mi.z]]), N)
    return rho, p, v, B


def logo_corr(proj, tgt):
    pz = proj - jnp.mean(proj); tz = tgt - jnp.mean(tgt)
    return float(jnp.sum(pz * tz) / jnp.sqrt(jnp.sum(pz * pz) * jnp.sum(tz * tz)))


def main():
    print(f"=== ladder inverse  schedule={SCHEDULE}  top={N_TOP}  gen_res={GEN_RES} ===", flush=True)
    out = args.out or f"best_state_ladder_{N_TOP}.npy"

    # --- turbulent background generated once at GEN_RES, resampled to levels ---
    cfg_gen = make_cfg(GEN_RES, forcing=True, backward=False)
    rv_top = get_registered_variables(cfg_gen)
    sh = (GEN_RES,) * 3
    z = jnp.zeros(sh)
    s0 = build_state(cfg_gen, rv_top, jnp.ones(sh) * RHO0,
                     (z, z, z), jnp.ones(sh) * P0, (jnp.ones(sh) * B0, z, z))
    cfg_gen = finalize_config(cfg_gen, s0.shape)
    t_gen = (24 * 1e4 * u.yr).to(CU.code_time).value
    t0 = time.time()
    turb = jax.block_until_ready(time_integration(s0, cfg_gen, PARAMS._replace(t_end=t_gen), rv_top))
    vi = rv_top.velocity_index
    rms = float(jnp.sqrt(jnp.mean(turb[vi.x]**2 + turb[vi.y]**2 + turb[vi.z]**2)))
    t_end = 0.5 / max(rms, 1e-12)
    inv = PARAMS._replace(t_end=t_end)
    print(f"top turb gen {time.time()-t0:.1f}s  rms={rms:.4f}  t_end={t_end:.4f}", flush=True)

    b1, b2, eps = 0.9, 0.999, 1e-8
    theta = None
    for (N, ksteps) in SCHEDULE:
        cfg = finalize_config(make_cfg(N, forcing=False, backward=True),
                              (turb.shape[0],) + (N,) * 3)
        rv = get_registered_variables(cfg)
        di = rv.density_index
        rho_N, p_N, v_N, B_N = fields_at(turb, rv_top, N)
        tgt = load_image("logo.png", N); tgt = tgt / jnp.sum(tgt) * jnp.sum(rho_N)
        theta = v_N if theta is None else prolong(theta, N)

        def loss(th):
            s = build_state(cfg, rv, rho_N, (th[0], th[1], th[2]), p_N, (B_N[0], B_N[1], B_N[2]))
            proj = jnp.sum(time_integration(s, cfg, inv, rv)[di], axis=2)
            return jnp.mean((proj - tgt) ** 2)

        def proj_of(th):
            s = build_state(cfg, rv, rho_N, (th[0], th[1], th[2]), p_N, (B_N[0], B_N[1], B_N[2]))
            return jnp.sum(time_integration(s, cfg, inv, rv)[di], axis=2)

        l0 = float(loss(theta))
        print(f"-- level N={N}  steps={ksteps}  L0={l0:.4e}  "
              f"corr0={logo_corr(proj_of(theta), tgt):.4f}", flush=True)
        if ksteps == 0:
            continue
        vg = jax.jit(jax.value_and_grad(loss))
        m = jnp.zeros_like(theta); vv = jnp.zeros_like(theta)

        @jax.jit
        def step(theta, m, vv, g, t):
            m = b1 * m + (1 - b1) * g
            vv = b2 * vv + (1 - b2) * g * g
            theta = theta - args.lr * (m / (1 - b1 ** t)) / (jnp.sqrt(vv / (1 - b2 ** t)) + eps)
            return theta, m, vv

        # Save the optimised state at this level — recoverable checkpoints for
        # long (e.g. multi-hour 128³) runs.
        def save_level(th, tag):
            st = build_state(cfg, rv, rho_N, (th[0], th[1], th[2]), p_N,
                             (B_N[0], B_N[1], B_N[2]))
            np.save(f"best_state_ladder_level{N}.npy", np.asarray(st))
            print(f"   [saved best_state_ladder_level{N}.npy @ {tag}]", flush=True)

        best = (l0, theta)
        for s in range(ksteps):
            ts = time.time()
            l, g = vg(theta)
            l = float(l)
            if l < best[0]:
                best = (l, theta)
            theta, m, vv = step(theta, m, vv, g, float(s + 1))
            if s < 2 or s % 5 == 0 or s == ksteps - 1:
                print(f"   N={N} step {s:3d}: L={l:.4e}  ({time.time()-ts:.1f}s)", flush=True)
            if s > 0 and s % 5 == 0:        # periodic checkpoint of the best so far
                save_level(best[1], f"step {s}")
        theta = best[1]
        save_level(theta, "level-end")
        print(f"-- level N={N} done  bestL={best[0]:.4e}  "
              f"corr={logo_corr(proj_of(theta), tgt):.4f}", flush=True)

    # final: top-res reconstruction (save the OPTIMISED INITIAL state)
    theta_top = theta if theta.shape[-1] == N_TOP else prolong(theta, N_TOP)
    rho_t, p_t, v_t, B_t = fields_at(turb, rv_top, N_TOP)
    cfg_fwd = finalize_config(make_cfg(N_TOP, forcing=False, backward=False), turb.shape)
    init_top = build_state(cfg_fwd, rv_top, rho_t, (theta_top[0], theta_top[1], theta_top[2]), p_t,
                           (B_t[0], B_t[1], B_t[2]))
    np.save(out, np.asarray(init_top))
    tgt_t = load_image("logo.png", N_TOP); tgt_t = tgt_t / jnp.sum(tgt_t) * jnp.sum(rho_t)
    proj_t = jnp.sum(time_integration(init_top, cfg_fwd, inv, rv_top)[rv_top.density_index], axis=2)
    print(f"TOP N={N_TOP}: logo_corr={logo_corr(proj_t, tgt_t):.4f}  "
          f"L={float(jnp.mean((proj_t-tgt_t)**2)):.4e}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
