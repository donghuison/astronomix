#!/usr/bin/env python
"""Field-level inference (initial-velocity reconstruction) on the PALLAS backend
with the native Pallas MHD reverse adjoint.

Reproduces image_optim.py's pipeline but routes the *gradient* through the GPU-
resident Pallas WENO backward (config.backend=PALLAS, differentiation_mode=
BACKWARDS) instead of the native-JAX backward, so the inverse problem scales
past the 64^3 memory wall.  Parameterised by resolution / checkpoints / steps so
the same script runs 64^3 and 256^3.

  1. forced turbulence run (seeded) -> turbulent background state
  2. t_end = crossing_time / 2 (turbulence OFF, BACKWARDS)
  3. Adam on the initial velocity, loss = MSE(z-projection, logo)
  4. save best_state_pallas_<N>.npy

Run in astx (jax 0.10.2) on a GPU (autocvd).  NOTE: the first grad step compiles
the MHD Pallas vjp kernels (~9 min/axis) — slow once, then cached.
"""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--resolution", type=int, default=64)
ap.add_argument("--backend", type=str, default="pallas", choices=["pallas", "native"])
ap.add_argument("--num-checkpoints", type=int, default=10)
ap.add_argument("--num-steps", type=int, default=150)
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--out", type=str, default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

from autocvd import autocvd
autocvd(num_gpus=1)

# ruff: noqa: E402
import numpy as np
import jax
# Persistent on-disk compilation cache so the slow (~9 min/axis) MHD Pallas vjp
# kernel compile is a true one-time cost reused across runs/sessions.
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
    PALLAS, NATIVE_JAX, BACKWARDS, finalize_config,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astropy import units as u
import astropy.constants as c


def load_image(path, height, width):
    img = jnp.array(Image.open(path).convert("L"))
    img = 1.0 - img / 255.0
    h, w = img.shape
    pad_h = (-h) % height
    pad_w = (-w) % width
    pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
    img = jnp.pad(img, pad)
    hp, wp = img.shape
    result = img.reshape(height, hp // height, width, wp // width).mean(axis=(1, 3))
    return 1.0 + (result - 0.01) / (1.0 - 0.01) * 0.01


def main():
    N = args.resolution
    gamma = 5 / 3
    box_size = 1.0
    out = args.out or f"best_state_{args.backend}_{N}.npy"
    print(f"=== inverse N={N} backend={args.backend} ckpt={args.num_checkpoints} "
          f"steps={args.num_steps} jax={jax.__version__} ===", flush=True)

    backend_kw = {}
    if args.backend == "pallas":
        backend_kw = dict(backend=PALLAS, pallas_block_shape=(4, 4, 8),
                          pallas_use_triton=True, pallas_interpret=False)
    else:
        backend_kw = dict(backend=NATIVE_JAX)

    code_length = 3 * u.parsec
    code_mass = 100 * u.M_sun
    code_velocity = 100 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)
    n_h = 2
    rho_0 = n_h * c.m_p / u.cm**3
    p_0 = 3e4 * u.K / u.cm**3 * c.k_B
    dE_dt_turb = 4.3e34 * u.erg / u.s
    B_0 = (13.5 * u.microgauss / c.mu0**0.5).to(code_units.code_magnetic_field).value

    # --- turbulence-generation config (forcing ON, FORWARDS) ---
    cfg = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        enforce_positivity=True, donate_state=False, dimensionality=3,
        box_size=box_size, num_cells=N,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=True),
        boundary_settings=BoundarySettings(
            *[BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
              for _ in range(3)]),
        **backend_kw,
    )
    rv = get_registered_variables(cfg)
    params = SimulationParams(
        C_cfl=0.8, dt_max=0.1, gamma=gamma,
        minimum_density=(1e-2 * rho_0).to(code_units.code_density).value,
        minimum_pressure=(1e-2 * p_0).to(code_units.code_pressure).value,
        turbulent_forcing_params=TurbulentForcingParams(
            energy_injection_rate=dE_dt_turb.to(
                code_units.code_energy / code_units.code_time).value),
    )

    sh = (N, N, N)
    zero = jnp.zeros(sh)
    rho = jnp.ones(sh) * rho_0.to(code_units.code_density).value
    p = jnp.ones(sh) * p_0.to(code_units.code_pressure).value
    Bx = jnp.ones(sh) * B_0
    bxb, byb, bzb = initialize_interface_fields(Bx, zero, zero)
    init = construct_primitive_state(
        config=cfg, registered_variables=rv, density=rho,
        velocity_x=zero, velocity_y=zero, velocity_z=zero, gas_pressure=p,
        magnetic_field_x=Bx, magnetic_field_y=zero, magnetic_field_z=zero,
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb)
    cfg = finalize_config(cfg, init.shape)

    t_gen = (24 * 1e4 * u.yr).to(code_units.code_time).value
    t0 = time.time()
    turb = time_integration(init, cfg, params._replace(t_end=t_gen), rv)
    turb = jax.block_until_ready(turb)
    vi = rv.velocity_index
    di = rv.density_index
    rms_v = float(jnp.sqrt(jnp.mean(turb[vi.x]**2 + turb[vi.y]**2 + turb[vi.z]**2)))
    crossing_time = box_size / max(rms_v, 1e-12)
    t_end = 0.5 * crossing_time
    print(f"turbulence gen: {time.time()-t0:.1f}s  rms_v={rms_v:.4f}  "
          f"t_cross={crossing_time:.4f}  t_end={t_end:.4f}", flush=True)

    target = load_image("logo.png", N, N)
    target = target / jnp.sum(target) * jnp.sum(turb[di])

    # --- inverse config (forcing OFF, BACKWARDS) ---
    cfg = cfg._replace(differentiation_mode=BACKWARDS,
                       num_checkpoints=args.num_checkpoints,
                       turbulent_forcing_config=TurbulentForcingConfig(
                           turbulent_forcing=False))
    inv_params = params._replace(t_end=t_end)

    def loss(velocity):
        s = turb.at[vi.x].set(velocity[0]).at[vi.y].set(velocity[1]).at[vi.z].set(velocity[2])
        fin = time_integration(s, cfg, inv_params, rv)
        proj = jnp.sum(fin[di], axis=2)
        return jnp.mean((proj - target) ** 2)

    v = jnp.stack([turb[vi.x], turb[vi.y], turb[vi.z]])
    grad_loss = jax.jit(jax.value_and_grad(loss))

    # Adam (optax.adam defaults: b1=0.9, b2=0.999, eps=1e-8) — implemented inline
    # so the script has no optax dependency (not installed in astx).
    b1, b2, adam_eps, lr = 0.9, 0.999, 1e-8, args.lr
    m = jnp.zeros_like(v)
    vv = jnp.zeros_like(v)

    @jax.jit
    def adam_step(v, m, vv, g, t):
        m = b1 * m + (1.0 - b1) * g
        vv = b2 * vv + (1.0 - b2) * g * g
        mhat = m / (1.0 - b1 ** t)
        vhat = vv / (1.0 - b2 ** t)
        v = v - lr * mhat / (jnp.sqrt(vhat) + adam_eps)
        return v, m, vv

    def logo_corr(vel):
        s = turb.at[vi.x].set(vel[0]).at[vi.y].set(vel[1]).at[vi.z].set(vel[2])
        proj = jnp.sum(time_integration(s, cfg, inv_params, rv)[di], axis=2)
        pz = proj - jnp.mean(proj); tz = target - jnp.mean(target)
        return float(jnp.sum(pz * tz) / jnp.sqrt(jnp.sum(pz*pz) * jnp.sum(tz*tz)))

    best = (np.inf, v)
    for step in range(args.num_steps):
        ts = time.time()
        l, g = grad_loss(v)
        l = float(l)
        if l < best[0]:           # l is the loss at the current (pre-update) v
            best = (l, v)
        v, m, vv = adam_step(v, m, vv, g, float(step + 1))
        if step < 3 or step % 10 == 0 or step == args.num_steps - 1:
            print(f"step {step:3d}: loss={l:.6e}  ({time.time()-ts:.1f}s)", flush=True)

    best_v = best[1]
    best_state = turb.at[vi.x].set(best_v[0]).at[vi.y].set(best_v[1]).at[vi.z].set(best_v[2])
    np.save(out, np.asarray(best_state))
    print(f"best loss={best[0]:.6e}  logo_corr={logo_corr(best_v):.4f}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
