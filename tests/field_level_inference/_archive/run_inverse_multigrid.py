#!/usr/bin/env python
"""Multiresolution (coarse-to-fine) field-level inference prototype.

Tests the idea: optimise the LOW-k components of the initial velocity (a coarse
control theta at N_lo) while driving BOTH a coarse (N_lo) and a fine (N_hi)
logo loss, with the fine-resolution gradient applied to the coarse control
*exactly* via autodiff through a differentiable spectral prolongation P:

    control     theta            (3, N_lo, N_lo, N_lo)
    fine vel    u_hi_highk + P(theta)        -> N_hi sim -> L_high
    coarse vel  theta                        -> N_lo sim -> L_low
    objective   L_high(theta) + lam * L_low(theta)

Because P is linear+differentiable, jax.grad(L_high o P) = P^T (grad_v L_high) =
the fine gradient restricted to the coarse modes -- the principled "apply the
high-res gradient to the low-res control".  Both sims use the PALLAS BACKWARDS
adjoint.  Run in astx on a GPU.  Prints L_high and L_low each step so you can
see whether they fall together.
"""
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("--n-lo", type=int, default=32)
ap.add_argument("--n-hi", type=int, default=64)
ap.add_argument("--num-steps", type=int, default=25)
ap.add_argument("--mode", type=str, default="proxy", choices=["proxy", "coupled"],
                help="proxy: optimise L_low (cheap coarse backward) only, MONITOR "
                     "L_high with a forward-only high-res sim (no high-res backward) "
                     "-- tests whether the filtered low-res gradient also lowers the "
                     "high-res loss.  coupled: optimise L_high(Pθ)+λ·L_low(θ) with the "
                     "full high-res backward each step.")
ap.add_argument("--monitor-every", type=int, default=2,
                help="proxy mode: evaluate the high-res forward loss every N steps.")
ap.add_argument("--lam", type=float, default=1.0, help="weight on the coarse loss")
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--num-checkpoints", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
import os
import numpy as np
import jax
# Persistent on-disk compilation cache: the MHD Pallas vjp kernel is a large
# Triton kernel that takes ~9 min/axis to compile.  Without a cache every run
# pays it again; with the cache it is a true one-time cost reused across runs
# and sessions (only entries whose compile took >30 s are stored).
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
    PALLAS, BACKWARDS, finalize_config,
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


def load_image(path, height, width):
    img = jnp.array(Image.open(path).convert("L"))
    img = 1.0 - img / 255.0
    h, w = img.shape
    pad = (((-h) % height) // 2, ((-h) % height) - ((-h) % height) // 2,
           ((-w) % width) // 2, ((-w) % width) - ((-w) % width) // 2)
    img = jnp.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])))
    hp, wp = img.shape
    result = img.reshape(height, hp // height, width, wp // width).mean(axis=(1, 3))
    return 1.0 + (result - 0.01) / (1.0 - 0.01) * 0.01


CU = CodeUnits(3 * u.parsec, 100 * u.M_sun, 100 * u.km / u.s)
RHO0 = (2 * c.m_p / u.cm**3).to(CU.code_density).value
P0 = (3e4 * u.K / u.cm**3 * c.k_B).to(CU.code_pressure).value
B0 = (13.5 * u.microgauss / c.mu0**0.5).to(CU.code_magnetic_field).value
DEDT = (4.3e34 * u.erg / u.s).to(CU.code_energy / CU.code_time).value
GAMMA = 5 / 3


def make_cfg(N, forcing, backward):
    cfg = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        enforce_positivity=True, donate_state=False, dimensionality=3,
        box_size=1.0, num_cells=N,
        differentiation_mode=BACKWARDS if backward else 0,
        num_checkpoints=args.num_checkpoints,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=forcing),
        boundary_settings=BoundarySettings(
            *[BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY) for _ in range(3)]),
        backend=PALLAS, pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True, pallas_interpret=False)
    return cfg


PARAMS = SimulationParams(
    C_cfl=0.8, dt_max=0.1, gamma=GAMMA,
    minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0,
    turbulent_forcing_params=TurbulentForcingParams(energy_injection_rate=DEDT))


def build_state(cfg, rv, rho, vx, vy, vz, p, Bx, By, Bz):
    bxb, byb, bzb = initialize_interface_fields(Bx, By, Bz)
    return construct_primitive_state(
        config=cfg, registered_variables=rv, density=rho,
        velocity_x=vx, velocity_y=vy, velocity_z=vz, gas_pressure=p,
        magnetic_field_x=Bx, magnetic_field_y=By, magnetic_field_z=Bz,
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb)


def main():
    Nlo, Nhi = args.n_lo, args.n_hi
    print(f"=== multigrid inverse  Nlo={Nlo} Nhi={Nhi}  lam={args.lam} "
          f"steps={args.num_steps}  jax={jax.__version__} ===", flush=True)

    # --- high-res turbulent background (forced run) ---
    cfg_gen = make_cfg(Nhi, forcing=True, backward=False)
    rv = get_registered_variables(cfg_gen)
    sh = (Nhi,) * 3
    z = jnp.zeros(sh)
    s0 = build_state(cfg_gen, rv, jnp.ones(sh) * RHO0, z, z, z, jnp.ones(sh) * P0,
                     jnp.ones(sh) * B0, z, z)
    cfg_gen = finalize_config(cfg_gen, s0.shape)
    t_gen = (24 * 1e4 * u.yr).to(CU.code_time).value
    t0 = time.time()
    turb = jax.block_until_ready(time_integration(s0, cfg_gen, PARAMS._replace(t_end=t_gen), rv))
    vi, di, pi = rv.velocity_index, rv.density_index, rv.pressure_index
    mi = rv.magnetic_index
    rms = float(jnp.sqrt(jnp.mean(turb[vi.x]**2 + turb[vi.y]**2 + turb[vi.z]**2)))
    t_end = 0.5 * (1.0 / max(rms, 1e-12))
    print(f"turb gen {time.time()-t0:.1f}s  rms={rms:.4f}  t_end={t_end:.4f}", flush=True)

    # high-res primitive fields
    rho_hi, p_hi = turb[di], turb[pi]
    vhi = jnp.stack([turb[vi.x], turb[vi.y], turb[vi.z]])
    Bhi = jnp.stack([turb[mi.x], turb[mi.y], turb[mi.z]])

    # --- low-res background = spectral restriction of the high-res fields ---
    rho_lo = restrict(rho_hi[None], Nlo)[0]
    p_lo = restrict(p_hi[None], Nlo)[0]
    B_lo = restrict(Bhi, Nlo)
    v_lo_bg = restrict(vhi, Nlo)

    cfg_hi = finalize_config(make_cfg(Nhi, forcing=False, backward=True), turb.shape)
    cfg_hi_fwd = finalize_config(make_cfg(Nhi, forcing=False, backward=False), turb.shape)
    rv_hi = rv
    s_lo_probe = build_state(make_cfg(Nlo, forcing=False, backward=True),
                             get_registered_variables(make_cfg(Nlo, False, True)),
                             rho_lo, v_lo_bg[0], v_lo_bg[1], v_lo_bg[2], p_lo,
                             B_lo[0], B_lo[1], B_lo[2])
    cfg_lo = finalize_config(make_cfg(Nlo, forcing=False, backward=True), s_lo_probe.shape)
    rv_lo = get_registered_variables(cfg_lo)
    inv_hi = PARAMS._replace(t_end=t_end)
    inv_lo = PARAMS._replace(t_end=t_end)

    # logo targets per resolution (normalised to each background's total mass)
    tgt_hi = load_image("logo.png", Nhi, Nhi); tgt_hi = tgt_hi / jnp.sum(tgt_hi) * jnp.sum(rho_hi)
    tgt_lo = load_image("logo.png", Nlo, Nlo); tgt_lo = tgt_lo / jnp.sum(tgt_lo) * jnp.sum(rho_lo)

    # fixed high-k part of the fine velocity; control replaces the low-k part
    u_hi_highk = vhi - prolong(restrict(vhi, Nlo), Nhi)

    def L_high_at(cfg, theta):
        v = u_hi_highk + prolong(theta, Nhi)
        s = build_state(cfg, rv_hi, rho_hi, v[0], v[1], v[2], p_hi, Bhi[0], Bhi[1], Bhi[2])
        proj = jnp.sum(time_integration(s, cfg, inv_hi, rv_hi)[di], axis=2)
        return jnp.mean((proj - tgt_hi) ** 2)

    def L_low(theta):
        s = build_state(cfg_lo, rv_lo, rho_lo, theta[0], theta[1], theta[2], p_lo,
                        B_lo[0], B_lo[1], B_lo[2])
        proj = jnp.sum(time_integration(s, cfg_lo, inv_lo, rv_lo)[di], axis=2)
        return jnp.mean((proj - tgt_lo) ** 2)

    theta = restrict(vhi, Nlo)  # init: coarse part of the turbulent velocity
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, args.lr
    m = jnp.zeros_like(theta); vv = jnp.zeros_like(theta)

    @jax.jit
    def step(theta, m, vv, g, t):
        m = b1 * m + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * g * g
        theta = theta - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(vv / (1 - b2 ** t)) + eps)
        return theta, m, vv

    if args.mode == "coupled":
        def total(theta):
            lh = L_high_at(cfg_hi, theta); ll = L_low(theta)
            return lh + args.lam * ll, (lh, ll)
        vg = jax.jit(jax.value_and_grad(total, has_aux=True))
        for s in range(args.num_steps):
            ts = time.time()
            (_, (lh, ll)), g = vg(theta)
            theta, m, vv = step(theta, m, vv, g, float(s + 1))
            print(f"step {s:3d}: L_high={float(lh):.4e}  L_low={float(ll):.4e}  "
                  f"({time.time()-ts:.1f}s)", flush=True)
    else:  # proxy: gradient from the CHEAP coarse backward; high-res loss is
        # forward-only monitoring (no high-res backward in this stage).
        vg_low = jax.jit(jax.value_and_grad(L_low))
        L_high_fwd = jax.jit(lambda th: L_high_at(cfg_hi_fwd, th))
        for s in range(args.num_steps):
            ts = time.time()
            ll, g = vg_low(theta)
            lh_str = ""
            if s % args.monitor_every == 0 or s == args.num_steps - 1:
                lh_str = f"  L_high(fwd)={float(L_high_fwd(theta)):.4e}"
            theta, m, vv = step(theta, m, vv, g, float(s + 1))
            print(f"step {s:3d}: L_low={float(ll):.4e}{lh_str}  "
                  f"({time.time()-ts:.1f}s)", flush=True)

    np.save(f"theta_multigrid_{Nlo}.npy", np.asarray(theta))
    print("wrote theta_multigrid", flush=True)


if __name__ == "__main__":
    main()
