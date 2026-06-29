#!/usr/bin/env python
"""Resolution-transfer diagnostic: take the GOOD 64³ reconstruction
(best_state_pallas_64.npy, corr~0.94), spectrally prolong its fields to higher
resolution, forward-evolve, and measure the best logo correlation (small
time-scan).  Quantifies how the resolution-specific inverse solution degrades
64 -> 128 -> 256 -- i.e. whether any signal survives the transfer (sets
expectations for the proxy-across-gap).  Run in astx on a GPU.
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--state", type=str, default="best_state_pallas_64.npy")
ap.add_argument("--resolutions", type=str, default="64,128,256")
ap.add_argument("--t-center", type=float, default=1.21)
ap.add_argument("--scan-frac", type=float, default=0.25)
ap.add_argument("--scan-n", type=int, default=11)
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
                        construct_primitive_state, time_integration, CodeUnits)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    PALLAS, FORWARDS, finalize_config,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig
from astropy import units as u
import astropy.constants as c
from _spectral_ops import prolong, restrict

CU = CodeUnits(3 * u.parsec, 100 * u.M_sun, 100 * u.km / u.s)
RHO0 = (2 * c.m_p / u.cm**3).to(CU.code_density).value
P0 = (3e4 * u.K / u.cm**3 * c.k_B).to(CU.code_pressure).value
GAMMA = 5 / 3
PARAMS = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=GAMMA,
                          minimum_density=1e-2 * RHO0, minimum_pressure=1e-2 * P0)


def make_cfg(N):
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        enforce_positivity=True, donate_state=False, dimensionality=3,
        box_size=1.0, num_cells=N, differentiation_mode=FORWARDS,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=False),
        boundary_settings=BoundarySettings(
            *[BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY) for _ in range(3)]),
        backend=PALLAS, pallas_block_shape=(4, 4, 8), pallas_use_triton=True)


def load_image(path, N):
    img = jnp.array(Image.open(path).convert("L")); img = 1.0 - img / 255.0
    h, w = img.shape; ph, pw = (-h) % N, (-w) % N
    img = jnp.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    hp, wp = img.shape
    r = img.reshape(N, hp // N, N, wp // N).mean(axis=(1, 3))
    return 1.0 + (r - 0.01) / 0.99 * 0.01


def corr(proj, tgt):
    pz = proj - jnp.mean(proj); tz = tgt - jnp.mean(tgt)
    return float(jnp.sum(pz * tz) / jnp.sqrt(jnp.sum(pz * pz) * jnp.sum(tz * tz)))


def resample(field, N):
    n = field.shape[-1]
    return field if n == N else (restrict(field, N) if N < n else prolong(field, N))


def main():
    st = jnp.asarray(np.load(args.state))
    n0 = st.shape[-1]
    rv0 = get_registered_variables(make_cfg(n0))
    di, vi, pi, mi = rv0.density_index, rv0.velocity_index, rv0.pressure_index, rv0.magnetic_index
    rho0, p0 = st[di], st[pi]
    v0 = jnp.stack([st[vi.x], st[vi.y], st[vi.z]])
    B0 = jnp.stack([st[mi.x], st[mi.y], st[mi.z]])
    print(f"loaded {args.state} n0={n0}", flush=True)

    for N in [int(x) for x in args.resolutions.split(",")]:
        cfg = make_cfg(N); rv = get_registered_variables(cfg)
        rho = resample(rho0[None], N)[0]; p = resample(p0[None], N)[0]
        v = resample(v0, N); B = resample(B0, N)
        bxb, byb, bzb = initialize_interface_fields(B[0], B[1], B[2])
        s = construct_primitive_state(
            config=cfg, registered_variables=rv, density=rho, gas_pressure=p,
            velocity_x=v[0], velocity_y=v[1], velocity_z=v[2],
            magnetic_field_x=B[0], magnetic_field_y=B[1], magnetic_field_z=B[2],
            interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb,
            interface_magnetic_field_z=bzb)
        cfg = finalize_config(cfg, s.shape)
        tgt = load_image("logo.png", N); tgt = tgt / jnp.sum(tgt) * jnp.sum(rho)
        lo, hi = (1 - args.scan_frac) * args.t_center, (1 + args.scan_frac) * args.t_center
        best = (-2.0, 0.0)
        for t in np.linspace(lo, hi, args.scan_n):
            fin = time_integration(s, cfg, PARAMS._replace(t_end=float(t)), rv)
            cval = corr(jnp.sum(fin[di], axis=2), tgt)
            if cval > best[0]:
                best = (cval, float(t))
        print(f"  N={N:4d}: best logo_corr={best[0]:+.4f} at t={best[1]:.3f}  "
              f"(rho prolong range [{float(rho.min()):.4g},{float(rho.max()):.4g}])", flush=True)


if __name__ == "__main__":
    main()
