#!/usr/bin/env python
"""Diagnostic: evolve best_state across a fine grid of times, measure how well
the z-projection matches the logo (correlation AND mse), and dump a filmstrip
of projections so the logo's appearance/disappearance time is directly visible.

The MSE used by make_panel_snaps is a poor logo detector (the logo contrast is
~1%, so MSE is dominated by the mean amplitude).  Correlation with the
mean-subtracted target is the right detector.  Run in astx on a GPU.
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--resolution", type=int, default=64)
ap.add_argument("--state", type=str, default="best_state.npy")
ap.add_argument("--lo", type=float, default=0.3)
ap.add_argument("--hi", type=float, default=2.2)
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--out", type=str, default="_diag_scan.npz")
args = ap.parse_args()

from autocvd import autocvd
autocvd(num_gpus=1)

# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image

from astronomix import (SimulationConfig, SimulationParams, get_registered_variables,
                        construct_primitive_state, time_integration, CodeUnits)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    PALLAS, finalize_config,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
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


def ncorr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / jnp.sqrt((a * a).sum() * (b * b).sum()))


def main():
    resolution = args.resolution
    gamma = 5 / 3
    box_size = 1.0
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        enforce_positivity=True, donate_state=False, dimensionality=3,
        box_size=box_size, num_cells=resolution,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=False),
        boundary_settings=BoundarySettings(
            *[BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY,
                                 right_boundary=PERIODIC_BOUNDARY) for _ in range(3)]),
        backend=PALLAS, pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True, pallas_interpret=False,
    )
    rv = get_registered_variables(config)
    code_length = 3 * u.parsec
    code_mass = 100 * u.M_sun
    code_velocity = 100 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)
    n_h = 2
    rho_0 = n_h * c.m_p / u.cm**3
    p_0 = 3e4 * u.K / u.cm**3 * c.k_B
    params = SimulationParams(
        C_cfl=0.8, dt_max=0.1, gamma=gamma,
        minimum_density=(1e-2 * rho_0).to(code_units.code_density).value,
        minimum_pressure=(1e-2 * p_0).to(code_units.code_pressure).value,
    )
    sh = (resolution,) * 3
    zero = jnp.zeros(sh)
    B_0 = (13.5 * u.microgauss / c.mu0**0.5).to(code_units.code_magnetic_field).value
    bxb, byb, bzb = initialize_interface_fields(jnp.ones(sh) * B_0, zero, zero)
    probe = construct_primitive_state(
        config=config, registered_variables=rv, density=jnp.ones(sh),
        velocity_x=zero, velocity_y=zero, velocity_z=zero, gas_pressure=jnp.ones(sh),
        magnetic_field_x=jnp.ones(sh) * B_0, magnetic_field_y=zero, magnetic_field_z=zero,
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb)
    config = finalize_config(config, probe.shape)

    best_state = jnp.asarray(np.load(args.state))
    di = rv.density_index
    vi = rv.velocity_index
    target = load_image("logo.png", resolution, resolution)
    target = target / jnp.sum(target) * jnp.sum(best_state[di])

    rms_v = float(jnp.sqrt(jnp.mean(best_state[vi.x]**2 + best_state[vi.y]**2 + best_state[vi.z]**2)))
    print(f"rms_v(best)={rms_v:.4f}  t_cross(best)={box_size/rms_v:.4f}")

    def evolve(t):
        if t <= 0.0:
            return best_state[di]
        fin = time_integration(best_state, config, params._replace(t_end=float(t)), rv)
        return fin[di]

    ts = np.linspace(args.lo, args.hi, args.n)
    projs = []
    corrs = []
    mses = []
    for t in ts:
        dens = evolve(t)
        proj = jnp.sum(dens, axis=2)
        projs.append(np.asarray(proj, dtype=np.float32))
        corrs.append(ncorr(proj, target))
        mses.append(float(jnp.mean((proj - target) ** 2)))
        print(f"  t={t:.4f}  corr={corrs[-1]:+.4f}  mse={mses[-1]:.4e}")

    best_i = int(np.argmax(corrs))
    print(f"==> max-corr t={ts[best_i]:.4f}  corr={corrs[best_i]:.4f}")
    print(f"==> min-mse  t={ts[int(np.argmin(mses))]:.4f}")
    np.savez(args.out, ts=ts, corrs=np.array(corrs), mses=np.array(mses),
             projs=np.stack(projs), target=np.asarray(target, dtype=np.float32))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
