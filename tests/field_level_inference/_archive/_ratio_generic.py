#!/usr/bin/env python
"""Forward vs value_and_grad runtime ratio for the WENO Pallas reverse kernel,
on a SIMPLE smooth IC (no turbulent forcing).  --hydro or --mhd.  The ratio is a
property of the reverse kernel, ~independent of the specific state, so a short
stable run suffices.  Reports forward time, backward time, ratio."""
import argparse, time
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["hydro", "mhd"], default="hydro")
ap.add_argument("--N", type=int, default=64)
ap.add_argument("--nchk", type=int, default=16)
ap.add_argument("--t-end", type=float, default=0.2)
args = ap.parse_args()

from autocvd import autocvd
autocvd(num_gpus=1)
import numpy as np
import jax, jax.numpy as jnp
from astronomix import (SimulationConfig, SimulationParams, get_registered_variables,
                        construct_primitive_state, time_integration)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    PALLAS, BACKWARDS, FORWARDS, finalize_config)
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields

N = args.N; MHD = args.mode == "mhd"

def cfg_of(backward):
    return SimulationConfig(solver_mode=FINITE_DIFFERENCE, mhd=MHD, progress_bar=False,
        enforce_positivity=True, donate_state=False, dimensionality=3, box_size=1.0, num_cells=N,
        differentiation_mode=BACKWARDS if backward else FORWARDS, num_checkpoints=args.nchk,
        boundary_settings=BoundarySettings(*[BoundarySettings1D(PERIODIC_BOUNDARY,PERIODIC_BOUNDARY) for _ in range(3)]),
        backend=PALLAS, pallas_block_shape=(4,4,8), pallas_use_triton=True)

rvc = cfg_of(False); rv = get_registered_variables(rvc)
params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5/3, minimum_density=1e-3, minimum_pressure=1e-3)
inv = params._replace(t_end=args.t_end)

# smooth IC: uniform rho,p; sinusoidal (Taylor-Green-ish) velocity, Mach ~0.3
x = jnp.linspace(0, 2*jnp.pi, N, endpoint=False)
X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
rho0 = jnp.ones((N,)*3); p0 = jnp.ones((N,)*3)
vx0 = 0.3*jnp.sin(X)*jnp.cos(Y)*jnp.cos(Z)
vy0 = -0.3*jnp.cos(X)*jnp.sin(Y)*jnp.cos(Z)
vz0 = jnp.zeros((N,)*3)
B0 = (0.2*jnp.ones((N,)*3), jnp.zeros((N,)*3), jnp.zeros((N,)*3))

def build(cfg, v):
    if MHD:
        bxb,byb,bzb = initialize_interface_fields(B0[0],B0[1],B0[2])
        return construct_primitive_state(config=cfg, registered_variables=rv, density=rho0, gas_pressure=p0,
            velocity_x=v[0],velocity_y=v[1],velocity_z=v[2],
            magnetic_field_x=B0[0],magnetic_field_y=B0[1],magnetic_field_z=B0[2],
            interface_magnetic_field_x=bxb,interface_magnetic_field_y=byb,interface_magnetic_field_z=bzb)
    return construct_primitive_state(config=cfg, registered_variables=rv, density=rho0, gas_pressure=p0,
        velocity_x=v[0],velocity_y=v[1],velocity_z=v[2])

theta0 = jnp.stack([vx0, vy0, vz0])
cfgF = finalize_config(cfg_of(False), (rv.num_vars,)+(N,)*3)
cfgB = finalize_config(cfg_of(True),  (rv.num_vars,)+(N,)*3)
di = rv.density_index

def loss_on(cfgx):
    def loss(theta):
        s = build(cfgx, theta)
        return jnp.mean(time_integration(s, cfgx, inv, rv)[di]**2)
    return loss
fwd = jax.jit(loss_on(cfgF))
vg  = jax.jit(jax.value_and_grad(loss_on(cfgB)))

def timeit(f, n=4):
    jax.block_until_ready(f(theta0))
    ts = []
    for _ in range(n):
        t0 = time.time(); jax.block_until_ready(f(theta0)); ts.append(time.time()-t0)
    return float(np.median(ts))

print(f"mode={args.mode} N={N} nchk={args.nchk} t_end={args.t_end}", flush=True)
lf = float(fwd(theta0)); print(f"  forward loss={lf:.4e} (finite={np.isfinite(lf)})", flush=True)
tf = timeit(fwd);                 print(f"  FORWARD only       : {tf:.3f} s", flush=True)
tb = timeit(lambda th: vg(th)[0]); print(f"  BACKWARD (val+grad): {tb:.3f} s", flush=True)
print(f"\n  >>> backward/forward = {tb/tf:.1f}x   ({args.mode}, N={N}, nchk={args.nchk})", flush=True)
