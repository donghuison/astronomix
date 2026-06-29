#!/usr/bin/env python
"""Estimate the timestep count N (and recomputation number r) for the inverse sim.
CFL dt on a real 64³ MHD state (CPU, no GPU). N propto 1/dx, so N_128 = 2*N_64."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import jax, jax.numpy as jnp
from math import comb

from astronomix import (SimulationConfig, SimulationParams, get_registered_variables,
                        CodeUnits, finalize_config)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D, PALLAS)
from astronomix._finite_difference._timestep_estimation._timestep_estimator import _cfl_time_step_fd
from astropy import units as u
import astropy.constants as c

CU = CodeUnits(3 * u.parsec, 100 * u.M_sun, 100 * u.km / u.s)
RHO0 = (2 * c.m_p / u.cm**3).to(CU.code_density).value
P0 = (3e4 * u.K / u.cm**3 * c.k_B).to(CU.code_pressure).value

state = jnp.asarray(np.load("best_state_pallas_64.npy"))   # (11,64,64,64) real inverse state
N = state.shape[-1]
cfg = SimulationConfig(solver_mode=FINITE_DIFFERENCE, mhd=True, dimensionality=3,
                       box_size=1.0, num_cells=N, backend=PALLAS,
                       boundary_settings=BoundarySettings(
                           *[BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY) for _ in range(3)]))
cfg = finalize_config(cfg, state.shape)
rv = get_registered_variables(cfg)
params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5/3,
                          minimum_density=1e-2*RHO0, minimum_pressure=1e-2*P0)

dx = 1.0 / N
dt = float(_cfl_time_step_fd(state, dx, 0.1, 5/3, cfg, params, rv, 0.8))
vi = rv.velocity_index
rms = float(jnp.sqrt(jnp.mean(state[vi.x]**2 + state[vi.y]**2 + state[vi.z]**2)))
t_end = 0.5 / max(rms, 1e-12)
N64 = int(np.ceil(t_end / dt))
N128 = 2 * N64
print(f"state {N}³  rms={rms:.4f}  t_end={t_end:.4f}  dt={dt:.5f}")
print(f"  N_64  = {N64}  timesteps   (t_end/dt)")
print(f"  N_128 = {N128} timesteps   (dt halves with dx)")

def r_of(Nsteps, C):
    r = 1
    while comb(C + r, r) < Nsteps:
        r += 1
    return r

print("\n recomputation number r = min t : C(C+t, t) >= N   (equinox treeverse)")
for C in (8, 16, 24, 32, 48, 64):
    print(f"  C={C:3d} checkpoints :  r_64={r_of(N64,C)}   r_128={r_of(N128,C)}")
print(f"\n  current runs use C=16  ->  r_64={r_of(N64,16)}, r_128={r_of(N128,16)}")
# min C for r=2 at 128
for C in range(2, 400):
    if r_of(N128, C) <= 2:
        print(f"  smallest C giving r=2 at 128³: C={C}  (mem ~ {C*11*128**3*8/1e9:.1f} GB of checkpoints)")
        break
