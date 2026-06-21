"""At what resolution does M~20 decaying MHD turbulence stay finite?

Same cold (p0=1e-3) random-solenoidal decaying setup, dual energy +
REDISTRIBUTE positivity + velocity cap, swept over N. Reports whether the run
reaches t_end finite and the minimum density reached.
"""
# ruff: noqa: E402
import numpy as np
from autocvd import autocvd
autocvd(num_gpus=1)

import jax.numpy as jnp
from astronomix.option_classes.simulation_config import POSITIVITY_REDISTRIBUTE
import tests.dual_energy.refactor_value_tests as v

GAMMA = v.GAMMA

if __name__ == "__main__":
    M = 20.0
    cs = np.sqrt(GAMMA * 1e-3); vrms = M * cs; vcap = 5.0 * vrms
    print(f"[M=20 resolution] cold p0=1e-3, v_rms={vrms:.3f}, dual + REDISTRIBUTE + vcap={vcap:.2f}, t_end=0.15")
    for N in (64, 96, 128):
        r = v.run_turb(M, dual=True, stage_mode=POSITIVITY_REDISTRIBUTE, cap=vcap, N=N)
        print(f"  N={N:3d}: finite={r['finite']!s:5} rho_min={r['rho_min']:.2e} "
              f"p_min={r['p_min']:.2e} max|vx|={r['vmax']:.3e}")
