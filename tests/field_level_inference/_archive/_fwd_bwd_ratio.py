#!/usr/bin/env python
"""Measure pure-forward vs value_and_grad (backward) runtime at 64³ for the logo
inverse sim (half crossing time), and the backward/forward ratio."""
import os, time
from autocvd import autocvd
autocvd(num_gpus=1)
import numpy as np
import jax, jax.numpy as jnp
from astronomix import (SimulationConfig, SimulationParams, get_registered_variables,
                        construct_primitive_state, time_integration, CodeUnits)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    PALLAS, BACKWARDS, FORWARDS, finalize_config)
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams)
from astropy import units as u
import astropy.constants as c

CU = CodeUnits(3*u.parsec, 100*u.M_sun, 100*u.km/u.s)
RHO0=(2*c.m_p/u.cm**3).to(CU.code_density).value
P0=(3e4*u.K/u.cm**3*c.k_B).to(CU.code_pressure).value
B0=(13.5*u.microgauss/c.mu0**0.5).to(CU.code_magnetic_field).value
DEDT=(4.3e34*u.erg/u.s).to(CU.code_energy/CU.code_time).value
N=64; NCHK=16

def cfg_of(forcing, backward):
    return SimulationConfig(solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        enforce_positivity=True, donate_state=False, dimensionality=3, box_size=1.0, num_cells=N,
        differentiation_mode=BACKWARDS if backward else FORWARDS, num_checkpoints=NCHK,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=forcing),
        boundary_settings=BoundarySettings(*[BoundarySettings1D(PERIODIC_BOUNDARY,PERIODIC_BOUNDARY) for _ in range(3)]),
        backend=PALLAS, pallas_block_shape=(4,4,8), pallas_use_triton=True)

def build(cfg, rv, rho, v, p, B):
    bxb,byb,bzb = initialize_interface_fields(B[0],B[1],B[2])
    return construct_primitive_state(config=cfg, registered_variables=rv, density=rho, gas_pressure=p,
        velocity_x=v[0],velocity_y=v[1],velocity_z=v[2], magnetic_field_x=B[0],magnetic_field_y=B[1],magnetic_field_z=B[2],
        interface_magnetic_field_x=bxb,interface_magnetic_field_y=byb,interface_magnetic_field_z=bzb)

cg=cfg_of(True,False); rv=get_registered_variables(cg)
params=SimulationParams(C_cfl=0.8,dt_max=0.1,gamma=5/3,minimum_density=1e-2*RHO0,minimum_pressure=1e-2*P0,
    turbulent_forcing_params=TurbulentForcingParams(energy_injection_rate=DEDT))
sh=(N,)*3; z=jnp.zeros(sh)
s0=build(finalize_config(cg,(11,)+sh),rv,jnp.ones(sh)*RHO0,(z,z,z),jnp.ones(sh)*P0,(jnp.ones(sh)*B0,z,z))
cg=finalize_config(cg,s0.shape)
turb=jax.block_until_ready(time_integration(s0,cg,params._replace(t_end=(24*1e4*u.yr).to(CU.code_time).value),rv))
di,vi,pi,mi=rv.density_index,rv.velocity_index,rv.pressure_index,rv.magnetic_index
rms=float(jnp.sqrt(jnp.mean(turb[vi.x]**2+turb[vi.y]**2+turb[vi.z]**2))); t_end=0.5/max(rms,1e-12)
inv=params._replace(t_end=t_end)
rho_bg,p_bg=turb[di],turb[pi]; Bbg=(turb[mi.x],turb[mi.y],turb[mi.z])
theta0=jnp.stack([turb[vi.x],turb[vi.y],turb[vi.z]])

cfgF=finalize_config(cfg_of(False,False),turb.shape)   # forward-only
cfgB=finalize_config(cfg_of(False,True),turb.shape)    # backward (checkpointed)

def loss_on(cfgx):
    def loss(theta):
        s=build(cfgx,rv,rho_bg,(theta[0],theta[1],theta[2]),p_bg,Bbg)
        return jnp.mean((jnp.sum(time_integration(s,cfgx,inv,rv)[di],axis=2)-rho_bg.sum(2))**2)
    return loss
fwd=jax.jit(loss_on(cfgF))
vg =jax.jit(jax.value_and_grad(loss_on(cfgB)))

def timeit(f, n=3):
    jax.block_until_ready(f(theta0))            # compile
    ts=[]
    for _ in range(n):
        t0=time.time(); jax.block_until_ready(f(theta0)); ts.append(time.time()-t0)
    return float(np.median(ts))

print(f"rms={rms:.4f}  t_end={t_end:.4f}  N_checkpoints={NCHK}", flush=True)
tf=timeit(fwd); print(f"FORWARD only      : {tf:.2f} s", flush=True)
tb=timeit(lambda th: vg(th)[0]); print(f"BACKWARD (val+grad): {tb:.2f} s", flush=True)
print(f"\n  backward / forward ratio = {tb/tf:.2f}x   (64³, N_chk={NCHK})", flush=True)
print(f"  (flow-limited dt would give ~{int(np.ceil(t_end/(0.8*(1/N)/rms)))} steps; "
      f"void-throttled CFL gives ~611)", flush=True)
