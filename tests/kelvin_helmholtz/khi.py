"""
# Kelvin-Helmholtz instability in 2D

We analyze the Kelvin-Helmholtz instability across 
Mach and Reynolds numbers.

## Setup

Where not indicated differently, 
our general setup follows Mandelker et al, 2016,
https://arxiv.org/pdf/1606.06289. Note that they
use different coordinate conventions, 

- their x-axis (transverse) is our y-axis
- their z-axis (flow direction) is our x-axis

### Baseline setup

Consider a dense slab with radius R_s flowing 
through a dilute background medium along the x-axis. 

- the simulation domain is [0, 1] x [0, 1] with periodic boundaries at x = 0 and x = 1
  and periodic (open in Mandelker et al.) boundaries at y = 0 and y = 1
- R_s = 1/160 in Mandelker et al.
- the slab is centered at y_c = 0.5
- slab and background are ideal gases with γ = 5/3 and P = 1 everywhere
- the background has density ρ_b = 1 and velocity v_b = 0
- the slab has density ρ_s = χ and velocity v_s = M_b * c_b * x̂, 
  where c_b = sqrt(γ * P / ρ_b) = sqrt(5/3) and M_b is the Mach 
  number of the hot dilute background flow relative to the slab
- diffusion can be included, the kinematic viscosity ν is set

Truly discontinuous density and velocity 
profiles lead to numerical noise at the grid-scale,
such that we use a smoothed transition

f(y) = f_b + 0.25 * (f_s - f_b) * (1 + tanh((R_s - (y - y_c)) / σ)) * (1 + tanh((R_s + (y - y_c)) / σ))

with smoothing length σ, e.g. σ = λ / 102, 
where λ is the wavelength of the perturbation.

Roediger et al. 2013 only use smoothing for the highest
Re run, they then smooth with σ being 1 to 2% of λ.

### Perturbation

There are multiple ways to perturb this system.

#### Velocity only perturbation

Following Sec. 3.1 in Roediger et al. 2013, we can perturb

v_y = v_0 sin(k * x) * [exp(-((y - y_c) - R_s)^2 / (σ_y^2)) + exp(-((y - y_c) + R_s)^2 / (σ_y^2))]

with 

- k = 2 * π / λ
- σ_y^2 = 0.3 * λ
- v_0 = 0.1 * v_s (the slab and also the shear velocity)

#### Pressure only perturbation (Sec. 3.3 in Mandelker et al.)

We can perturb the pressure with

P_1 = A * cos(k * x) * [exp(-((y - y_c) - R_s)^2 / (2 Σ^2)) + exp(-((y - y_c) + R_s)^2 / (2 Σ^2))]

e.g. with the same A and k as above and e.g. Σ = 5 * σ or

- A = 0.05
- Σ = 5 * σ
- k = 2 * π / λ, e.g. λ = R_s or λ = 2 * R_s

#### Eigenmode perturbations

##### Mandelker et al. analytical approach

Following Sec. 2.1 and 2.3 in Mandelker et al., by inserting
small perturbations into the Euler equations, one can relate
the perturbations in velocity, density, and pressure
to each other.

Denote \partial_y f as f' and k_x = k * cos(θ).

Given a pressure perturbation P_1(x, y),

ρ_1 = -1 / (k^2 (v_k - ω / k)^2) * [P_1'' - 2v_k' / (v_k - ω / k) * P_1' - k^2 P_1]
v_{1x} = -cos(θ) / (ρ_0 (v_k - ω / k)) * [v_k' / (k^2 cos^2(θ) (v_k - ω / k)) * P_1' + P_1]
v_{1y} = i / (ρ_0 k (v_k - ω / k)) * P_1'

where 

- k is the wavenumber of the perturbation
- θ is the angle of the perturbation wavevector with respect to the flow direction (x-axis)
- v_k = v_0 \cdot k = v_0 * cos(θ) is the component of the background 
  velocity in the direction of the perturbation wavevector
- ω is the complex frequency of the perturbation

In our case of the slab flowing along the x-axis, we have

- subscript s denotes the slab and subscript b denotes the background
- θ = 0 (perturbation aligned with the flow)
- v_k = v_0 = const. -> v_k' = 0

so

ρ_1 = -1 / (k^2 (v_0 - ω / k)^2) * [P_1'' - k^2 P_1]
v_{1x} = -1 / (ρ_0 (v_0 - ω / k)) * P_1
v_{1y} = i / (ρ_0 k (v_0 - ω / k)) * P_1'

For the slab, one finds the pressure perturbation P_1(y)
to be

         { A * exp(-q_b * (y - y_c - R_s)) for y > y_c + R_s
P_1(y) = | A * S(q_s * (y - y_c)) / S(q_s * R_s) for y_c - R_s <= y <= y_c + R_s
         { ±A * exp(q_b * (y - y_c + R_s)) for y < y_c - R_s

where

- S = cosh for pinch modes (plus sign in the bottom line)
- S = sinh for sinusoidal modes (minus sign in the bottom line)

where

- q_b = k * sqrt(1 - (Δv_b / c_b)^2)
- q_s = k * sqrt(1 - (Δv_s / c_s)^2)

where

- Δv_b = v_b - ω/k
- Δv_s = v_s - ω/k

Now we are still missing the complex frequency ω.
It follows from solving Eq. 25 in Mandelker et al.

(ω - k * v_b)^2 / (ω - k * v_s)^2 = -ρ_s / ρ_b * q_b / q_s * T(q_s * R_s)

where

- T = tanh for sinusoidal modes
- T = coth for pinch modes

This is solved by numerical root finding, under 
the constraint that Re(q_b) > 0 and 
Re(q_s) > 0 (wave decays towards
the edge).

The characteristic time of perturbation growth in 
the linear regime is the Kelvin-Helmholtz time 
k_KH = 1 / Im(ω).

Smoothing is applied to all initial perturbations 
(not fully consistent, but sufficient).

For more details, see the Mandelker et al. paper.

##### Purely numerical eigenmodes (optional)

One might alternatively directly linearize the Euler equations
around the background flow and numerically solve the resulting
eigenvalues problem. We might use JAX autodiff to linearize
the RHS of our fluid solver directly and solve for the eigenmodes
of the resulting linear operator, this allows us to allows us
to find the eigenmodes of our numerical scheme with the smoothed
initial conditions, and is flexible regarding the inclusion of 
additional physics like viscosity.

Ideally, one would implement both approaches 
and check for consistency.

## Criticality

### Critical Mach number

The critical Mach number is given by

M_crit = (1 + χ^{-1/3})^{3/2}  (Eq. 22 in Mandelker et al.)

where χ = ρ_s / ρ_b is the density contrast 
between the slab and the background.

- M_b < M_crit: surface modes grow rapidly, stream disrupts
- M_b > M_crit: surface modes suppressed, stream remains stable

### Critical Reynolds number

Roediger et al. 2013 (https://arxiv.org/pdf/1309.2635) empirically found

  Re_crit = 880 / Δ       (Eq. 22, for constant kinematic viscosity)
  Re_0 = 1320 / sqrt(Δ)   (Eq. 23)
  Δ = (ρ_cold + ρ_hot)² / (ρ_cold ρ_hot)

with the viscous growth time (Eq. 21):

  τ_KH,visc = τ_KH,inv × [1 + Re₀ / (Re − Re_crit)]

while the non-viscous growth time is

	t_kh = jnp.sqrt(Delta) / (2 * jnp.pi) * wavelength / v_slab

where approximately for

- Re >> Re_crit: viscosity negligible, usual KHI growth
- Re <  Re_crit:  viscosity dominates, KHI suppressed

However, suppresion by viscosity is a gradual effect and
it is hard to define what "suppressed" means in practice.

## Experimental diagnostics

### Growth time
Consider the maximum (or minimum) v_y, v_ymax as a function of time.

- until ~ 1 t_kh, v_ymax reflects sound waves from the initialization and we ignore it
- from ~ 1 t_kh to ~ 5 t_kh, we measure the growth time t_kh_measured by fitting
    v_ymax(t) = v_ymax(t=0) * exp(t / t_kh_measured)

"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# typing
from dataclasses import dataclass
from typing import NamedTuple

# numerics
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colors as mcolors

# astronomix
from astronomix import (
    SimulationConfig,
    get_helper_data,
    SimulationParams,
    time_integration,
    construct_primitive_state,
    get_registered_variables,
)
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    KINEMATIC_VISCOSITY,
    MINMOD,
    MUSCL,
    OPEN_BOUNDARY,
    SPLIT,
    UNSPLIT,
    SnapshotSettings,
    finalize_config,
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

SINGLE_INTERFACE = 0
SLAB = 1
interface_mode = SINGLE_INTERFACE

VELOCITY_PERTURBATION = 0
PRESSURE_PERTURBATION = 1
# EIGENMODE_PERTURBATION = 2

# global parameters
y_center = 0.5
background_density = 1.0
pressure = 1.0 # uniform pressure everywhere
background_velocity = 0.0
box_size = 1.0
gamma = 5/3

num_cells = 1024

@dataclass
class PerturbationSetup:

	#: The amplitude of the perturbation.
	amplitude: float

	#: The wavelength of the perturbation.
	wavelength: float

@dataclass
class PressurePerturbationSetup(PerturbationSetup):

	#: The width of the Gaussian envelope for the pressure perturbation.
	gaussian_width: float # Σ

class KHISetup(NamedTuple):

	#: The number of cells in the x and y directions.
	num_cells: int

	#: Simulation duration.
	simulation_time: float

	#: The perturbation type.
	perturbation_type: int

	#: The perturbation setup.
	perturbation_setup: PerturbationSetup

	#: Switch for whether to include diffusion (viscosity) or not.
	diffusion: bool

	#: The kinematic viscosity of the fluid, if diffusion is included.
	viscosity: float

	#: The Mach number of the slab relative to the background.
	mach_number: float # M_b

	#: The density contrast between the slab and the background.
	density_contrast: float # χ

	#: The radius (half-width) of the slab.
	slab_radius: float # R_s

	#: The smoothing length for the initial conditions.
	smoothing_length: float # σ

def slab_profile(f_b, f_s, Y, y_center, slab_radius, smoothing_length):
    """Tanh transition from f_b (background) to f_s (stream)."""

	# f(y) = f_b + 0.25 * (f_s - f_b) * (1 + tanh((R_s - (y - y_c)) / σ)) * (1 + tanh((R_s + (y - y_c)) / σ))
    return f_b + 0.25 * (f_s - f_b) * (
		(1 + jnp.tanh((slab_radius - (Y - y_center)) / smoothing_length)) *
		(1 + jnp.tanh((slab_radius + (Y - y_center)) / smoothing_length))
	)

def single_interface(f_l, f_u, Y, y_center, smoothing_length):
	return 0.5 * (f_l * (1 - jnp.tanh((Y - y_center) / smoothing_length)) + f_u * (1 + jnp.tanh((Y - y_center) / smoothing_length)))

def velocity_perturbation(cell_centers, slab_radius, perturbation_setup):
	# NOTE: here without smoothing
	k = 2 * jnp.pi / perturbation_setup.wavelength
	X = cell_centers[:, :, 0] # x-coordinates of cell centers
	Y = cell_centers[:, :, 1] # y-coordinates of cell centers
	sigma_y = 0.3 * perturbation_setup.wavelength
	vy_pert = perturbation_setup.amplitude * jnp.sin(k * X) * (
		jnp.exp(-((Y - y_center) - slab_radius)**2 / (sigma_y**2)) + 
		jnp.exp(-((Y - y_center) + slab_radius)**2 / (sigma_y**2))
	)
	return vy_pert

def single_interface_velocity_perturbation(cell_centers, perturbation_setup):
	# v_y = A * sin(k * x) * exp(-((y - y_c) / σ)^2)
	k = 2 * jnp.pi / perturbation_setup.wavelength
	X = cell_centers[:, :, 0] # x-coordinates of cell centers
	Y = cell_centers[:, :, 1] # y-coordinates of cell centers
	sigma = 0.2 * perturbation_setup.wavelength
	vy_pert = perturbation_setup.amplitude * jnp.sin(k * X) * jnp.exp(-((Y - y_center) / sigma)**2)
	return vy_pert

def pressure_perturbation(cell_centers, slab_radius, y_center, perturbation_setup):
	# P_1 = A * cos(k * x) * [exp(-((y - y_c) - R_s)^2 / (2 Σ^2)) + exp(-((y - y_c) + R_s)^2 / (2 Σ^2))]
	k = 2 * jnp.pi / perturbation_setup.wavelength
	X = cell_centers[:, :, 0] # x-coordinates of cell centers
	Y = cell_centers[:, :, 1] # y-coordinates of cell centers
	P_1 = perturbation_setup.amplitude * jnp.cos(k * X) * (
		jnp.exp(-((Y - y_center) - slab_radius)**2 / (2 * perturbation_setup.gaussian_width**2)) + 
		jnp.exp(-((Y - y_center) + slab_radius)**2 / (2 * perturbation_setup.gaussian_width**2))
	)
	return P_1

def simulate_khi(setup: KHISetup, return_snapshots = False):

	# set up the simulation configuration
	config = SimulationConfig(
		solver_mode = FINITE_DIFFERENCE,
		progress_bar = True,
		dimensionality = 2,
		box_size = box_size,
		num_cells = setup.num_cells,
		diffusion = setup.diffusion,
		viscosity_type = KINEMATIC_VISCOSITY,
		boundary_settings = BoundarySettings(
			x = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),   # flow dir
			y = BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),           # transverse
    	),
		return_snapshots = return_snapshots,
		num_snapshots = 300,
		snapshot_settings = SnapshotSettings(
			return_states = True,
		)
	)

	# set up the simulation parameters
	params = SimulationParams(
		viscosity = setup.viscosity,
		t_end = setup.simulation_time,
		C_cfl = 1.5 if config.solver_mode == FINITE_DIFFERENCE else 0.4, 
		gamma = gamma,
	)

	# helper data
	helper_data = get_helper_data(config)
	registered_variables = get_registered_variables(config)

	# construct the unperturbed initial conditions
	cell_centers = helper_data.geometric_centers
	Y = cell_centers[:, :, 1] # y-coordinates of cell centers
	stream_density = setup.density_contrast * background_density

	c_background = jnp.sqrt(params.gamma * pressure / background_density)

	smooth_profiles = True

	if smooth_profiles:
		if interface_mode == SINGLE_INTERFACE:

			density = single_interface(
				f_l = stream_density,
				f_u = background_density,
				Y = Y,
				y_center = y_center,
				smoothing_length = setup.smoothing_length,
			)

			v_mach = setup.mach_number * c_background
			v_x = single_interface(
				f_l = -v_mach / 2,
				f_u = v_mach / 2,
				Y = Y,
				y_center = y_center,
				smoothing_length = setup.smoothing_length,
			)
		
		elif interface_mode == SLAB:
			density = slab_profile(
				f_b = background_density,
				f_s = stream_density,
				Y = Y,
				y_center = y_center,
				slab_radius = setup.slab_radius,
				smoothing_length = setup.smoothing_length,
			)

			v_x = slab_profile(
				f_b = background_velocity,
				f_s = setup.mach_number * c_background,
				Y = Y,
				y_center = y_center,
				slab_radius = setup.slab_radius,
				smoothing_length = setup.smoothing_length,
			)
	else:
		if interface_mode == SLAB:
			density = jnp.where(
				jnp.abs(Y - y_center) <= setup.slab_radius,
				stream_density,
				background_density
			)

			v_x = jnp.where(
				jnp.abs(Y - y_center) <= setup.slab_radius,
				setup.mach_number * c_background,
				background_velocity
			)
		else:
			raise NotImplementedError("Single interface without smoothing not implemented yet.")

	v_y = jnp.zeros_like(v_x)

	P = jnp.full_like(v_x, pressure)

	primitive_state_unperturbed = construct_primitive_state(
		config = config,
		registered_variables = registered_variables,
		density = density,
		velocity_x = v_x,
		velocity_y = v_y,
		gas_pressure = P,
	)

	# add perturbation
	if setup.perturbation_type == VELOCITY_PERTURBATION:
		if interface_mode == SLAB:
			vy_pert = velocity_perturbation(
				cell_centers = cell_centers,
				slab_radius = setup.slab_radius,
				perturbation_setup = setup.perturbation_setup,
			)
			primitive_state = primitive_state_unperturbed.at[
				registered_variables.velocity_index.y
			].add(vy_pert)
		elif interface_mode == SINGLE_INTERFACE:
			vy_pert = single_interface_velocity_perturbation(
				cell_centers = cell_centers,
				perturbation_setup = setup.perturbation_setup
			)
			primitive_state = primitive_state_unperturbed.at[
				registered_variables.velocity_index.y
			].add(vy_pert)
	elif setup.perturbation_type == PRESSURE_PERTURBATION:
		if interface_mode == SLAB:
			P_pert = pressure_perturbation(
				cell_centers = cell_centers,
				slab_radius = setup.slab_radius,
				y_center = y_center,
				perturbation_setup = setup.perturbation_setup,
			)
			primitive_state = primitive_state_unperturbed.at[
				registered_variables.pressure_index
			].add(P_pert)
		elif interface_mode == SINGLE_INTERFACE:
			raise NotImplementedError("Single interface with pressure perturbation not implemented yet.")
	else:
		raise NotImplementedError("Only velocity and pressure perturbations are implemented so far.")

	# finalize the simulation configuration
	config = finalize_config(config, primitive_state.shape)

	# run the simulation
	result = time_integration(
		primitive_state,
		config, 
		params,
		registered_variables
	)

	return result, helper_data, registered_variables

def example_setup_run(
	density_contrast,
	Re_or_nu,
	mach_number,
	adapt_simulation_time = False,
	return_snapshots = True,
	nu_specified = False
):

	simulation_time = 2.0
	slab_radius = 0.1

	if nu_specified:
		kinematic_viscosity = Re_or_nu
		diffusion = True if kinematic_viscosity > 0 else False
	else:
		diffusion = True if Re_or_nu != float("inf") else False
		reynolds_number = Re_or_nu

	slab_density = density_contrast * background_density
	c_background = jnp.sqrt(gamma * pressure / background_density)
	c_slab = float(jnp.sqrt(gamma * pressure / slab_density))

	M_crit = (1 + density_contrast**(-1/3))**(3/2)
	v_slab = mach_number * c_background

	Delta = (slab_density + background_density)**2 / (slab_density * background_density)
	Re_crit = 880 / Delta
	Re = Re_or_nu

	perturbation_type = VELOCITY_PERTURBATION
	
	# wavelength = box_size / 5
	# amplitude = 0.01 * c_slab

	# in the Tirso paper
	wavelength = box_size / 2
	amplitude = mach_number * c_background / 20
	# smoothing_length = wavelength / 10
	smoothing_length = wavelength / 20
	# smoothing_length = wavelength / 20

	# Roedinger like
	# perturbation_type = VELOCITY_PERTURBATION
	# wavelength = box_size / 4
	# amplitude = 0.1 * v_slab
	# # amplitude for Ma = 0.5
	# # amplitude = 0.1 * 0.5 * c_background

	# smoothing_length = wavelength / 102

	if not nu_specified:
		kinematic_viscosity = wavelength * v_slab / Re

	if adapt_simulation_time:
		# KHI growth time from Eq. 2 in Roediger et al 2013
		# t_kh = jnp.sqrt(Delta) / (2 * jnp.pi) * wavelength / v_slab
		# Tirso paper
		t_kh = jnp.sqrt(Delta) * wavelength / v_slab
		print(f"Kelvin-Helmholtz time (inviscid): {t_kh:.3f}")
		# e.g. 20 * t_kh
		# simulation_time = 20.0 * t_kh
		# Tirso
		simulation_time = 2.0 * t_kh
		print(f"Adapting simulation time to {simulation_time:.3f} to capture KHI growth.")

	setup = KHISetup(
		num_cells = num_cells,
		simulation_time = simulation_time,
		perturbation_type = perturbation_type,
		perturbation_setup = PressurePerturbationSetup(
			amplitude = amplitude,
			wavelength = wavelength,
			gaussian_width = 5 * smoothing_length,
		),
		diffusion = diffusion,
		viscosity = kinematic_viscosity,
		mach_number = mach_number,
		density_contrast = density_contrast,
		slab_radius = slab_radius,
		smoothing_length = smoothing_length,
	)

	result, helper_data, registered_variables = simulate_khi(setup, return_snapshots = return_snapshots)

	return result, registered_variables, helper_data, Re_crit, M_crit

def side_by_side_comparison():

	density_contrast_A = 10.0
	reynolds_number_A = 100.0 # float("inf")
	mach_number_A = 0.5

	density_contrast_B = 10.0
	reynolds_number_B = 300.0
	mach_number_B = 0.5

	print(f"👨‍🔧 Running setup A: χ={density_contrast_A}, Re={reynolds_number_A}, M={mach_number_A}")
	result_A, registered_variables_A, helper_data_A, Re_crit_A, M_crit_A = example_setup_run(
		density_contrast = density_contrast_A,
		Re_or_nu = reynolds_number_A,
		mach_number = mach_number_A,
		adapt_simulation_time = True,
	)

	print(f"👨‍🔧 Running setup B: χ={density_contrast_B}, Re={reynolds_number_B}, M={mach_number_B}")
	result_B, registered_variables_B, helper_data_B, Re_crit_B, M_crit_B = example_setup_run(
		density_contrast = density_contrast_B,
		Re_or_nu = reynolds_number_B,
		mach_number = mach_number_B,
		adapt_simulation_time = True,
	)

	fig, axs = plt.subplots(2, 3, figsize=(15, 10))
	extent = [0, box_size, 0, box_size]

	num_snapshots = len(result_A.states)

	# Initialize images for animation
	im_A_rho = axs[0, 0].imshow(result_A.states[0][registered_variables_A.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
	im_A_vy = axs[0, 1].imshow(result_A.states[0][registered_variables_A.velocity_index.y].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	im_A_P = axs[0, 2].imshow(result_A.states[0][registered_variables_A.pressure_index].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)

	im_B_rho = axs[1, 0].imshow(result_B.states[0][registered_variables_B.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
	im_B_vy = axs[1, 1].imshow(result_B.states[0][registered_variables_B.velocity_index.y].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	im_B_P = axs[1, 2].imshow(result_B.states[0][registered_variables_B.pressure_index].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)

	# Set labels
	re_A_str = "∞" if reynolds_number_A == float("inf") else f"{reynolds_number_A:.0f}"
	setup_A_str = f"χ={density_contrast_A:.0f}, Re={re_A_str}, M={mach_number_A:.2f}"

	re_B_str = "∞" if reynolds_number_B == float("inf") else f"{reynolds_number_B:.0f}"
	setup_B_str = f"χ={density_contrast_B:.0f}, Re={re_B_str}, M={mach_number_B:.2f}"

	axs[0, 0].set_ylabel(setup_A_str, fontsize=11, fontweight='bold')
	axs[1, 0].set_ylabel(setup_B_str, fontsize=11, fontweight='bold')

	axs[0, 0].set_title('Density')
	axs[0, 1].set_title('Transverse Velocity ($v_y$)')
	axs[0, 2].set_title('Pressure')

	# Set equal aspect ratio for all axes
	for ax in axs.flat:
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])

	for ax, im in [
		(axs[0, 0], im_A_rho),
		(axs[0, 1], im_A_vy),
		(axs[0, 2], im_A_P),
		(axs[1, 0], im_B_rho),
		(axs[1, 1], im_B_vy),
		(axs[1, 2], im_B_P),
	]:
		cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
		fig.colorbar(im, cax=cax)


	suptitle_text = fig.suptitle("", fontsize=16, fontweight='bold')

	def animate(frame):
		time = result_A.time_points[frame]
		suptitle_text.set_text(
			rf"KHI, $t$ = {time:.3f}, $M_\text{{crit}} \approx$ {M_crit_A:.2f}, $Re_\text{{crit}} \approx$ {Re_crit_A:.0f}"
		)
		
		im_A_rho.set_data(result_A.states[frame][registered_variables_A.density_index].T)
		im_A_vy.set_data(result_A.states[frame][registered_variables_A.velocity_index.y].T)
		im_A_P.set_data(result_A.states[frame][registered_variables_A.pressure_index].T)
		
		im_B_rho.set_data(result_B.states[frame][registered_variables_B.density_index].T)
		im_B_vy.set_data(result_B.states[frame][registered_variables_B.velocity_index.y].T)
		im_B_P.set_data(result_B.states[frame][registered_variables_B.pressure_index].T)
		
		return [suptitle_text, im_A_rho, im_A_vy, im_A_P, im_B_rho, im_B_vy, im_B_P]

	print("🎬 Creating animation...")
	anim = animation.FuncAnimation(fig, animate, frames=num_snapshots, interval=50, blit=False, repeat=True)
	plt.tight_layout()

	anim.save('figures/khi_comparison.gif', writer='pillow')

	# also save the final state as a static figure
	fig_final, axs_final = plt.subplots(2, 3, figsize=(15, 10))
	extent = [0, box_size, 0, box_size]
	im_A_rho_final = axs_final[0, 0].imshow(result_A.states[-1][registered_variables_A.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
	im_A_vy_final = axs_final[0, 1].imshow(result_A.states[-1][registered_variables_A.velocity_index.y].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	im_A_P_final = axs_final[0, 2].imshow(result_A.states[-1][registered_variables_A.pressure_index].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	im_B_rho_final = axs_final[1, 0].imshow(result_B.states[-1][registered_variables_B.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
	im_B_vy_final = axs_final[1, 1].imshow(result_B.states[-1][registered_variables_B.velocity_index.y].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	im_B_P_final = axs_final[1, 2].imshow(result_B.states[-1][registered_variables_B.pressure_index].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	axs_final[0, 0].set_ylabel(setup_A_str, fontsize=11, fontweight='bold')
	axs_final[1, 0].set_ylabel(setup_B_str, fontsize=11, fontweight='bold')
	axs_final[0, 0].set_title('Density')
	axs_final[0, 1].set_title('Transverse Velocity ($v_y$)')
	axs_final[0, 2].set_title('Pressure')
	for ax in axs_final.flat:
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])
	for ax, im in [
		(axs_final[0, 0], im_A_rho_final),
		(axs_final[0, 1], im_A_vy_final),
		(axs_final[0, 2], im_A_P_final),
		(axs_final[1, 0], im_B_rho_final),
		(axs_final[1, 1], im_B_vy_final),
		(axs_final[1, 2], im_B_P_final),
	]:
		cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
		fig_final.colorbar(im, cax=cax)
	plt.tight_layout()
	fig_final.suptitle(f"KHI final state, $t$ = {result_A.time_points[-1]:.3f}", fontsize=16, fontweight='bold')
	plt.savefig('figures/khi_comparison_final.png')

# def parameter_sweep():

# 	"""
# 	We are interested in the following sweeps:
	
# 	- at a fixed contrast of χ = 10, vary the Reynolds number and Mach number across the critical values (should include 1.0)
# 	- at a fixed sub-critical Mach number, vary the density contrast and Reynolds number across the critical values
# 	- at a fixed non-critical Reynolds number, vary the density contrast and Mach number across the critical values
# 	"""

# 	pass

def khi_growth_over_time():
	"""
	Reproduces Fig. 3 from https://arxiv.org/pdf/2504.15345.

	Plots ln(v_y / v_y0) vs t / τ_KH for χ = 10 and varying Mach numbers
	at infinite Reynolds number (inviscid). Solid lines indicate unstable
	KHI (M < M_crit), dash-dotted lines indicate suppressed instability
	(M > M_crit).
	"""

	density_contrast = 10.0
	reynolds_number = float("inf")

	# Mach numbers from 0.1 to 1.8 in steps of 0.1 (matching the paper's sweep)
	mach_numbers = jnp.arange(0.1, 1.9, 0.1)
	# mach_numbers = [0.5, 2.5]

	# Critical Mach number for χ = 10 (Eq. 27 in Mandelker et al. 2016)
	# M_crit = (1 + density_contrast**(-1/3))**(3/2)
	M_crit = 1.65
	print(f"Critical Mach number for χ={density_contrast}: {M_crit:.3f}")

	# Precompute shared quantities for τ_KH calculation
	slab_density = density_contrast * background_density
	c_background = float(jnp.sqrt(gamma * pressure / background_density))
	Delta = float((slab_density + background_density)**2 / (slab_density * background_density))

	# The perturbation wavelength used in example_setup_run
	# wavelength = box_size / 5
	# Tirso paper
	wavelength = box_size / 2

	# Colormap setup: rainbow from blue (low M) to red (high M)
	cmap = plt.cm.jet
	norm = mcolors.Normalize(vmin=0.0, vmax=1.8)

	fig, ax = plt.subplots(figsize=(8, 6))

	for i, M in enumerate(mach_numbers):
		print(f"Running M = {M:.1f} ...")
		print(f"Simulation {i + 1}/{len(mach_numbers)}")

		v_slab = M * c_background
		# t_kh = jnp.sqrt(Delta) / (2 * jnp.pi) * wavelength / v_slab
		t_kh = jnp.sqrt(Delta) * wavelength / v_slab
		t_kh = float(t_kh)
		print(f"  τ_KH = {t_kh:.4f}")

		result, registered_variables, helper_data, Re_crit, M_crit_val = example_setup_run(
			density_contrast = density_contrast,
			Re_or_nu = reynolds_number,
			mach_number = M,
			adapt_simulation_time = True,
			return_snapshots = True,
		)

		num_snapshots = len(result.states)
		times = jnp.array([float(result.time_points[j]) for j in range(num_snapshots)])

		# X = helper_data.geometric_centers[:, :, 0]
		# k = 2 * jnp.pi / wavelength
		# sin_kx = jnp.sin(k * X)  # shape (nx, ny)
		# cos_kx = jnp.cos(k * X)  # shape (nx, ny)

		# mode_amplitudes = []

		# center_idx = num_cells // 2

		# for j in range(num_snapshots):
		# 	vy_field = result.states[j][registered_variables.velocity_index.y]
			
		# 	# Project onto both components, to remove phase dependence
		# 	proj_sin = jnp.mean(vy_field * sin_kx, axis=0) * 2  # shape (ny,)
		# 	proj_cos = jnp.mean(vy_field * cos_kx, axis=0) * 2  # shape (ny,)
			
		# 	# The true amplitude is the vector magnitude of the Fourier coefficients
		# 	amp_y = jnp.sqrt(proj_sin**2 + proj_cos**2)
			
		# 	# mode at interface
		# 	mode_amp = amp_y[center_idx]
		# 	mode_amplitudes.append(float(mode_amp))

		# Prepare McNally's discrete convolution variables
		k = 2 * jnp.pi / wavelength
		X = helper_data.geometric_centers[:, :, 0]
		Y = helper_data.geometric_centers[:, :, 1]

		sin_kx = jnp.sin(k * X)
		cos_kx = jnp.cos(k * X)

		# This is Eq 8 (d_i) from McNally: The exponential KHI envelope
		# (Assuming the single interface is at y_center = 0.5)
		W = jnp.exp(-k * jnp.abs(Y - y_center)) 

		sum_W = jnp.sum(W) # Denominator for Eq 9

		mode_amplitudes = []
		for j in range(num_snapshots):
			vy_field = result.states[j][registered_variables.velocity_index.y]
			
			# This evaluates Eq 6 (s_i) and Eq 7 (c_i) integrated over the domain
			sum_s = jnp.sum(vy_field * sin_kx * W)
			sum_c = jnp.sum(vy_field * cos_kx * W)
			
			# This evaluates Eq 9 (M) from McNally
			mode_amp = 2 * jnp.sqrt((sum_s / sum_W)**2 + (sum_c / sum_W)**2)
			
			mode_amplitudes.append(float(mode_amp))

		mode_amplitudes = jnp.array(mode_amplitudes)

		# Normalize time by τ_KH
		t_normalized = times / t_kh

		# Normalize by initial mode amplitude and take log
		amp0 = mode_amplitudes[0]
		if amp0 > 0:
			ln_ratio = jnp.log(mode_amplitudes / amp0)
		else:
			ln_ratio = jnp.zeros_like(mode_amplitudes)

		# Only plot up to ~2 τ_KH
		mask = t_normalized <= 2.0
		t_plot = t_normalized[mask]
		ln_plot = ln_ratio[mask]
		last_mask_index = jnp.sum(mask) - 1

		# Line style: solid if M < M_crit, dash-dotted if M >= M_crit
		linestyle = '-' if M < M_crit else '-.'
		linewidth = 2.0 if M >= M_crit else 1.5
		color = cmap(norm(M))

		ax.plot(t_plot, ln_plot, linestyle=linestyle, color=color, linewidth=linewidth)
		print(f"  Done. Final ln(A/A0) = {float(ln_plot[-1]):.2f}")

		# also plot the final density snapshot
		fig_snap, ax_snap = plt.subplots(figsize=(6, 5))
		extent = [0, box_size, 0, box_size]
		im = ax_snap.imshow(result.states[last_mask_index][registered_variables.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
		ax_snap.set_title(f"KHI density snapshot, M={M:.1f}, t={times[last_mask_index]:.3f}")
		cbar = fig_snap.colorbar(im, ax=ax_snap)
		cbar.set_label('Density')
		fig_snap.tight_layout()
		fig_snap.savefig(f'figures/khi_density_snapshot_M{M:.1f}.png', dpi=150)
		print(f"Saved figures/khi_density_snapshot_M{M:.1f}.png")

	# # Vertical dashed line at t/τ_KH = 1
	# ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)

	# Colorbar
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax)
	cbar.set_label(r'$\mathcal{M}_\mathrm{h}$', fontsize=14)

	ax.set_xlabel(r'$t\;/\;\tau_\mathrm{KH}$', fontsize=14)
	ax.set_ylabel(r'$\ln\!\left(\frac{v_y}{v_{y0}}\right)$', fontsize=14)
	# ax.set_xlim(0, 2.0)
	# ax.set_ylim(-4.5, 2.5)
	ax.set_title(rf'Growth of KHI with time for $\chi = {density_contrast:.0f}$', fontsize=14)

	fig.tight_layout()
	fig.savefig('figures/khi_growth_over_time.png', dpi=150)
	print("Saved figures/khi_growth_over_time.png")


def parameter_sweep():
	
	density_contrast = 10.0

	num_nu = 100
	num_Ma = 100

	nu_ref = 1809 # "Spitzer viscosity"
	# nus = jnp.linspace(0.0, 0.6, num_nu) * nu_ref
	Res = jnp.geomspace(30, 3000, num_nu)
	mach_numbers = jnp.linspace(0.5, 2.5, num_Ma)

	# num_nu = 1
	# num_Ma = 1

	# Res = [30]
	# mach_numbers = [1.4]

	# Critical Mach number for χ = 10 (Eq. 27 in Mandelker et al. 2016)
	M_crit = (1 + density_contrast**(-1/3))**(3/2)
	print(f"Critical Mach number for χ={density_contrast}: {M_crit:.3f}")

	# Precompute shared quantities for τ_KH calculation
	slab_density = density_contrast * background_density
	c_background = float(jnp.sqrt(gamma * pressure / background_density))
	Delta = float((slab_density + background_density)**2 / (slab_density * background_density))

	wavelength = box_size / 2

	heat_map = jnp.zeros((num_nu, num_Ma))

	for nu_index in range(num_nu):
		for mach_index in range(num_Ma):

			M = mach_numbers[mach_index]
			# nu = nus[nu_index]
			Re = Res[nu_index]

			# print(f"Running M = {M:.1f}, ν = {nu:.2e} ...")
			print(f"Running M = {M:.1f}, Re = {Re:.2e} ...")
			print(f"Simulation {nu_index * num_Ma + mach_index + 1}/{num_nu * num_Ma}")

			v_slab = M * c_background
			# t_kh = jnp.sqrt(Delta) / (2 * jnp.pi) * wavelength / v_slab
			t_kh = jnp.sqrt(Delta) * wavelength / v_slab
			t_kh = float(t_kh)
			print(f"  τ_KH = {t_kh:.4f}")

			result, registered_variables, helper_data, Re_crit, M_crit_val = example_setup_run(
				density_contrast = density_contrast,
				Re_or_nu = Re,
				mach_number = M,
				adapt_simulation_time = True,
				return_snapshots = True,
				nu_specified = False,
			)

			num_snapshots = len(result.states)
			times = jnp.array([float(result.time_points[j]) for j in range(num_snapshots)])

			X = helper_data.geometric_centers[:, :, 0]
			k = 2 * jnp.pi / wavelength
			sin_kx = jnp.sin(k * X)  # shape (nx, ny)
			cos_kx = jnp.cos(k * X)  # shape (nx, ny)

			mode_amplitudes = []
			for j in range(num_snapshots):
				vy_field = result.states[j][registered_variables.velocity_index.y]
				
				# Project onto both components, to remove phase dependence
				proj_sin = jnp.mean(vy_field * sin_kx, axis=0) * 2  # shape (ny,)
				proj_cos = jnp.mean(vy_field * cos_kx, axis=0) * 2  # shape (ny,)
				
				# The true amplitude is the vector magnitude of the Fourier coefficients
				amp_y = jnp.sqrt(proj_sin**2 + proj_cos**2)
				
				# Max over y to get the envelope peak
				mode_amp = jnp.max(amp_y)
				mode_amplitudes.append(float(mode_amp))

			mode_amplitudes = jnp.array(mode_amplitudes)

			# Normalize time by τ_KH
			t_normalized = times / t_kh

			# Normalize by initial mode amplitude and take log
			amp0 = mode_amplitudes[0]
			if amp0 > 0:
				ln_ratio = jnp.log(mode_amplitudes / amp0)
			else:
				ln_ratio = jnp.zeros_like(mode_amplitudes)

			# Only plot up to ~2 τ_KH
			mask = t_normalized <= 2.0
			t_plot = t_normalized[mask]
			ln_plot = ln_ratio[mask]
			last_mask_index = jnp.sum(mask) - 1

			result_value = ln_ratio[last_mask_index]
			heat_map = heat_map.at[nu_index, mach_index].set(result_value)
			print(f"  Done. Final ln(A/A0) = {result_value:.2f}")

			# also plot the final density snapshot
			fig_snap, ax_snap = plt.subplots(figsize=(6, 5))
			extent = [0, box_size, 0, box_size]
			im = ax_snap.imshow(result.states[last_mask_index][registered_variables.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
			ax_snap.set_title(f"KHI density snapshot, M={M:.1f}, Re={Re:.2e}, t={times[last_mask_index]:.3f}")
			cbar = fig_snap.colorbar(im, ax=ax_snap)
			cbar.set_label('Density')
			fig_snap.tight_layout()
			fig_snap.savefig(f'figures/sweep/khi_density_snapshot_M{M:.1f}_Re{Re:.2e}.png', dpi=150)
			print(f"Saved figures/sweep/khi_density_snapshot_M{M:.1f}_Re{Re:.2e}.png")

	print(heat_map)
	# save the heatmap for later data analysis
	jnp.save(f'khi_growth_heatmap_chi{density_contrast:.0f}.npy', heat_map)
	print(f"Saved khi_growth_heatmap_chi{density_contrast:.0f}.npy")

	# plot the results as a heatmap
	fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
	im = ax_heat.imshow(heat_map, origin='lower', aspect='auto', extent=[mach_numbers[0], mach_numbers[-1], Res[0], Res[-1]], cmap='viridis')
	ax_heat.set_xlabel('Mach number')
	ax_heat.set_yscale('log')
	ax_heat.set_ylabel('Reynolds number')
	ax_heat.set_title(f'KHI growth (ln(A/A0) at t ~ 2 τ_KH) for χ={density_contrast:.0f}')
	cbar = fig_heat.colorbar(im, ax=ax_heat)
	cbar.set_label(r'$\ln\!\left(\frac{A}{A_0}\right)$', fontsize=14)
	fig_heat.tight_layout()
	fig_heat.savefig(f'figures/khi_growth_heatmap_chi{density_contrast:.0f}.png', dpi=150)
	print(f"Saved figures/khi_growth_heatmap_chi{density_contrast:.0f}.png")


if __name__ == "__main__":
	# side_by_side_comparison()
	# parameter_sweep()
	khi_growth_over_time()