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
  number of the slab relative to the background
- diffusion can be included, the kinematic viscosity ν is set

Truly discontinuous density and velocity 
profiles lead to numerical noise at the grid-scale,
such that we use a smoothed transition

f(y) = f_b + 0.25 * (f_s - f_b) * (1 + tanh((R_s - (y - y_c)) / σ)) * (1 + tanh((R_s + (y - y_c)) / σ))

with smoothing length σ, e.g. σ = λ / 102, 
where λ is the wavelength of the perturbation.

### Perturbation

There are multiple ways to perturb this system.

#### Velocity only perturbation

We can perturb the y-velocity, e.g. with a 
transverse sinusoidal perturbation:

v_y(x, y) = A * sin(k * x)

where

- e.g. A = 0.01 * c_s is a small perturbation amplitude
- k = 2 * π / λ, λ = 1 / n, e.g. n = 5 (waves)

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

where for

- Re >> Re_crit: viscosity negligible, usual KHI growth
- Re <  Re_crit:  viscosity dominates, KHI suppressed

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
    KINEMATIC_VISCOSITY,
    SnapshotSettings,
    finalize_config,
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

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

def velocity_perturbation(cell_centers, perturbation_setup):
	k = 2 * jnp.pi / perturbation_setup.wavelength
	X = cell_centers[:, :, 0] # x-coordinates of cell centers
	vy_pert = perturbation_setup.amplitude * jnp.sin(k * X)
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
			y = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),   # transverse
    	),
		return_snapshots = return_snapshots,
		num_snapshots = 100,
		snapshot_settings = SnapshotSettings(
			return_states = True,
		)
	)

	# set up the simulation parameters
	params = SimulationParams(
		viscosity = setup.viscosity,
		t_end = setup.simulation_time,
		C_cfl = 1.5,
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
		vy_pert = velocity_perturbation(
			cell_centers = cell_centers,
			perturbation_setup = setup.perturbation_setup,
		)
		primitive_state = primitive_state_unperturbed.at[
			registered_variables.velocity_index.y
		].add(vy_pert)
	elif setup.perturbation_type == PRESSURE_PERTURBATION:
		P_pert = pressure_perturbation(
			cell_centers = cell_centers,
			slab_radius = setup.slab_radius,
			y_center = y_center,
			perturbation_setup = setup.perturbation_setup,
		)
		primitive_state = primitive_state_unperturbed.at[
			registered_variables.pressure_index
		].add(P_pert)
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

	return result, registered_variables

def example_setup_run(density_contrast, reynolds_number, mach_number):

	num_cells = 512
	simulation_time = 2.0
	slab_radius = 0.1
	smoothing_length = 0.008
	diffusion = reynolds_number is not jnp.inf and reynolds_number is not float("inf")

	slab_density = density_contrast * background_density
	c_background = jnp.sqrt(gamma * pressure / background_density)
	c_slab = float(jnp.sqrt(gamma * pressure / slab_density))
	perturbation_type = PRESSURE_PERTURBATION
	wavelength = box_size / 5
	amplitude = 0.01 * c_slab

	M_crit = (1 + density_contrast**(-1/3))**(3/2)

	v_slab = mach_number * c_background

	Delta = (slab_density + background_density)**2 / (slab_density * background_density)
	Re_crit = 880 / Delta
	Re = reynolds_number

	kinematic_viscosity = wavelength * v_slab / Re

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

	result, registered_variables = simulate_khi(setup, return_snapshots = True)

	return result, registered_variables, Re_crit, M_crit

density_contrast_A = 10.0
reynolds_number_A = float("inf")
mach_number_A = 0.9

density_contrast_B = 10.0
reynolds_number_B = 50.0
mach_number_B = 0.9

print(f"👨‍🔧 Running setup A: χ={density_contrast_A}, Re={reynolds_number_A}, M={mach_number_A}")
result_A, registered_variables_A, Re_crit_A, M_crit_A = example_setup_run(
	density_contrast = density_contrast_A,
	reynolds_number = reynolds_number_A,
	mach_number = mach_number_A,
)

print(f"👨‍🔧 Running setup B: χ={density_contrast_B}, Re={reynolds_number_B}, M={mach_number_B}")
result_B, registered_variables_B, Re_crit_B, M_crit_B = example_setup_run(
	density_contrast = density_contrast_B,
	reynolds_number = reynolds_number_B,
	mach_number = mach_number_B,
)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
extent = [0, box_size, 0, box_size]

num_snapshots = len(result_A.states)

# Initialize images for animation
im_A_rho = axs[0, 0].imshow(result_A.states[0][registered_variables_A.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent)
im_A_vy = axs[0, 1].imshow(result_A.states[0][registered_variables_A.velocity_index.y].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
im_A_P = axs[0, 2].imshow(result_A.states[0][registered_variables_A.pressure_index].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)

im_B_rho = axs[1, 0].imshow(result_B.states[0][registered_variables_B.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent)
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