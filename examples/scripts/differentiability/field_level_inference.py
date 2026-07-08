"""
Differentiability: field-level inference in 3D.

We optimize the initial velocity field of a 3D MHD simulation so that the
z-projected final density matches a target image (the astronomix logo). This is
the simplest version of the "logo" example: the whole forward simulation is
differentiated end-to-end with reverse-mode autodiff and an Adam optimizer.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# optimization
import optax

# image loading
from PIL import Image

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    FINITE_DIFFERENCE,
    BACKWARDS,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
)

# astronomix functions
from astronomix import (
    time_integration,
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
    initialize_interface_fields,
)

# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# configure a small differentiable 3D MHD simulation (reverse-mode)
num_cells = 32
rho_0, p_0, B_0 = 1.0, 1.0, 1.0

config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    mhd = True,
    progress_bar = False,
    dimensionality = 3,
    box_size = 1.0,
    num_cells = num_cells,
    differentiation_mode = BACKWARDS,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
)

registered_variables = get_registered_variables(config)
params = SimulationParams(t_end = 0.4, C_cfl = 0.8, gamma = 5 / 3)

# uniform background threaded by a uniform field along x
rho = jnp.ones((num_cells,) * 3) * rho_0
p = jnp.ones((num_cells,) * 3) * p_0
B_x = jnp.ones((num_cells,) * 3) * B_0
B_y = jnp.zeros((num_cells,) * 3)
B_z = jnp.zeros((num_cells,) * 3)
bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

config = finalize_config(config, (registered_variables.num_vars,) + (num_cells,) * 3)

# load the logo, downsample to the grid, normalize to the background column mass
def load_target(path, N):
    img = jnp.asarray(Image.open(path).convert("L"))
    img = 1.0 - img / 255.0
    h, w = img.shape
    ph, pw = (-h) % N, (-w) % N
    img = jnp.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    hp, wp = img.shape
    target = img.reshape(N, hp // N, N, wp // N).mean(axis=(1, 3))
    return target / jnp.sum(target) * (rho_0 * N)

target = load_target(Path(__file__).resolve().parent / "logo.png", num_cells)

# the optimized parameters are the three initial velocity components
def build_state(velocity):
    return construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = velocity[0],
        velocity_y = velocity[1],
        velocity_z = velocity[2],
        gas_pressure = p,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        interface_magnetic_field_x = bxb,
        interface_magnetic_field_y = byb,
        interface_magnetic_field_z = bzb,
    )

def loss_fn(velocity):
    final_state = time_integration(build_state(velocity), config, params, registered_variables)
    projection = jnp.sum(final_state[registered_variables.density_index], axis=2)
    return jnp.mean((projection - target) ** 2)

value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

# optimize the initial velocity field with Adam
velocity = jnp.zeros((3,) + (num_cells,) * 3)
optimizer = optax.adam(learning_rate=3e-3)
opt_state = optimizer.init(velocity)

losses = []
for step in range(50):
    loss, grads = value_and_grad(velocity)
    updates, opt_state = optimizer.update(grads, opt_state)
    velocity = optax.apply_updates(velocity, updates)
    losses.append(float(loss))
    print(f"step {step:3d}: loss = {loss:.4e}")

# plot the loss curve, the target and the reconstructed projection
final_state = time_integration(build_state(velocity), config, params, registered_variables)
projection = jnp.sum(final_state[registered_variables.density_index], axis=2)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(losses)
axs[0].set_yscale("log")
axs[0].set_title("loss")
axs[1].imshow(np.asarray(target).T, origin="lower", cmap="inferno")
axs[1].set_title("target")
axs[2].imshow(np.asarray(projection).T, origin="lower", cmap="inferno")
axs[2].set_title("reconstruction")
fig.savefig(figures_dir / "field_level_inference.png", dpi=200, bbox_inches="tight")
