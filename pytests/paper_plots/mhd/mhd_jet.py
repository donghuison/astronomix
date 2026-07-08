"""Magnetically driven jet (3D MHD) — paper figure, finite-difference solver.

A magnetic tower launched from a magnetized central region in an initially
uniform medium (vector-potential initial condition), evolved with the
finite-difference (WENO + constrained-transport) solver.  This is the
FD-only version of the jet test.

The figure is a density slice through the jet axis at t = 5.0.  The full 3D
density field is cached under ``data/`` so the figure can be re-sliced /
re-styled without re-running the (expensive) 256^3 simulation.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/mhd/mhd_jet.py [--res N] [--rerun]
"""

# general
import sys

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    OPEN_BOUNDARY,
    PALLAS,
)

# astronomix containers
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    PositivityConfig,
)

# astronomix functions
from astronomix import (
    get_registered_variables,
    construct_primitive_state,
    time_integration,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.initial_condition_generation.magnetic_field_from_vector_potential import (
    setup_magnetic_fields_from_vector_potential,
)

# shared figure helpers
from _common import DATA_DIR
from _common import FIG_DIR
from _common import rerun_requested
from _common import mhd_registered_variables


GAMMA = 5.0 / 3.0
BOX_SIZE = 24.0
T_END = 5.0
C_CFL = 0.8
RHO_0 = 1.0
P_0 = 1.0
A0 = 20.0


def _resolution_from_argv():
    """Return the per-dimension resolution requested via ``--res N`` (default 256)."""
    if "--res" in sys.argv:
        return int(sys.argv[sys.argv.index("--res") + 1])
    return 256


def cache_path(num_cells):
    """Return the cache path for the FD jet run at ``num_cells``^3."""
    return DATA_DIR / f"mhd_jet_fd_{num_cells}.npz"


def simulate(num_cells):
    """Run the finite-difference MHD jet simulation at ``num_cells``^3.

    Sets up the magnetic-tower initial condition from a vector potential in an
    initially uniform medium, evolves it to ``T_END`` with the finite-difference
    (WENO + constrained-transport) solver on the Pallas backend, and returns the
    final density field.

    Args:
        num_cells: The per-dimension resolution of the cubic grid.

    Returns:
        The final density field as a numpy array of shape
        ``(num_cells, num_cells, num_cells)``.
    """

    # -------------------------------------------------------------
    # ============ ↓ Grid and solver configuration ↓ =============
    # -------------------------------------------------------------

    grid_spacing = BOX_SIZE / num_cells
    center = BOX_SIZE / 2.0

    config = SimulationConfig(
        positivity_config=PositivityConfig(default_positivity_protection=True),
        solver_mode=FINITE_DIFFERENCE,
        # FD/WENO runs ~10x faster through the Pallas (Triton) backend;
        # bit-compatible with native JAX.
        backend=PALLAS,
        pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True,
        pallas_interpret=False,
        grid_spacing=grid_spacing,
        mhd=True,
        progress_bar=True,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
            BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
            BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
    )
    rv = get_registered_variables(config)

    # -------------------------------------------------------------
    # ============ ↑ Grid and solver configuration ↑ =============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =============== ↓ Initial magnetic tower ↓ =================
    # -------------------------------------------------------------

    # The magnetic field is seeded from a vector potential so that the
    # constrained-transport solver starts from a divergence-free field.
    def jet_vector_potential(X, Y, Z):
        r = jnp.sqrt((X - center) ** 2 + (Y - center) ** 2 + (Z - center) ** 2)
        A_x = -jnp.exp(-r**2) * (Y - center)
        A_y = jnp.exp(-r**2) * (X - center)
        A_z = 0.5 * A0 * jnp.exp(-r**2)
        return A_x, A_y, A_z

    B_x, B_y, B_z, bxb, byb, bzb = setup_magnetic_fields_from_vector_potential(
        config=config,
        vector_potential_func=jet_vector_potential,
    )

    # -------------------------------------------------------------
    # =============== ↑ Initial magnetic tower ↑ =================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Uniform background and evolve ↓ =============
    # -------------------------------------------------------------

    shape = (num_cells, num_cells, num_cells)
    rho = jnp.ones(shape) * RHO_0
    zeros = jnp.zeros(shape)
    p = jnp.ones(shape) * P_0

    params = SimulationParams(
        C_cfl=C_CFL,
        dt_max=0.1,
        t_end=T_END,
        gamma=GAMMA,
        minimum_density=1e-2 * RHO_0,
        minimum_pressure=1e-2 * P_0,
    )

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=rv,
        density=rho,
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        gas_pressure=p,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
    )

    config = finalize_config(config, initial_state.shape)
    final_state = time_integration(initial_state, config, params, rv)

    density = np.asarray(final_state[rv.density_index])
    return density

    # -------------------------------------------------------------
    # ============ ↑ Uniform background and evolve ↑ =============
    # -------------------------------------------------------------


def get_run(num_cells, rerun):
    """Return the FD jet density field, from cache or by (re-)running.

    Args:
        num_cells: The per-dimension resolution of the run.
        rerun: When True, ignore any cache and re-run the simulation.

    Returns:
        The final density field as a numpy array.
    """
    path = cache_path(num_cells)
    if path.exists() and not rerun:
        return np.load(path)["density"]
    print(f"running FD MHD jet at {num_cells}^3 ...")
    density = simulate(num_cells)
    np.savez(path, density=density)
    print(f"  cached -> {path}")
    return density


def plot(num_cells, rerun):
    """Render the jet-axis density slice figure.

    Loads (or regenerates) the cached density field, slices it through the jet
    axis (the x-z plane at mid-y) and saves the figure as both ``.png`` and
    ``.pdf`` (``mhd_jet_fd_{num_cells}``).

    Args:
        num_cells: The per-dimension resolution of the run to plot.
        rerun: When True, force the underlying simulation to be re-run.
    """
    density = get_run(num_cells, rerun)
    y_index = num_cells // 2

    # Slice through the jet axis: the x-z plane at mid-y.
    density_slice = density[:, y_index, :].T

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    image = ax.imshow(
        density_slice,
        origin="lower",
        extent=(0, BOX_SIZE, 0, BOX_SIZE),
        cmap="YlOrRd",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(image, cax=cax, label="density")

    plt.tight_layout()
    out = FIG_DIR / f"mhd_jet_fd_{num_cells}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    num_cells = _resolution_from_argv()
    plot(num_cells, rerun_requested())
