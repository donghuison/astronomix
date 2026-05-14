"""
Here we expose the RHS of the solver for analysis purposes.
"""

from functools import partial

import jax

from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_x, _weno_flux_y, _weno_flux_z
from astronomix._fluid_equations._equations import primitive_state_from_conserved
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._physics_modules._viscosity._viscosity import fd_viscosity_source
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.option_classes.simulation_config import CONSERVATIVE_GAS_STATE, FINITE_DIFFERENCE, FINITE_VOLUME
from astronomix.time_stepping._utils import _pad, _unpad


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _exposed_rhs(conserved_state, params, config, registered_variables):
    """
    Computes the right-hand side of the fluid equations.
    TODO: Refactor with simulator RHS directly.
    """

    # pad the state with ghost cells according to the boundary conditions
    conserved_state = _pad(conserved_state, config)

    # apply boundary conditions on the padded state
    conserved_state = _boundary_handler(
        conserved_state, config, registered_variables, params, CONSERVATIVE_GAS_STATE
    )

    # retrieve the primitive state from the conserved state
    primitive_state = primitive_state_from_conserved(
        conserved_state,
        params.gamma,
        config,
        registered_variables,
    )

    if config.solver_mode == FINITE_DIFFERENCE:
        if not config.mhd:
            dF_x = _weno_flux_x(conserved_state, params, config, registered_variables)

            if config.dimensionality >= 2:
                dF_y = _weno_flux_y(conserved_state, params, config, registered_variables)

            if config.dimensionality == 3:
                dF_z = _weno_flux_z(conserved_state, params, config, registered_variables)

            # Calculate RHS for conserved fluid variables
            if config.dimensionality == 1:
                rhs_q = -1/config.grid_spacing * (
                    (dF_x - _shift(dF_x, 1, axis=1))
                )
            elif config.dimensionality == 2:
                rhs_q = -1/config.grid_spacing * (
                    (dF_x - _shift(dF_x, 1, axis=1))
                    + (dF_y - _shift(dF_y, 1, axis=2))
                )
            elif config.dimensionality == 3:
                rhs_q = -1/config.grid_spacing * (
                    (dF_x - _shift(dF_x, 1, axis=1))
                    + (dF_y - _shift(dF_y, 1, axis=2))
                    + (dF_z - _shift(dF_z, 1, axis=3))
                )
        else:
            raise NotImplementedError("Extracted RHS currently only implemented for hydrodynamics.")
        
        # add source terms to the RHS
        if config.diffusion:
            rhs_q = fd_viscosity_source(primitive_state, params, config, registered_variables)

        # further physics modules are currently not implemented in the extracted RHS

    elif config.solver_mode == FINITE_VOLUME:
        raise NotImplementedError("Extracted RHS currently only implemented for finite difference solver.")

    # unpad the RHS to remove the ghost cells
    rhs_q = _unpad(rhs_q, config)

    return rhs_q