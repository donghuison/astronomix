"""
Trubulent radiative mixing layer simulations.

Setup based on Lancaster 2026 (unpublished):

Basic setup:

- box with dimensions (Lx, Ly, Lz) = (L_box, L_box, 1.5 * L_box), L_box = 1.0
- boundaries
    - x: periodic
    - y: periodic
    - z: bottom outflow, top fixed to U_hot (see below)
- the state z > z_center is the hot state
- the state z < z_center is the cold state
- initialization with two constant states along z,
  with a smooth transition at z_center = L_box / 2
    - a tanh profile of thickness \Delta x / 2, with
      \Delta x = L_box / num_cells
- these states are given by
    - U_hot = (\rho, v_x, v_y, v_z, P)_hot = (\rho_0, v_rel / 2, 0, 0, P_0)
    - U_cold = (\rho, v_x, v_y, v_z, P)_cold = (χ \rho_0, -v_rel / 2, 0, 0, P_0)
  with \rho_0 = P_0 = 1
- as a simplification, T := P / \rho is used in the setup
- the adiabatic sound speed in the hot medium is
    - c_hot = sqrt(γ P_0 / \rho_0) = sqrt(γ)
- the temperature of the two phases are
    - T_hot = P_0 / \rho_0 = 1
    - T_cold = P_0 / (χ \rho_0) = 1 / χ

Cooling function: TODO

"""

# TODO