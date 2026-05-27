"""
Here we implement a simple mixing layer
cooling based on Lancaster 2026.
"""

from functools import partial

import jax
import jax.numpy as jnp

from astronomix.option_classes.simulation_params import SimulationParams

from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix._modules._cooling.cooling_options import COOLING_CURVE_TYPE, EXPLICIT_COOLING, IMPLICIT_COOLING, CoolingConfig, CoolingCurveConfig, MixingCoolingParams
from astronomix.option_classes.simulation_config import FIELD_TYPE, STATE_TYPE

def _cooling_rate(
    temperature: jnp.ndarray,
    density: jnp.ndarray,
    mixing_cooling_params: MixingCoolingParams,
    gamma: float = 5 / 3,
    beta_low: float = -2,
    beta_high: float = 3,
) -> jnp.ndarray:
    """
    Returns dT/dt in units where we have simplified
    T = P / rho, so T is in units of velocity^2.

    xi: float, # xi = t_sh / t_coolmin, in Fig. 3 of Lancaster et al. 2026: \in {10, 100, 1000}
    mach_number: float, # in Lancaster et al. 2026: \in {1/2, 1/8}
    chi: float = 1e2,
    """

    xi = mixing_cooling_params.xi
    mach_number = mixing_cooling_params.mach_number
    chi = mixing_cooling_params.density_contrast

    # assume L_box = 1.0
    L_box = 1.0

    # assume rho_0 = P_0 = 1.0
    rho0 = 1.0
    P0 = 1.0

    # T = P / rho
    P = temperature * density
    # or P = P_0 (isobaric)

    # adiabatic sound speed in the hot medium
    c_s_hot = jnp.sqrt(gamma * P0 / rho0)

    # relative shear velocity, mach number M = v_rel / c_s_hot
    v_rel = mach_number * c_s_hot

    # simplification: T = P / rho
    # (so T in units velocity^2)
    # such that T = (gamma - 1) * e

    t_sh = L_box / v_rel
    t_coolmin = t_sh / xi

    T_hot = P0 / rho0
    T_cold = T_hot / chi

    # peak of the cooling rate
    T_pk = (T_cold ** 2 * T_hot) ** (1/3)

    # peak cooling rate
    edot_max = P0 / (gamma - 1) / t_coolmin * (P / P0) ** 2

    # power law index
    beta = jnp.where(
        temperature < T_pk,
        beta_low,
        beta_high
    )

    # cooling rate as a function of temperature
    edot_cooling = edot_max * (temperature / T_pk) ** (-beta)

    # heating rate as a function of temperature
    T_lim = 1.05 * T_hot
    c_heat = (T_cold / T_pk) ** ((beta_high - beta_low) * (1 + jnp.log(T_cold / T_pk) / jnp.log(chi)))
    alpha_heat = (beta_low - beta_high) * (jnp.log(T_cold / T_pk) / jnp.log(chi)) - beta_high
    f = jnp.where(
        temperature < T_lim,
        (temperature / T_pk) ** alpha_heat,
        (T_lim / T_pk) ** alpha_heat * (temperature / T_lim) ** (-beta_high - 0.5)
    )
    edot_heating = c_heat * edot_max * f

    # net cooling rate
    edot_net = edot_heating - edot_cooling

    return (gamma - 1) * edot_net / density

@partial(jax.jit, static_argnames=("cooling_curve_config",))
def update_temperature_explicit(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    gamma: float,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    return (
        temperature
        + _cooling_rate(
            temperature,
            density,
            cooling_curve_params,
            gamma,
        )
        * time_step
    )

@partial(jax.jit, static_argnames=("cooling_curve_config",))
def update_temperature_implicit(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    gamma: float,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:

    def implicit_eq(T_new):
        return (temperature
        + _cooling_rate(
            T_new,
            density,
            cooling_curve_params,
            gamma,
        ) * time_step)

    # use a simple fixed point iteration
    # - maybe do newton or bisection method later
    max_iter = 50
    tol = 1e-6

    def cond_fun(state):
        i, T_old = state
        T_candidate = implicit_eq(T_old)
        diff = jnp.max(jnp.abs(T_candidate - T_old))
        return (i < max_iter) & (diff > tol)

    def body_fun(state):
        i, T_old = state
        T_new = implicit_eq(T_old)
        return (i + 1, T_new)

    state = (0, temperature)
    _, T_final = jax.lax.while_loop(cond_fun, body_fun, state)
    return T_final


@partial(jax.jit, static_argnames=("cooling_config", "registered_variables"))
def update_pressure_by_cooling_mixing(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
    cooling_config: CoolingConfig,
    simulation_params: SimulationParams,
    time_step: float,
) -> STATE_TYPE:
    
    # here assume T = P / rho
    
    cooling_curve_config = cooling_config.cooling_curve_config

    # get the parameters
    cooling_params = simulation_params.cooling_params
    gamma = simulation_params.gamma

    # get the density and pressure
    density = primitive_state[registered_variables.density_index]
    pressure = primitive_state[registered_variables.pressure_index]

    # get the temperature
    temperature = pressure / density

    if cooling_config.cooling_method == IMPLICIT_COOLING:
        new_temperature = update_temperature_implicit(
            density,
            temperature,
            time_step,
            gamma,
            cooling_curve_config,
            cooling_params.cooling_curve_params,
        )
    elif cooling_config.cooling_method == EXPLICIT_COOLING:
        new_temperature = update_temperature_explicit(
            density,
            temperature,
            time_step,
            gamma,
            cooling_curve_config,
            cooling_params.cooling_curve_params,
        )

    new_temperature = jnp.where(
        (new_temperature > cooling_params.floor_temperature),
        new_temperature,
        temperature,
    )

    # update the pressure
    new_pressure = new_temperature * density

    # set the new pressure
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        new_pressure
    )

    # return the updated primitive state
    return primitive_state