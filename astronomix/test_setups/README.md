# Standartized Test Setups

To test fluid simulators a broad range of test setups is required.
In `astronomix` such test setups are provided here in a standarized format:

```python
"""
# Test Setup Name

Description of the test setup.

## References

- References to papers describing the test setup.

"""

def setup_test_name(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
) -> tuple[STATE_TYPE, SimulationConfig, SimulationParams]:
    """
    Function to set up the test.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.

    Returns:
        state: Initial state of the simulation.
        config: Updated simulation configuration.
        params: Updated simulation parameters.
    """
    # Code to set up the test goes here.

# optional, sometimes consistency checks
# are also possible without a reference solution
# e.g. conservation of mass
def test_name_solution(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
) -> STATE_TYPE:
    """
    Reference solution for the test setup, if available.
    This might eigher be an analytical solution or a previously 
    saved numerical result.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.

    Returns:
        state: Reference solution state of the simulation.
    """
```