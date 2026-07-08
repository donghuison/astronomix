"""Sensitivity / gradient paper figures.

Regenerates ``gradient_3d_gaussian`` (AD vs analytic sensitivity of a 3D Gaussian
initial condition) and ``gradient_convergence_test`` (continuous convergence of
the AD gradient across solvers) by running the sensitivity example.
"""

# testing
from _common import run_example_and_collect


def main():
    run_example_and_collect(
        "sensitivity.py",
        {
            "gradient_3d_gaussian.png": "gradient_3d_gaussian.png",
            "gradient_convergence_test.png": "gradient_convergence_test.png",
        },
    )


if __name__ == "__main__":
    main()
