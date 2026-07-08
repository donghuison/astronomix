"""KHI eigenmode-initialization paper figures.

Regenerates the three eigenmode figures used in the methods paper — the
eigenvalue spectrum coloured by effective transverse wavenumber, the growth
transient (eigenmode seed vs velocity seed), and the developed final-state
comparison — by running the ``eigen_initialization`` example.
"""

# testing
from _common import run_example_and_collect


def main():
    run_example_and_collect(
        "eigen_initialization.py",
        {
            "eigenvalue_spectrum.png": "eigenvalue_spectrum_final.png",
            "eigenmode_transient_test.png": "eigenmode_transient_test.png",
            "eigenmode_final_states.png": "eigenmode_final_states.png",
        },
    )


if __name__ == "__main__":
    main()
