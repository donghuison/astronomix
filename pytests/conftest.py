"""Pytest configuration for the astronomix test / benchmark suite."""


def pytest_addoption(parser):
    parser.addoption(
        "--reproduce-paper",
        action="store_true",
        default=False,
        help="regenerate every methods-paper figure under pytests/paper_plots "
             "(slow, requires a GPU).",
    )
