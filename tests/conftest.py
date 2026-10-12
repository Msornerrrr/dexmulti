from pathlib import Path

import numpy as np
import pytest

from dexmulti.library import load_library

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def grasp_pull_library():
    """Library of the paper's Grasp+Pull session (38 demos), as used on the robot."""
    return load_library(FIXTURES / "grasp_pull_library.npz")


@pytest.fixture
def rng():
    return np.random.default_rng(0)
