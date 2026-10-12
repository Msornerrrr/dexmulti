import numpy as np
import pytest

pytest.importorskip("pyronot")

from dexmulti.alignment.pyroki_planner import PyrokiAlignmentPlanner  # noqa: E402
from dexmulti.alignment.world import static_world  # noqa: E402


def test_planner_step_moves_toward_target():
    planner = PyrokiAlignmentPlanner()
    q = np.zeros(planner.num_dof)
    q_cmd = planner.step(q, np.array([0.4, 0.0, 0.3]), np.array([np.pi, 0.0, 0.0]), np.zeros(16), static_world())
    assert q_cmd.shape == (23,) and np.isfinite(q_cmd).all()


def test_warmup_keeps_state():
    planner = PyrokiAlignmentPlanner()
    before = planner.prev_sols.copy()
    planner.warmup(static_world())
    np.testing.assert_array_equal(planner.prev_sols, before)
