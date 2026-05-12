"""Tests for the preprocessing step."""

from __future__ import annotations

import numpy as np

from simplenet.case import BRANCH_COLUMNS, BUS_COLUMNS, GEN_COLUMNS, PowerCase
from simplenet.preprocess import preprocess


def _stub_case() -> PowerCase:
    bus = np.zeros((4, BUS_COLUMNS))
    bus[:, 0] = [1, 2, 3, 4]
    bus[:, 1] = [3, 1, 4, 1]
    bus[2, 5] = 10.0

    branch = np.zeros((4, BRANCH_COLUMNS))
    branch[:, 0] = [1, 2, 3, 1]
    branch[:, 1] = [2, 3, 4, 4]
    branch[:, 3] = [0.1, 0.2, 0.3, 0.4]
    branch[:, 10] = [1, 1, 1, 0]

    gen = np.zeros((3, GEN_COLUMNS))
    gen[:, 0] = [1, 3, 4]
    gen[:, 7] = [1, 1, 1]

    return PowerCase(bus=bus, branch=branch, gen=gen)


def test_preprocess_drops_isolated_and_oos() -> None:
    case = _stub_case()
    excluded = np.array([2, 3, 4])
    new_case, new_excluded, stats = preprocess(case, excluded)

    assert stats.isolated_buses == 1
    assert stats.branches_removed >= 2
    assert stats.generators_removed == 1

    assert new_case.n_bus() == 3
    assert 3 not in new_case.bus[:, 0]
    np.testing.assert_array_equal(np.sort(new_excluded), [2.0, 4.0])

    assert 3 not in new_case.gen[:, 0]


def test_preprocess_does_not_mutate_input() -> None:
    case = _stub_case()
    original_bus = case.bus.copy()
    preprocess(case, np.array([2]))
    np.testing.assert_array_equal(case.bus, original_bus)
