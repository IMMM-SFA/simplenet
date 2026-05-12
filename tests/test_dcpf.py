"""Tests for the standalone DC power flow."""

from __future__ import annotations

import numpy as np

from simplenet.case import (
    BRANCH_COLUMNS,
    BUS_COLUMNS,
    GEN_COLUMNS,
    PowerCase,
)
from simplenet.dcpf import run_dcpf


def _three_bus_case() -> PowerCase:
    bus = np.zeros((3, BUS_COLUMNS))
    bus[:, 0] = [1, 2, 3]
    bus[:, 1] = [3, 1, 1]
    bus[:, 2] = [0, 100, 0]
    bus[:, 7] = 1.0

    branch = np.zeros((3, BRANCH_COLUMNS))
    branch[:, 0] = [1, 2, 1]
    branch[:, 1] = [2, 3, 3]
    branch[:, 3] = [0.1, 0.1, 0.2]
    branch[:, 10] = 1

    gen = np.zeros((2, GEN_COLUMNS))
    gen[0, 0] = 1
    gen[0, 7] = 1
    gen[1, 0] = 3
    gen[1, 1] = 100
    gen[1, 7] = 1

    return PowerCase(bus=bus, branch=branch, gen=gen)


def test_three_bus_dcpf_balances() -> None:
    case = _three_bus_case()
    result = run_dcpf(case)

    assert result.converged
    np.testing.assert_allclose(result.theta[0], 0.0)

    base = case.base_mva
    bus_lookup = {int(b): i for i, b in enumerate(case.bus[:, 0].astype(int))}
    p_gen_per_bus = np.zeros(3)
    for g in case.gen:
        p_gen_per_bus[bus_lookup[int(g[0])]] += g[1] / base
    p_gen_per_bus[0] = result.p_gen[0] / base
    p_load_per_bus = case.bus[:, 2] / base

    from simplenet.ymatrix import build_b_for_dcpf

    B, p_shift = build_b_for_dcpf(case)
    p_inj = B @ result.theta
    np.testing.assert_allclose(p_inj + p_shift / base, p_gen_per_bus - p_load_per_bus, atol=1e-9)
