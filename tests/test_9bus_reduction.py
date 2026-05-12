"""End-to-end regression test that replicates ``Example_9bus.m``.

The MATLAB reference (``matlab/NetworkReduction2/Example_9bus.m``)
states explicitly:

    There are 4 equivalent branches generated in the reduction process
    branch between bus: 2-7, 2-9, 4-6, 7-9, all equivalent branches
    have circuit number 99.

    Link gives generator bus mapping showing how generators are moved.
    E.g: Generator on bus 1 is moved to bus 4, all other generators are
    not moved since they were on retained buses.
"""

from __future__ import annotations

import numpy as np

from simplenet.case import F_BUS, T_BUS, PowerCase
from simplenet.pipeline import reduce_network


def _branch_pair(branch: np.ndarray, fbus: int, tbus: int) -> np.ndarray:
    f = branch[:, F_BUS].astype(int)
    t = branch[:, T_BUS].astype(int)
    mask = ((f == fbus) & (t == tbus)) | ((f == tbus) & (t == fbus))
    return np.where(mask)[0]


def test_9bus_retains_6_buses(case9: PowerCase) -> None:
    result = reduce_network(case9, [1, 5, 8], pf_flag=False)
    assert result.reduced_case.n_bus() == 6
    np.testing.assert_array_equal(
        np.sort(result.reduced_case.bus[:, 0].astype(int)),
        np.array([2, 3, 4, 6, 7, 9]),
    )


def test_9bus_equivalent_branches(case9: PowerCase) -> None:
    """Expected equivalent branches between (2,7), (2,9), (4,6), (7,9)."""

    result = reduce_network(case9, [1, 5, 8], pf_flag=False)
    eq_value = result.eq_bcirc_value
    assert eq_value == 99

    eq_mask = result.bcirc == eq_value
    eq_branches = result.reduced_case.branch[eq_mask]
    eq_pairs = {tuple(sorted([int(r[F_BUS]), int(r[T_BUS])])) for r in eq_branches}
    expected_pairs = {(2, 7), (2, 9), (4, 6), (7, 9)}
    assert eq_pairs == expected_pairs


def test_9bus_generator_on_bus_1_moves_to_4(case9: PowerCase) -> None:
    result = reduce_network(case9, [1, 5, 8], pf_flag=False)
    gen_buses = result.reduced_case.gen[:, 0].astype(int).tolist()
    assert 1 not in gen_buses
    assert 4 in gen_buses

    link = result.link
    bus_1_row = link[link[:, 0] == 1]
    if bus_1_row.size:
        assert int(bus_1_row[0, 1]) == 4


def test_9bus_branch_shunts_zeroed(case9: PowerCase) -> None:
    """``MakeMPCr.m`` line 106 sets ``mpc.branch[:, 5] = 0``."""

    result = reduce_network(case9, [1, 5, 8], pf_flag=False)
    np.testing.assert_allclose(result.reduced_case.branch[:, 4], 0.0)


def test_9bus_dc_pf_consistency(case9: PowerCase) -> None:
    """Solving DC PF on the full and reduced models gives angles that
    agree up to a constant shift (the two models have different slack
    buses; full uses bus 1 (eliminated), reduced auto-promotes the
    first remaining PV bus)."""

    from simplenet.dcpf import run_dcpf

    result = reduce_network(case9, [1, 5, 8], pf_flag=True)
    reduced_pf = run_dcpf(result.reduced_case)
    assert reduced_pf.converged

    full_pf = run_dcpf(case9)
    full_bus_ids = case9.bus[:, 0].astype(int)
    full_theta = dict(zip(full_bus_ids, full_pf.theta, strict=False))

    red_bus_ids = result.reduced_case.bus[:, 0].astype(int)
    ref_bus = int(red_bus_ids[0])
    shift = full_theta[ref_bus] - reduced_pf.theta[0]
    for k, bus_id in enumerate(red_bus_ids):
        np.testing.assert_allclose(reduced_pf.theta[k] + shift, full_theta[int(bus_id)], atol=1e-6)


def test_no_excluded_buses_returns_input(case9: PowerCase) -> None:
    """An empty exclusion list should leave the case unchanged."""

    result = reduce_network(case9, [], pf_flag=False)
    assert result.reduced_case.n_bus() == case9.n_bus()
    assert result.reduced_case.n_branch() == case9.n_branch()
    np.testing.assert_array_equal(result.bcirc, np.ones(case9.n_branch(), dtype=int))
