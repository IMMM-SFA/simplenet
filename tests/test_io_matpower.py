"""Tests for the MATPOWER ``.m`` parser."""

from __future__ import annotations

import numpy as np

from simplenet.case import PowerCase


def test_load_case9(case9: PowerCase) -> None:
    """Parser handles inline matrix literals from test_9bus_case.m."""

    assert case9.version == "2"
    assert case9.base_mva == 100.0
    assert case9.bus.shape == (9, 13)
    assert case9.gen.shape == (3, 21)
    assert case9.branch.shape == (9, 13)
    assert case9.gencost is not None
    assert case9.gencost.shape == (3, 7)

    np.testing.assert_array_equal(case9.bus[:, 0], np.arange(1, 10))
    np.testing.assert_allclose(case9.bus[case9.bus[:, 0] == 5, 2], [90])
    np.testing.assert_allclose(case9.bus[case9.bus[:, 0] == 7, 2], [100])
    np.testing.assert_allclose(case9.bus[case9.bus[:, 0] == 9, 2], [125])


def test_pypower_roundtrip(case9: PowerCase) -> None:
    """``PowerCase`` round-trips through pypower-style dicts."""

    d = case9.to_pypower()
    assert d["baseMVA"] == case9.base_mva
    rebuilt = PowerCase.from_pypower(d)
    np.testing.assert_array_equal(rebuilt.bus, case9.bus)
    np.testing.assert_array_equal(rebuilt.gen, case9.gen)
    np.testing.assert_array_equal(rebuilt.branch, case9.branch)
    assert rebuilt.base_mva == case9.base_mva


def test_parser_ignores_readmatrix(tmp_path) -> None:
    """A case file that delegates to xlsx via ``readmatrix`` does not crash.

    Mirrors ``case_ACTIVSg10kCopy2.m`` style files where the matrices
    are populated from an external xlsx.
    """

    from simplenet.io.matpower import load_m

    path = tmp_path / "case_delegating.m"
    path.write_text(
        """
function mpc = case_delegating
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = readmatrix('matlab2.xlsx','Sheet','Bus');
mpc.gen = readmatrix('matlab2.xlsx','Sheet','Gen');
mpc.branch = readmatrix('matlab2.xlsx','Sheet','Branch');
""".strip()
    )
    case = load_m(path)
    assert case.base_mva == 100.0
    assert case.bus.shape == (0, 13)
    assert case.gen.shape == (0, 21)
    assert case.branch.shape == (0, 13)
