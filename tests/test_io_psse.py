"""Tests for the PSS/E ``.RAW`` parser."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np

from simplenet.case import (
    BR_X,
    BS,
    BUS_I,
    BUS_TYPE,
    F_BUS,
    GEN_BUS,
    GS,
    ISOLATED_BUS,
    PD,
    PG,
    PMAX,
    PQ_BUS,
    PV_BUS,
    QD,
    REF_BUS,
    T_BUS,
    TAP,
    PowerCase,
)
from simplenet.io.psse import load_raw
from simplenet.pipeline import reduce_network


def test_load_raw_9bus_matches_matpower_case9(fixtures_dir: Path) -> None:
    """The 9-bus PSS/E fixture parses to the same network as ``case9.m``."""

    case = load_raw(fixtures_dir / "case9.raw")
    assert isinstance(case, PowerCase)
    assert case.base_mva == 100.0
    assert case.bus.shape[0] == 9
    assert case.gen.shape[0] == 3
    assert case.branch.shape[0] == 9

    np.testing.assert_array_equal(case.bus[:, BUS_I].astype(int), np.arange(1, 10))
    assert case.bus[case.bus[:, BUS_I] == 1, BUS_TYPE][0] == REF_BUS
    assert case.bus[case.bus[:, BUS_I] == 2, BUS_TYPE][0] == PV_BUS
    assert case.bus[case.bus[:, BUS_I] == 4, BUS_TYPE][0] == PQ_BUS

    np.testing.assert_allclose(case.bus[case.bus[:, BUS_I] == 5, PD], [90])
    np.testing.assert_allclose(case.bus[case.bus[:, BUS_I] == 7, PD], [100])
    np.testing.assert_allclose(case.bus[case.bus[:, BUS_I] == 9, PD], [125])
    np.testing.assert_allclose(case.bus[case.bus[:, BUS_I] == 5, QD], [30])

    np.testing.assert_allclose(
        sorted(case.gen[:, GEN_BUS].astype(int).tolist()), [1, 2, 3]
    )
    pg_by_bus = {int(b): float(p) for b, p in zip(case.gen[:, GEN_BUS], case.gen[:, PG], strict=False)}
    assert pg_by_bus[1] == 0.0
    assert pg_by_bus[2] == 163.0
    assert pg_by_bus[3] == 85.0
    pmax_by_bus = {int(b): float(p) for b, p in zip(case.gen[:, GEN_BUS], case.gen[:, PMAX], strict=False)}
    assert pmax_by_bus[2] == 300.0

    edges = {
        tuple(sorted([int(r[F_BUS]), int(r[T_BUS])])): float(r[BR_X])
        for r in case.branch
    }
    assert edges[(1, 4)] == 0.0576
    np.testing.assert_allclose(edges[(4, 5)], 0.092)
    np.testing.assert_allclose(edges[(8, 9)], 0.161)


def test_load_raw_9bus_drives_reduction(fixtures_dir: Path) -> None:
    """End-to-end: a PSS/E-loaded case reduces to the same 9-bus result."""

    case = load_raw(fixtures_dir / "case9.raw")
    result = reduce_network(case, [1, 5, 8], pf_flag=False)

    retained = sorted(int(b) for b in result.reduced_case.bus[:, BUS_I])
    assert retained == [2, 3, 4, 6, 7, 9]

    eq_pairs = {
        tuple(sorted([int(r[F_BUS]), int(r[T_BUS])]))
        for r, bc in zip(result.reduced_case.branch, result.bcirc, strict=False)
        if int(bc) == result.eq_bcirc_value
    }
    assert eq_pairs == {(2, 7), (2, 9), (4, 6), (7, 9)}

    gen_buses = sorted(int(b) for b in result.reduced_case.gen[:, GEN_BUS])
    assert gen_buses == [2, 3, 4]


def test_load_raw_two_winding_transformer(tmp_path: Path) -> None:
    """A two-winding transformer becomes a branch with tap ratio and angle."""

    raw = dedent(
        """\
        0,   100.00, 33, 0, 1, 60.00     / PSS/E 33 raw
        2-bus transformer test
        sanity-check tap + phase shifter
             1,'B1          ', 230.0,3,    1,    1,    1,1.0000,    0.00, 1.10000, 0.90000, 1.10000, 0.90000
             2,'B2          ', 115.0,1,    1,    1,    1,1.0000,    0.00, 1.10000, 0.90000, 1.10000, 0.90000
        0 /End of Bus data, Begin Load data
        0 /End of Load data, Begin Fixed shunt data
        0 /End of Fixed shunt data, Begin Generator data
        0 /End of Generator data, Begin Branch data
        0 /End of Branch data, Begin Transformer data
             1,     2,     0,'1 ',1,1,1,0.00000E+0,0.00000E+0,2,'XFMR-1 2 1   ',1,   1,1.0000
         0.00000E+0, 5.00000E-2,   100.00
         1.05000,  230.000,  -3.000,   200.00,   200.00,   200.00,0,    0,1.10000,0.90000,1.10000,0.90000,  33,0,0.00000E+0,0.00000E+0,0.00000E+0
         1.00000,  115.000
        0 /End of Transformer data, Begin Area interchange data
        0 /End of Area interchange data, Begin Two-terminal dc line data
        0 /End of Two-terminal dc line data, Begin VSC dc line data
        0 /End of VSC dc line data, Begin Impedance correction data
        0 /End of Impedance correction data, Begin Multi-terminal dc line data
        0 /End of Multi-terminal dc line data, Begin Multi-section line data
        0 /End of Multi-section line data, Begin Zone data
        0 /End of Zone data, Begin Inter-area transfer data
        0 /End of Inter-area transfer data, Begin Owner data
        0 /End of Owner data, Begin FACTS device data
        0 /End of FACTS device data, Begin Switched shunt data
        0 /End of Switched shunt data
        Q
        """
    )
    p = tmp_path / "xfmr.raw"
    p.write_text(raw, encoding="utf-8")
    case = load_raw(p)
    assert case.branch.shape[0] == 1
    np.testing.assert_allclose(case.branch[0, BR_X], 0.05)
    np.testing.assert_allclose(case.branch[0, TAP], 1.05)
    np.testing.assert_allclose(case.branch[0, 9], -3.0)


def test_load_raw_switched_shunt_folds_into_bs(tmp_path: Path) -> None:
    """A switched shunt's BINIT becomes part of the bus ``Bs``."""

    raw = dedent(
        """\
        0,   100.00, 33, 0, 1, 60.00     / PSS/E 33 raw
        switched shunt test
        single bus
             1,'B1          ', 230.0,3,    1,    1,    1,1.0000,    0.00, 1.10000, 0.90000, 1.10000, 0.90000
        0 /End of Bus data, Begin Load data
        0 /End of Load data, Begin Fixed shunt data
             1,'1 ',1,   1.000,  12.000
        0 /End of Fixed shunt data, Begin Generator data
        0 /End of Generator data, Begin Branch data
        0 /End of Branch data, Begin Transformer data
        0 /End of Transformer data, Begin Area interchange data
        0 /End of Area interchange data, Begin Two-terminal dc line data
        0 /End of Two-terminal dc line data, Begin VSC dc line data
        0 /End of VSC dc line data, Begin Impedance correction data
        0 /End of Impedance correction data, Begin Multi-terminal dc line data
        0 /End of Multi-terminal dc line data, Begin Multi-section line data
        0 /End of Multi-section line data, Begin Zone data
        0 /End of Zone data, Begin Inter-area transfer data
        0 /End of Inter-area transfer data, Begin Owner data
        0 /End of Owner data, Begin FACTS device data
        0 /End of FACTS device data, Begin Switched shunt data
             1,1,0,1, 1.10000, 0.90000,   0,  100.0,'        ',  20.00000,1,  5.00000
        0 /End of Switched shunt data
        Q
        """
    )
    p = tmp_path / "shunt.raw"
    p.write_text(raw, encoding="utf-8")
    case = load_raw(p)
    np.testing.assert_allclose(case.bus[0, GS], 1.0)
    np.testing.assert_allclose(case.bus[0, BS], 12.0 + 20.0)


def test_load_raw_three_winding_transformer_adds_star_bus(tmp_path: Path) -> None:
    """A three-winding transformer expands to a star bus plus three branches."""

    raw = dedent(
        """\
        0,   100.00, 33, 0, 1, 60.00     / PSS/E 33 raw
        3-winding xfmr test
        adds a star bus
             1,'HV          ', 500.0,3,    1,    1,    1,1.0000,    0.00, 1.10000, 0.90000, 1.10000, 0.90000
             2,'MV          ', 230.0,1,    1,    1,    1,1.0000,    0.00, 1.10000, 0.90000, 1.10000, 0.90000
             3,'LV          ',  34.5,1,    1,    1,    1,1.0000,    0.00, 1.10000, 0.90000, 1.10000, 0.90000
        0 /End of Bus data, Begin Load data
        0 /End of Load data, Begin Fixed shunt data
        0 /End of Fixed shunt data, Begin Generator data
        0 /End of Generator data, Begin Branch data
        0 /End of Branch data, Begin Transformer data
             1,     2,     3,'1 ',1,1,1,0.00000E+0,0.00000E+0,2,'TG-1         ',1,   1,1.0000
         0.00000E+0, 1.00000E-1,   100.00, 0.00000E+0, 1.50000E-1,   100.00, 0.00000E+0, 2.00000E-1,   100.00,1.00000,    0.00
         1.00000,  500.000,    0.00,   200.00,   200.00,   200.00,0,    0,1.10000,0.90000,1.10000,0.90000,  33,0,0.00000E+0,0.00000E+0,0.00000E+0
         1.00000,  230.000,    0.00,   200.00,   200.00,   200.00,0,    0,1.10000,0.90000,1.10000,0.90000,  33,0,0.00000E+0,0.00000E+0,0.00000E+0
         1.00000,   34.500,    0.00,   200.00,   200.00,   200.00,0,    0,1.10000,0.90000,1.10000,0.90000,  33,0,0.00000E+0,0.00000E+0,0.00000E+0
        0 /End of Transformer data, Begin Area interchange data
        0 /End of Area interchange data, Begin Two-terminal dc line data
        0 /End of Two-terminal dc line data, Begin VSC dc line data
        0 /End of VSC dc line data, Begin Impedance correction data
        0 /End of Impedance correction data, Begin Multi-terminal dc line data
        0 /End of Multi-terminal dc line data, Begin Multi-section line data
        0 /End of Multi-section line data, Begin Zone data
        0 /End of Zone data, Begin Inter-area transfer data
        0 /End of Inter-area transfer data, Begin Owner data
        0 /End of Owner data, Begin FACTS device data
        0 /End of FACTS device data, Begin Switched shunt data
        0 /End of Switched shunt data
        Q
        """
    )
    p = tmp_path / "tw.raw"
    p.write_text(raw, encoding="utf-8")
    case = load_raw(p)
    assert case.bus.shape[0] == 4  # 3 real + 1 star
    assert case.branch.shape[0] == 3

    star_id = int(case.bus[-1, BUS_I])
    assert star_id not in (1, 2, 3)
    star_branches = {int(r[T_BUS]) for r in case.branch}
    assert star_branches == {star_id}
    from_buses = sorted(int(r[F_BUS]) for r in case.branch)
    assert from_buses == [1, 2, 3]


def test_load_raw_isolates_disconnected_bus(tmp_path: Path) -> None:
    """IDE=4 maps to MATPOWER's ISOLATED_BUS so preprocess can drop it."""

    raw = dedent(
        """\
        0,   100.00, 33, 0, 1, 60.00     / PSS/E 33 raw
        isolated bus test
        single record
             1,'B1          ', 230.0,3,    1,    1,    1,1.0000,    0.00, 1.10000, 0.90000, 1.10000, 0.90000
             2,'B2          ', 230.0,4,    1,    1,    1,1.0000,    0.00, 1.10000, 0.90000, 1.10000, 0.90000
        0 /End of Bus data, Begin Load data
        0 /End of Load data, Begin Fixed shunt data
        0 /End of Fixed shunt data, Begin Generator data
        0 /End of Generator data, Begin Branch data
        0 /End of Branch data, Begin Transformer data
        0 /End of Transformer data, Begin Area interchange data
        0 /End of Area interchange data, Begin Two-terminal dc line data
        0 /End of Two-terminal dc line data, Begin VSC dc line data
        0 /End of VSC dc line data, Begin Impedance correction data
        0 /End of Impedance correction data, Begin Multi-terminal dc line data
        0 /End of Multi-terminal dc line data, Begin Multi-section line data
        0 /End of Multi-section line data, Begin Zone data
        0 /End of Zone data, Begin Inter-area transfer data
        0 /End of Inter-area transfer data, Begin Owner data
        0 /End of Owner data, Begin FACTS device data
        0 /End of FACTS device data, Begin Switched shunt data
        0 /End of Switched shunt data
        Q
        """
    )
    p = tmp_path / "iso.raw"
    p.write_text(raw, encoding="utf-8")
    case = load_raw(p)
    assert int(case.bus[case.bus[:, BUS_I] == 2, BUS_TYPE][0]) == ISOLATED_BUS
