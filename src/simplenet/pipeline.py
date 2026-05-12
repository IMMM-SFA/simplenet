"""End-to-end network reduction pipeline.

The public entry point is :func:`reduce_network`, the Python analogue
of ``matlab/NetworkReduction2/MPReduction.m``. It glues the preprocess,
Y-matrix, Kron, boundary, generator-move, and load-redistribution
steps together, then prunes equivalent branches whose reactance is at
least ten times the maximum original branch reactance (see
``MPReduction.m`` lines 89-91).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO

import numpy as np

from simplenet.assemble import (
    AssembleResult,
    assemble_reduced,
    generate_bcirc,
)
from simplenet.boundary import find_boundary_buses
from simplenet.case import BR_X, GEN_BUS, PowerCase
from simplenet.generators import GenMoveResult, move_external_generators
from simplenet.kron import kron_reduce
from simplenet.preprocess import PreprocessStats, preprocess
from simplenet.redistribute import redistribute_loads
from simplenet.ymatrix import build_b_for_reduction


@dataclass
class ReductionResult:
    """Output of :func:`reduce_network`."""

    reduced_case: PowerCase
    link: np.ndarray
    bcirc: np.ndarray
    boundary_buses: np.ndarray
    eq_bcirc_value: int
    preprocess_stats: PreprocessStats
    summary: str
    log: list[str] = field(default_factory=list)


def _emit(log: list[str], buf: StringIO, msg: str) -> None:
    log.append(msg)
    buf.write(msg + "\n")


def _kron_split(case: PowerCase, external_bus_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(internal_idx, external_idx)`` arrays for the Y matrix."""

    bus_ids = case.bus[:, 0].astype(np.int64, copy=False)
    ext_set = {int(b) for b in external_bus_ids}
    ext_idx = np.array([i for i, b in enumerate(bus_ids) if int(b) in ext_set], dtype=np.int64)
    int_idx = np.array([i for i, b in enumerate(bus_ids) if int(b) not in ext_set], dtype=np.int64)
    return int_idx, ext_idx


def _do_reduction(
    case: PowerCase,
    external_bus_ids: np.ndarray,
    bcirc: np.ndarray,
) -> AssembleResult:
    """Single Kron reduction step + reduced-case assembly."""

    boundary = find_boundary_buses(case, external_bus_ids)
    y = build_b_for_reduction(case)
    int_idx, ext_idx = _kron_split(case, external_bus_ids)
    k_result = kron_reduce(y, ext_idx, int_idx)
    return assemble_reduced(case, external_bus_ids, boundary, k_result, bcirc)


def reduce_network(
    case: PowerCase,
    excluded_bus_ids: np.ndarray | list[int],
    *,
    pf_flag: bool = True,
) -> ReductionResult:
    """Run the full network reduction pipeline.

    Parameters
    ----------
    case
        The full :class:`PowerCase` model.
    excluded_bus_ids
        Original bus IDs of the buses to be eliminated ("external").
    pf_flag
        If ``True`` (matching MATPOWER's ``Pf_flag = 1``), a DC power
        flow is solved on the full model before load redistribution.

    Returns
    -------
    ReductionResult
        Contains the reduced :class:`PowerCase`, the
        per-branch circuit-number vector, the generator ``Link``
        mapping, and a human-readable summary mirroring the diary
        output from ``reduction_test.m``.
    """

    log: list[str] = []
    summary = StringIO()
    _emit(log, summary, "Reduction process start")
    _emit(log, summary, "Preprocess data")

    excluded = np.asarray(list(excluded_bus_ids), dtype=float).ravel()
    case, excluded, prep_stats = preprocess(case, excluded)
    _emit(log, summary, f"Eliminate {prep_stats.isolated_buses} isolated buses")
    _emit(log, summary, f"Eliminate {prep_stats.branches_removed} branches")
    _emit(log, summary, f"Eliminate {prep_stats.generators_removed} generators")
    if prep_stats.dclines_removed:
        _emit(log, summary, f"Eliminate {prep_stats.dclines_removed} dc lines")
    _emit(log, summary, "Preprocessing complete")

    if case.dcline is not None and case.dcline.shape[0]:
        ext_set = set(int(b) for b in excluded)
        if np.any(np.isin(case.dcline[:, 0].astype(np.int64), list(ext_set))) or np.any(
            np.isin(case.dcline[:, 1].astype(np.int64), list(ext_set))
        ):
            raise ValueError("not able to eliminate HVDC line terminals")

    bcirc_full = generate_bcirc(case.branch)
    max_x_orig = float(np.max(np.abs(case.branch[:, BR_X]))) if case.n_branch() else 0.0

    bus_id_arr = case.bus[:, 0].astype(np.int64)
    ext_set = {int(b) for b in excluded}
    internal_bus_ids = bus_id_arr[~np.isin(bus_id_arr, list(ext_set))]

    if excluded.size == 0:
        _emit(log, summary, "No external buses, reduced model is same as full model")
        result = ReductionResult(
            reduced_case=case,
            link=np.column_stack([bus_id_arr, bus_id_arr]),
            bcirc=bcirc_full,
            boundary_buses=np.zeros(0, dtype=np.int64),
            eq_bcirc_value=99,
            preprocess_stats=prep_stats,
            summary=summary.getvalue(),
            log=log,
        )
        return result

    _emit(log, summary, "Convert input data model")
    _emit(log, summary, "Creating Y matrix of input full model")
    _emit(log, summary, "Do first round reduction eliminating all external buses")

    boundary_ids = find_boundary_buses(case, excluded)
    first = _do_reduction(case, excluded, bcirc_full)

    if case.n_gen():
        ex_with_gen = np.intersect1d(case.gen[:, GEN_BUS].astype(np.int64), excluded.astype(np.int64))
    else:
        ex_with_gen = np.zeros(0, dtype=np.int64)
    _emit(log, summary, f"{ex_with_gen.size} external generators are to be placed")

    if case.n_gen():
        external_non_gen = np.setdiff1d(excluded.astype(np.int64), case.gen[:, GEN_BUS].astype(np.int64))
    else:
        external_non_gen = excluded.astype(np.int64)

    if external_non_gen.size:
        _emit(log, summary, "Do second round reduction eliminating all external non-generator buses")
        second = _do_reduction(case, external_non_gen, bcirc_full)
        case_with_gens = second.reduced_case
    else:
        case_with_gens = case.copy()

    _emit(log, summary, "Placing External generators")
    move_result: GenMoveResult = move_external_generators(case_with_gens, internal_bus_ids, ac_flag=False)

    reduced = first.reduced_case
    bcirc_reduced = first.bcirc

    if reduced.n_gen() and move_result.new_gen_bus.size == reduced.n_gen():
        reduced.gen[:, GEN_BUS] = move_result.new_gen_bus

    _emit(log, summary, "Redistribute loads")
    reduced = redistribute_loads(case, reduced, pf_flag=pf_flag)

    if reduced.n_branch() and max_x_orig > 0:
        threshold = 10.0 * max_x_orig
        mask = np.abs(reduced.branch[:, BR_X]) >= threshold
        if np.any(mask):
            reduced.branch = reduced.branch[~mask]
            bcirc_reduced = bcirc_reduced[~mask]

    eq_count = int(np.sum(bcirc_reduced == first.eq_bcirc_value)) if bcirc_reduced.size else 0
    _emit(log, summary, "**********Reduction Summary****************")
    _emit(log, summary, f"{reduced.n_bus()} buses in reduced model")
    _emit(log, summary, f"{reduced.n_branch()} branches in reduced model, including {eq_count} equivalent lines")
    _emit(log, summary, f"{reduced.n_gen()} generators in reduced model")
    if reduced.dcline is not None and reduced.dcline.shape[0]:
        _emit(log, summary, f"{reduced.dcline.shape[0]} HVDC lines in reduced model")

    _emit(log, summary, "**********Generator Placement Results**************")
    for row in move_result.link:
        if row[0] != row[1] and row[1] != -1:
            _emit(log, summary, f"External generator on bus {int(row[0])} is moved to {int(row[1])}")

    return ReductionResult(
        reduced_case=reduced,
        link=move_result.link,
        bcirc=bcirc_reduced,
        boundary_buses=np.asarray(boundary_ids, dtype=np.int64),
        eq_bcirc_value=first.eq_bcirc_value,
        preprocess_stats=prep_stats,
        summary=summary.getvalue(),
        log=log,
    )
