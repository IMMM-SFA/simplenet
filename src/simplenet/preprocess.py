"""Preprocess raw input case: drop isolated buses / out-of-service branches.

Port of ``matlab/NetworkReduction2/PreProcessData.m``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplenet.case import (
    BR_STATUS,
    BUS_I,
    BUS_TYPE,
    F_BUS,
    GEN_BUS,
    ISOLATED_BUS,
    T_BUS,
    PowerCase,
)


@dataclass
class PreprocessStats:
    """Counts of items removed during preprocessing (matches MATLAB log)."""

    isolated_buses: int = 0
    branches_removed: int = 0
    generators_removed: int = 0
    dclines_removed: int = 0


def preprocess(case: PowerCase, excluded: np.ndarray) -> tuple[PowerCase, np.ndarray, PreprocessStats]:
    """Drop isolated buses, oos branches, and sync the external bus list.

    Parameters
    ----------
    case
        The full input :class:`PowerCase`. Not mutated.
    excluded
        1-D array of bus IDs (original numbering) the user wants to
        eliminate. The result drops any IDs that have already been
        removed as isolated.

    Returns
    -------
    case : PowerCase
        New :class:`PowerCase` with isolated buses, out-of-service
        branches, branches touching isolated buses, generators on
        isolated buses, and HVDC lines touching isolated buses
        removed.
    excluded : np.ndarray
        Pruned external-bus array (still in original bus numbering).
    stats : PreprocessStats
        Counts of items removed during preprocessing.
    """

    case = case.copy()
    case.bus = case.bus[np.argsort(case.bus[:, BUS_I], kind="stable")]
    case.branch = case.branch[np.lexsort((case.branch[:, T_BUS], case.branch[:, F_BUS]))]

    stats = PreprocessStats()

    branches_before = case.branch.shape[0]
    in_service = case.branch[:, BR_STATUS] != 0
    case.branch = case.branch[in_service]

    isolated_mask = case.bus[:, BUS_TYPE] == ISOLATED_BUS
    isolated_buses = case.bus[isolated_mask, BUS_I]
    stats.isolated_buses = int(isolated_buses.size)

    isolated_set = set(isolated_buses.tolist())

    def in_isolated(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros(0, dtype=bool)
        return np.isin(arr, isolated_buses)

    branch_touches = in_isolated(case.branch[:, F_BUS]) | in_isolated(case.branch[:, T_BUS])
    case.branch = case.branch[~branch_touches]
    stats.branches_removed = int(branches_before - case.branch.shape[0])

    case.bus = case.bus[~isolated_mask]

    if case.gen.shape[0]:
        gen_mask = in_isolated(case.gen[:, GEN_BUS])
        stats.generators_removed = int(np.sum(gen_mask))
        case.gen = case.gen[~gen_mask]
        if case.gencost is not None:
            case.gencost = case.gencost[~gen_mask]
        if case.gentype is not None:
            case.gentype = [t for t, m in zip(case.gentype, gen_mask, strict=False) if not m]
        if case.genfuel is not None:
            case.genfuel = [t for t, m in zip(case.genfuel, gen_mask, strict=False) if not m]

    excluded = np.asarray(excluded, dtype=float).ravel()
    if isolated_set:
        excluded = excluded[~np.isin(excluded, list(isolated_set))]

    if case.dcline is not None and case.dcline.shape[0]:
        dc_mask = in_isolated(case.dcline[:, 0]) | in_isolated(case.dcline[:, 1])
        stats.dclines_removed = int(np.sum(dc_mask))
        case.dcline = case.dcline[~dc_mask]

    return case, excluded, stats
