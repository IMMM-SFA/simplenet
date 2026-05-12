"""Boundary bus identification.

Port of ``matlab/NetworkReduction2/DefBoundary.m``: a *boundary* bus is
a retained (internal) bus that shares a branch with at least one
external bus.
"""

from __future__ import annotations

import numpy as np

from simplenet.case import F_BUS, T_BUS, PowerCase


def find_boundary_buses(case: PowerCase, external_bus_ids: np.ndarray) -> np.ndarray:
    """Return the original-numbered IDs of the boundary buses.

    Parameters
    ----------
    case
        The full :class:`PowerCase` (post-preprocess).
    external_bus_ids
        1-D array of bus IDs (in *original* numbering) that will be
        eliminated.
    """

    if case.n_branch() == 0 or external_bus_ids.size == 0:
        return np.zeros(0, dtype=case.bus[:, 0].dtype if case.n_bus() else np.int64)

    ext_set = np.asarray(external_bus_ids).astype(case.bus[:, 0].dtype, copy=False)
    fbus = case.branch[:, F_BUS]
    tbus = case.branch[:, T_BUS]

    f_ext = np.isin(fbus, ext_set)
    t_ext = np.isin(tbus, ext_set)

    crossing = f_ext ^ t_ext
    if not np.any(crossing):
        return np.zeros(0, dtype=ext_set.dtype)

    internal_endpoints = np.where(
        crossing,
        np.where(f_ext, tbus, fbus),
        np.nan,
    )
    internal_endpoints = internal_endpoints[~np.isnan(internal_endpoints)]
    return np.unique(internal_endpoints).astype(ext_set.dtype, copy=False)
