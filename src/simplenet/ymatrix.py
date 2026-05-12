"""Build sparse DC susceptance matrices used by reduction and DC power flow.

Two flavors:

``build_b_for_reduction``
    Models the full-model bus susceptance matrix used by the Kron
    reduction, mirroring ``Initiation.m`` / ``BuildYMat.m``:

    - branch susceptance ``b = 1/x`` (tap ratio ignored)
    - branch shunts (``mpc.branch[:, 4] / 2``) added to **both diagonal
      entries**
    - bus shunts ``mpc.bus[:, BS] / baseMVA`` added to the diagonal

``build_b_for_dcpf``
    Models the DC power flow B' matrix used by both ``LoadRedistribution.m``
    and the standalone DC PF:

    - branch susceptance ``b = 1 / (x * tap)`` (``tap = 0`` is rewritten
      to ``1`` per MATPOWER convention)
    - **no** branch or bus shunts
    - phase shifters return an injection vector added to the RHS
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from simplenet.case import (
    BR_B,
    BR_X,
    BS,
    F_BUS,
    SHIFT,
    T_BUS,
    TAP,
    PowerCase,
)


def _internal_bus_indices(case: PowerCase) -> tuple[np.ndarray, dict[int, int]]:
    """Return the 0-indexed internal index for every branch endpoint.

    The case's bus rows are assumed already sorted by ``bus_i`` (which
    ``preprocess`` guarantees). Returns ``(f_idx, t_idx)`` arrays of
    branch endpoint internal indices, plus the
    ``original_bus_id -> internal_index`` map.
    """

    bus_ids = case.bus[:, 0].astype(np.int64, copy=False)
    id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}
    return bus_ids, id_to_idx


def _branch_endpoint_indices(case: PowerCase, id_to_idx: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    fbus = case.branch[:, F_BUS].astype(np.int64)
    tbus = case.branch[:, T_BUS].astype(np.int64)
    f_idx = np.fromiter((id_to_idx[int(b)] for b in fbus), dtype=np.int64, count=fbus.size)
    t_idx = np.fromiter((id_to_idx[int(b)] for b in tbus), dtype=np.int64, count=tbus.size)
    return f_idx, t_idx


def build_b_for_reduction(case: PowerCase) -> sp.csc_matrix:
    """Build the symmetric bus susceptance matrix used by the Kron reduction.

    Parameters
    ----------
    case
        The full input case. Buses must already be sorted by bus ID
        (see :func:`simplenet.preprocess.preprocess`).

    Returns
    -------
    sp.csc_matrix
        ``N_bus x N_bus`` sparse CSC matrix. Buses are 0-indexed in
        the order of ``case.bus`` rows.
    """

    n = case.n_bus()
    _, id_to_idx = _internal_bus_indices(case)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    if case.n_branch():
        f_idx, t_idx = _branch_endpoint_indices(case, id_to_idx)
        x = case.branch[:, BR_X]
        with np.errstate(divide="raise", invalid="raise"):
            line_b = 1.0 / x
        shunt_half = case.branch[:, BR_B] / 2.0

        rows.append(f_idx)
        cols.append(t_idx)
        data.append(-line_b)
        rows.append(t_idx)
        cols.append(f_idx)
        data.append(-line_b)

        diag_contrib = line_b + shunt_half
        rows.append(f_idx)
        cols.append(f_idx)
        data.append(diag_contrib)
        rows.append(t_idx)
        cols.append(t_idx)
        data.append(diag_contrib)

    bus_shunt = case.bus[:, BS] / case.base_mva
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    data.append(bus_shunt)

    row_arr = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    col_arr = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    val_arr = np.concatenate(data) if data else np.zeros(0)
    return sp.csc_matrix((val_arr, (row_arr, col_arr)), shape=(n, n))


def build_b_for_dcpf(
    case: PowerCase,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the DC power flow matrix and phase-shifter injection vector.

    Parameters
    ----------
    case
        The input case.

    Returns
    -------
    B : sp.csc_matrix
        ``N_bus x N_bus`` sparse CSC bus susceptance matrix used for
        the DC power flow equation ``B theta = P_net``.
    P_shift : np.ndarray
        Per-bus injection contribution from phase-shifting
        transformers, in per-unit on ``baseMVA``.
    """

    n = case.n_bus()
    _, id_to_idx = _internal_bus_indices(case)

    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    data_l: list[np.ndarray] = []
    p_shift = np.zeros(n)

    if case.n_branch():
        f_idx, t_idx = _branch_endpoint_indices(case, id_to_idx)
        x = case.branch[:, BR_X]
        tap = case.branch[:, TAP].copy()
        tap[tap == 0] = 1.0
        b = 1.0 / (x * tap)

        rows_l.append(f_idx)
        cols_l.append(t_idx)
        data_l.append(-b)
        rows_l.append(t_idx)
        cols_l.append(f_idx)
        data_l.append(-b)

        rows_l.append(f_idx)
        cols_l.append(f_idx)
        data_l.append(b)
        rows_l.append(t_idx)
        cols_l.append(t_idx)
        data_l.append(b)

        shift_rad = case.branch[:, SHIFT] * np.pi / 180.0
        if np.any(shift_rad):
            inj = shift_rad * b
            np.add.at(p_shift, f_idx, -inj)
            np.add.at(p_shift, t_idx, inj)

    if not rows_l:
        return sp.csc_matrix((n, n)), p_shift

    row_arr = np.concatenate(rows_l)
    col_arr = np.concatenate(cols_l)
    val_arr = np.concatenate(data_l)
    return sp.csc_matrix((val_arr, (row_arr, col_arr)), shape=(n, n)), p_shift
