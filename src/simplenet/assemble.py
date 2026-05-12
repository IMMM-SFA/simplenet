"""Assemble the reduced :class:`PowerCase` from a Kron reduction result.

Port of ``matlab/NetworkReduction2/MakeMPCr.m`` and
``matlab/NetworkReduction2/GenerateBCIRC.m``. Produces the reduced
bus / branch / gen arrays plus the per-branch circuit number vector
that ``MPReduction.m`` exposes externally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simplenet.case import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_STATUS,
    BR_X,
    BRANCH_COLUMNS,
    BS,
    F_BUS,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    T_BUS,
    TAP,
    PowerCase,
    pad_to_columns,
)
from simplenet.kron import KronResult


def generate_bcirc(branch: np.ndarray) -> np.ndarray:
    """Generate per-branch circuit numbers (port of GenerateBCIRC.m).

    Branches are not reordered. For each unique (fbus, tbus) pair the
    first occurrence gets ``BCIRC = 1`` and any subsequent occurrence
    is numbered ``2, 3, ...`` to flag parallel lines.

    Parameters
    ----------
    branch
        MATPOWER-format branch matrix.

    Returns
    -------
    np.ndarray
        1-D ``int64`` array of length ``branch.shape[0]`` with per-row
        circuit numbers.
    """

    n = branch.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    keys = list(zip(branch[:, F_BUS].astype(np.int64), branch[:, T_BUS].astype(np.int64), strict=False))
    counts: dict[tuple[int, int], int] = {}
    out = np.zeros(n, dtype=np.int64)
    for i, k in enumerate(keys):
        counts[k] = counts.get(k, 0) + 1
        out[i] = counts[k]
    return out


def _equivalent_bcirc(orig_max: int) -> int:
    """Pick the equivalent-branch circuit tag per MakeMPCr.m line 81.

    ``EqBCIRC = max(99, 10^ceil(log10(max(BCIRC)-1)) - 1)``.
    """

    if orig_max <= 1:
        return 99
    log_arg = orig_max - 1
    if log_arg < 1:
        return 99
    return max(99, int(10 ** np.ceil(np.log10(log_arg))) - 1)


@dataclass
class AssembleResult:
    """Output of :func:`assemble_reduced`.

    Attributes
    ----------
    reduced_case
        The reduced :class:`~simplenet.case.PowerCase`.
    bcirc
        Per-branch circuit numbers for the reduced model. Equivalent
        branches use ``eq_bcirc_value`` (typically ``99``).
    eq_bcirc_value
        Sentinel circuit number applied to every equivalent branch.
    """

    reduced_case: PowerCase
    bcirc: np.ndarray
    eq_bcirc_value: int


def assemble_reduced(
    case: PowerCase,
    external_bus_ids: np.ndarray,
    boundary_bus_ids: np.ndarray,
    kron_result: KronResult,
    bcirc: np.ndarray,
    *,
    tol: float = 1e-12,
) -> AssembleResult:
    """Build the reduced :class:`PowerCase` from the Kron result.

    Steps (matching MakeMPCr.m):

    1. Drop bus rows for external buses and branches touching them.
    2. Add equivalent branches between boundary buses where the
       difference between the reduced and original internal Y blocks
       is non-trivial.
    3. Set ``Bs`` on every retained bus to ``(diag(Y_red) -
       sum_of_branch_susceptances_at_bus) * baseMVA``.
    4. Zero out branch shunts (column ``b``) - all shunts now live on
       the buses.

    Parameters
    ----------
    case
        Full input :class:`PowerCase`.
    external_bus_ids
        Original bus IDs to be eliminated.
    boundary_bus_ids
        Retained bus IDs that share at least one branch with an
        external bus (see :func:`simplenet.boundary.find_boundary_buses`).
    kron_result
        Result of :func:`simplenet.kron.kron_reduce` on the full Y
        matrix with the same partition.
    bcirc
        Per-branch circuit numbers for ``case.branch`` (typically
        from :func:`generate_bcirc`).
    tol
        Threshold for treating ``y_red[i, j] - y_ii_orig[i, j]`` as a
        new equivalent branch. Entries below this are dropped.

    Returns
    -------
    AssembleResult
        Reduced :class:`PowerCase`, the updated branch-circuit vector
        (with the equivalent-branch sentinel appended), and the
        sentinel value used.
    """

    case = case.copy()
    bus_ids = case.bus[:, 0].astype(np.int64, copy=False)
    ext_set: set[int] = {int(b) for b in np.asarray(external_bus_ids).ravel()}
    boundary_set: set[int] = {int(b) for b in np.asarray(boundary_bus_ids).ravel()}

    int_mask = np.array([int(b) not in ext_set for b in bus_ids])
    int_idx = np.where(int_mask)[0]
    int_bus_ids = bus_ids[int_idx]
    y_pos_for_bus: dict[int, int] = {int(b): k for k, b in enumerate(int_bus_ids)}

    if case.n_branch():
        fbus_int = case.branch[:, F_BUS].astype(np.int64)
        tbus_int = case.branch[:, T_BUS].astype(np.int64)
        branch_keep = ~(np.isin(fbus_int, list(ext_set)) | np.isin(tbus_int, list(ext_set)))
        branch_retained = case.branch[branch_keep]
        bcirc_retained = np.asarray(bcirc, dtype=np.int64)[branch_keep]
    else:
        branch_retained = np.zeros((0, max(case.branch.shape[1], BRANCH_COLUMNS)))
        bcirc_retained = np.zeros(0, dtype=np.int64)

    y_red = kron_result.y_red
    y_ii_orig = kron_result.y_ii_orig
    diff = y_red - y_ii_orig

    bound_pos = np.array(
        sorted(y_pos_for_bus[b] for b in boundary_set if b in y_pos_for_bus),
        dtype=np.int64,
    )

    eq_from: list[int] = []
    eq_to: list[int] = []
    eq_x: list[float] = []
    if bound_pos.size >= 2:
        sub = diff[np.ix_(bound_pos, bound_pos)]
        iu, ju = np.triu_indices(bound_pos.size, k=1)
        vals = sub[iu, ju]
        mask = np.abs(vals) > tol
        for k_idx in np.where(mask)[0]:
            i_pos = int(bound_pos[iu[k_idx]])
            j_pos = int(bound_pos[ju[k_idx]])
            d = float(vals[k_idx])
            eq_from.append(int(int_bus_ids[i_pos]))
            eq_to.append(int(int_bus_ids[j_pos]))
            eq_x.append(-1.0 / d)

    branch_retained_padded = pad_to_columns(branch_retained, BRANCH_COLUMNS)

    orig_max_bcirc = int(bcirc.max()) if bcirc.size else 1
    eq_bcirc_value = _equivalent_bcirc(orig_max_bcirc)

    if eq_from:
        ncols = branch_retained_padded.shape[1] if branch_retained_padded.shape[1] else BRANCH_COLUMNS
        eq_branches = np.zeros((len(eq_from), ncols))
        eq_branches[:, F_BUS] = eq_from
        eq_branches[:, T_BUS] = eq_to
        eq_branches[:, BR_X] = eq_x
        eq_branches[:, RATE_A] = 99999.0
        eq_branches[:, RATE_B] = 99999.0
        eq_branches[:, RATE_C] = 99999.0
        eq_branches[:, TAP] = 1.0
        eq_branches[:, SHIFT] = 0.0
        eq_branches[:, BR_STATUS] = 1.0
        eq_branches[:, ANGMIN] = -360.0
        eq_branches[:, ANGMAX] = 360.0
        new_branch = np.vstack([branch_retained_padded, eq_branches])
        new_bcirc = np.concatenate([bcirc_retained, np.full(len(eq_from), eq_bcirc_value, dtype=np.int64)])
    else:
        new_branch = branch_retained_padded
        new_bcirc = bcirc_retained.copy()

    new_branch[:, BR_B] = 0.0

    new_bus = case.bus[int_idx].copy()

    bus_shunt = np.diag(y_red).copy()
    if new_branch.shape[0]:
        f_arr = new_branch[:, F_BUS].astype(np.int64)
        t_arr = new_branch[:, T_BUS].astype(np.int64)
        x_arr = new_branch[:, BR_X]
        susc = 1.0 / x_arr
        for i in range(new_branch.shape[0]):
            f_pos = y_pos_for_bus[int(f_arr[i])]
            t_pos = y_pos_for_bus[int(t_arr[i])]
            bus_shunt[f_pos] -= susc[i]
            bus_shunt[t_pos] -= susc[i]
    bus_shunt *= case.base_mva

    bus_pos_for_id = {int(b): i for i, b in enumerate(new_bus[:, 0].astype(np.int64))}
    bus_shunt_aligned = np.zeros(new_bus.shape[0])
    for k, bus_id in enumerate(int_bus_ids):
        bus_shunt_aligned[bus_pos_for_id[int(bus_id)]] = bus_shunt[k]
    new_bus[:, BS] = bus_shunt_aligned

    case.bus = new_bus
    case.branch = new_branch

    return AssembleResult(reduced_case=case, bcirc=new_bcirc, eq_bcirc_value=eq_bcirc_value)
