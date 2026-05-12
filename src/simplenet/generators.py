"""Move external generators to their nearest internal bus.

Port of ``matlab/NetworkReduction2/MoveExGen.m``. The MATLAB code builds
a sparse network from the "second" reduction (where only the
non-generator externals are eliminated) and runs a layered shortest
electrical-distance search to attach every external generator to its
closest retained bus. We achieve the same result by collapsing parallel
lines and running multi-source Dijkstra (``scipy.sparse.csgraph``).

Inputs to :func:`move_external_generators`:

``case_with_gens``
    The :class:`PowerCase` that resulted from the *second* reduction -
    its bus set is ``internal U external-with-generator``.
``internal_bus_ids``
    Bus IDs (original numbering) that are *truly* internal (retained)
    and serve as the multi-source set.
``ac_flag``
    If ``True`` use ``|z| = sqrt(r^2 + x^2)`` as the edge weight, else
    ``|x|``. The MATLAB ``reduction_test.m`` always passes ``0``, but
    we expose it for flexibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from simplenet.case import BR_R, BR_X, F_BUS, T_BUS, PowerCase


@dataclass
class GenMoveResult:
    """Output of :func:`move_external_generators`.

    Attributes
    ----------
    new_gen_bus
        Array of length ``case.n_gen()`` with the post-move bus id for
        every generator (in the case's original bus numbering).
    link
        2-D array of shape ``(case.n_bus(), 2)`` whose first column is
        the original bus id and whose second column is the bus id that
        bus is "mapped to" (self for internal buses; nearest internal
        for external-with-generator buses).
    islanded
        Bus IDs that could not reach any internal bus; their generators
        are dropped from ``new_gen_bus`` (mapped to ``None`` via a
        sentinel of ``-1``).
    """

    new_gen_bus: np.ndarray
    link: np.ndarray
    islanded: np.ndarray


def _collapse_parallel_lines(branch: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse parallel branches into a single equivalent weight.

    For DC (no resistance) the parallel combination of impedances
    ``x_1, x_2`` is ``1 / (1/x_1 + 1/x_2)``. We apply the same formula
    to the weight directly because for DC ``weight = |x|`` (and
    ``ac_flag=0`` is the only supported call site).
    """

    if branch.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
    keys = np.array(
        sorted({
            (int(min(f, t)), int(max(f, t)))
            for f, t in zip(
                branch[:, F_BUS].astype(np.int64),
                branch[:, T_BUS].astype(np.int64),
                strict=False,
            )
        }),
        dtype=np.int64,
    )
    key_to_idx = {tuple(k.tolist()): i for i, k in enumerate(keys)}
    parallel: dict[int, list[float]] = {i: [] for i in range(keys.shape[0])}
    for i in range(branch.shape[0]):
        a = int(branch[i, F_BUS])
        b = int(branch[i, T_BUS])
        k = (min(a, b), max(a, b))
        parallel[key_to_idx[k]].append(weight[i])
    out_w = np.zeros(keys.shape[0])
    for i, ws in parallel.items():
        if len(ws) == 1:
            out_w[i] = abs(ws[0])
            continue
        inv_sum = 0.0
        for w in ws:
            if abs(w) > 0:
                inv_sum += 1.0 / abs(w)
        out_w[i] = 1.0 / inv_sum if inv_sum > 0 else float("inf")
    return keys, out_w


def move_external_generators(
    case_with_gens: PowerCase,
    internal_bus_ids: np.ndarray,
    *,
    ac_flag: bool = False,
) -> GenMoveResult:
    """Find the closest internal bus for every external generator.

    Parameters
    ----------
    case_with_gens
        The :class:`PowerCase` that resulted from the *second* Kron
        reduction. Its bus set must contain every internal bus plus
        every external bus that hosts a generator.
    internal_bus_ids
        Original-numbered bus IDs that should be treated as
        retained (multi-source set for the Dijkstra search).
    ac_flag
        When ``True`` use ``sqrt(r^2 + x^2)`` as the edge weight.
        When ``False`` (default, matching ``reduction_test.m``)
        use ``|x|``.

    Returns
    -------
    GenMoveResult
        ``new_gen_bus`` field is suitable for use as the first column
        of the reduced case's ``gen`` matrix.
    """

    bus_ids = case_with_gens.bus[:, 0].astype(np.int64, copy=False)
    n = bus_ids.size
    id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}

    internal_bus_ids = np.asarray(internal_bus_ids).astype(np.int64, copy=False).ravel()
    internal_idx = np.array(
        [id_to_idx[int(b)] for b in internal_bus_ids if int(b) in id_to_idx],
        dtype=np.int64,
    )

    if case_with_gens.n_branch():
        if ac_flag:
            r = case_with_gens.branch[:, BR_R]
            x = case_with_gens.branch[:, BR_X]
            weight = np.sqrt(r * r + x * x)
        else:
            weight = np.abs(case_with_gens.branch[:, BR_X])
        keys, w = _collapse_parallel_lines(case_with_gens.branch, weight)
    else:
        keys = np.zeros((0, 2), dtype=np.int64)
        w = np.zeros(0)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (a_id, b_id), wt in zip(keys, w, strict=False):
        if not np.isfinite(wt):
            continue
        if int(a_id) not in id_to_idx or int(b_id) not in id_to_idx:
            continue
        ai = id_to_idx[int(a_id)]
        bi = id_to_idx[int(b_id)]
        if ai == bi:
            continue
        rows.append(ai)
        cols.append(bi)
        data.append(float(wt))
        rows.append(bi)
        cols.append(ai)
        data.append(float(wt))

    graph = sp.csr_matrix((data, (rows, cols)), shape=(n, n)) if rows else sp.csr_matrix((n, n))

    if internal_idx.size and graph.nnz:
        dist, _pred, sources = csg.dijkstra(
            graph,
            indices=internal_idx,
            return_predecessors=True,
            directed=False,
            min_only=True,
        )
    elif internal_idx.size:
        dist = np.full(n, np.inf)
        dist[internal_idx] = 0.0
        sources = np.full(n, -9999, dtype=np.int64)
        sources[internal_idx] = internal_idx
    else:
        dist = np.full(n, np.inf)
        sources = np.full(n, -9999, dtype=np.int64)

    link = np.empty((n, 2), dtype=np.int64)
    link[:, 0] = bus_ids
    islanded: list[int] = []
    for i in range(n):
        if np.isinf(dist[i]):
            link[i, 1] = -1
            islanded.append(int(bus_ids[i]))
        else:
            link[i, 1] = int(bus_ids[int(sources[i])])

    if case_with_gens.n_gen():
        gen_bus = case_with_gens.gen[:, 0].astype(np.int64, copy=False)
        bus_pos = {int(b): i for i, b in enumerate(bus_ids)}
        new_gen_bus = np.empty(gen_bus.size, dtype=np.int64)
        for i, gb in enumerate(gen_bus):
            pos = bus_pos.get(int(gb))
            new_gen_bus[i] = -1 if pos is None else int(link[pos, 1])
    else:
        new_gen_bus = np.zeros(0, dtype=np.int64)

    return GenMoveResult(new_gen_bus=new_gen_bus, link=link, islanded=np.asarray(islanded, dtype=np.int64))
