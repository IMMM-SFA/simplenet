"""Standalone DC power flow.

Solves :math:`B' \\theta = P_{\\text{net}}` for non-slack buses where
``B'`` is the bus susceptance matrix from
:func:`simplenet.ymatrix.build_b_for_dcpf` and ``P_net`` aggregates
generator output minus load minus phase-shifter and HVDC injections.
Slack-bus angle is fixed at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from simplenet.case import (
    GEN_BUS,
    PD,
    PG,
    PV_BUS,
    REF_BUS,
    VA,
    PowerCase,
)
from simplenet.ymatrix import build_b_for_dcpf


@dataclass
class DCPFResult:
    """Result of :func:`run_dcpf`. ``theta`` is in **radians**."""

    theta: np.ndarray
    p_gen: np.ndarray
    converged: bool


def _find_slack_index(case: PowerCase) -> int:
    types = case.bus[:, 1].astype(int)
    refs = np.where(types == REF_BUS)[0]
    if refs.size == 0:
        pvs = np.where(types == PV_BUS)[0]
        if pvs.size == 0:
            return 0
        return int(pvs[0])
    return int(refs[0])


def run_dcpf(case: PowerCase) -> DCPFResult:
    """Run a DC power flow on ``case`` and return the bus angles.

    The result also includes a slack-bus rebalance: the slack
    generator's ``Pg`` is overwritten so that total generation equals
    total demand plus phase-shifter and HVDC contributions, matching
    MATPOWER's behavior.

    If no REF bus (``type == 3``) is present, the first PV bus
    (``type == 2``) is auto-promoted to slack. This is needed when the
    full model's slack bus has been eliminated by the reduction
    pipeline.

    Parameters
    ----------
    case
        The input case.

    Returns
    -------
    DCPFResult
        Bus angles (radians), updated generator ``Pg`` column, and a
        convergence flag.
    """

    n = case.n_bus()
    if n == 0:
        return DCPFResult(theta=np.zeros(0), p_gen=np.zeros(0), converged=True)

    B, p_shift = build_b_for_dcpf(case)
    bus_ids = case.bus[:, 0].astype(np.int64)
    bus_id_to_idx = {int(b): i for i, b in enumerate(bus_ids)}

    p_load = case.bus[:, PD] / case.base_mva

    p_gen_per_bus = np.zeros(n)
    if case.n_gen():
        gen_bus = case.gen[:, GEN_BUS].astype(np.int64)
        gen_p = case.gen[:, PG] / case.base_mva
        gen_status = case.gen[:, 7]
        for i, gb in enumerate(gen_bus):
            if gen_status[i] == 0:
                continue
            pos = bus_id_to_idx.get(int(gb))
            if pos is None:
                continue
            p_gen_per_bus[pos] += gen_p[i]

    p_net = p_gen_per_bus - p_load + p_shift / case.base_mva

    slack = _find_slack_index(case)
    keep = np.ones(n, dtype=bool)
    keep[slack] = False
    keep_idx = np.where(keep)[0]

    B_reduced = B[keep_idx, :][:, keep_idx]
    rhs = p_net[keep_idx] - B[keep_idx, :][:, [slack]].toarray().ravel() * 0.0

    theta = np.zeros(n)
    if keep_idx.size:
        try:
            x = spla.spsolve(B_reduced.tocsc(), rhs)
        except RuntimeError:
            return DCPFResult(theta=np.zeros(n), p_gen=np.zeros(case.n_gen()), converged=False)
        theta[keep_idx] = x

    p_gen_new = case.gen[:, PG].copy() if case.n_gen() else np.zeros(0)
    if case.n_gen():
        B_full = B @ theta
        slack_inj = float(B_full[slack] + p_load[slack] - p_shift[slack] / case.base_mva)
        gen_bus = case.gen[:, GEN_BUS].astype(np.int64)
        for i, gb in enumerate(gen_bus):
            if int(gb) == int(bus_ids[slack]) and case.gen[i, 7] != 0:
                p_gen_new[i] = slack_inj * case.base_mva
                break

    return DCPFResult(theta=theta, p_gen=p_gen_new, converged=True)


def annotate_case_with_solution(case: PowerCase, result: DCPFResult) -> PowerCase:
    """Return a copy of ``case`` with ``Va`` and slack ``Pg`` updated.

    The bus ``Vm`` column is set to 1.0 throughout (DC assumption).

    Parameters
    ----------
    case
        Input case to annotate. Not mutated.
    result
        DC power flow result from :func:`run_dcpf`.

    Returns
    -------
    PowerCase
        Copy of ``case`` with ``Va`` (degrees), ``Vm`` (= 1.0), and the
        slack generator's ``Pg`` overwritten from ``result``.
    """

    out = case.copy()
    out.bus[:, VA] = np.rad2deg(result.theta)
    out.bus[:, 7] = 1.0
    if out.n_gen() and result.p_gen.size == out.n_gen():
        out.gen[:, PG] = result.p_gen
    return out
