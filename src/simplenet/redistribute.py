"""Redistribute loads on the reduced model.

Port of ``matlab/NetworkReduction2/LoadRedistribution.m``. The goal is
to make a DC power flow on the reduced model reproduce the full
model's bus voltage angles for the retained buses, given that
external generators have already been relocated.

Algorithm:

1. If ``pf_flag`` is true, solve a DC PF on the full model first;
   otherwise reuse ``case.bus[:, VA]`` directly.
2. Map every retained bus's ``Vm``/``Va`` from the full solution.
3. Build the reduced model's DC PF matrix ``B_r`` (with tap ratios,
   no shunts) and compute ``P_inj = B_r * theta``.
4. Adjust each phase-shifting transformer's contribution and any
   HVDC line injections.
5. Set ``Pd_new = Pg_total_at_bus - P_inj`` so the reduced DC PF
   matches the full solution exactly.
"""

from __future__ import annotations

import numpy as np

from simplenet.case import (
    GEN_BUS,
    GEN_STATUS,
    PD,
    PG,
    VA,
    VM,
    PowerCase,
)
from simplenet.dcpf import annotate_case_with_solution, run_dcpf
from simplenet.ymatrix import build_b_for_dcpf


def redistribute_loads(
    full_case: PowerCase,
    reduced_case: PowerCase,
    *,
    pf_flag: bool = True,
) -> PowerCase:
    """Return a new reduced :class:`PowerCase` with ``Pd`` rebalanced.

    Parameters
    ----------
    full_case
        The full model (post-preprocess) before reduction.
    reduced_case
        The reduced model after external-generator placement.
    pf_flag
        If ``True`` (default, matching ``reduction_test.m`` with
        ``Pf_flag = 2``... but the algorithm treats anything non-zero
        as "solve DC PF"), run a DC PF on the full model before
        redistribution. If ``False`` use the existing ``Va`` values
        from ``full_case.bus``.
    """

    if pf_flag:
        result = run_dcpf(full_case)
        if not result.converged:
            raise RuntimeError(
                "unable to solve dc powerflow with original full model, "
                "load cannot be redistributed"
            )
        full_case = annotate_case_with_solution(full_case, result)

    reduced = reduced_case.copy()

    full_bus_ids = full_case.bus[:, 0].astype(np.int64)
    full_va = full_case.bus[:, VA]
    full_vm = full_case.bus[:, VM]
    full_lookup = {int(b): i for i, b in enumerate(full_bus_ids)}

    red_bus_ids = reduced.bus[:, 0].astype(np.int64)
    for k, b in enumerate(red_bus_ids):
        if int(b) in full_lookup:
            j = full_lookup[int(b)]
            reduced.bus[k, VM] = full_vm[j]
            reduced.bus[k, VA] = full_va[j]

    theta = np.deg2rad(reduced.bus[:, VA])

    B_red, p_shift_red = build_b_for_dcpf(reduced)
    p_inj = B_red @ theta * reduced.base_mva
    p_inj = p_inj + p_shift_red

    generation_per_bus = np.zeros(reduced.n_bus())
    if reduced.n_gen():
        bus_lookup = {int(b): i for i, b in enumerate(red_bus_ids)}
        gen_full_p = None
        if full_case.n_gen():
            gen_full_p = {int(g[GEN_BUS]): 0.0 for g in full_case.gen}
            for g in full_case.gen:
                if g[GEN_STATUS] == 0:
                    continue
                gen_full_p[int(g[GEN_BUS])] = gen_full_p.get(int(g[GEN_BUS]), 0.0) + float(g[PG])

        for i in range(reduced.n_gen()):
            gen_bus_id = int(reduced.gen[i, GEN_BUS])
            pg_val = float(reduced.gen[i, PG])
            if gen_full_p is not None and gen_bus_id in gen_full_p:
                pg_val = gen_full_p[gen_bus_id]
                reduced.gen[i, PG] = pg_val
            pos = bus_lookup.get(gen_bus_id)
            if pos is None:
                continue
            if reduced.gen[i, GEN_STATUS] == 0:
                continue
            generation_per_bus[pos] += pg_val

    p_load = generation_per_bus - p_inj

    if reduced.dcline is not None and reduced.dcline.shape[0] and full_case.dcline is not None:
        dc = full_case.dcline
        bus_lookup = {int(b): i for i, b in enumerate(red_bus_ids)}
        for row in dc:
            f_id = int(row[0])
            t_id = int(row[1])
            pf_dc = float(row[3])
            pt_dc = float(row[4])
            if f_id in bus_lookup and t_id in bus_lookup:
                p_load[bus_lookup[f_id]] -= pf_dc
                p_load[bus_lookup[t_id]] += pt_dc

    reduced.bus[:, PD] = p_load
    return reduced
