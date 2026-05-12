# simplenet

<p align="left">
  <a href="https://github.com/IMMM-SFA/simplenet/actions/workflows/tests.yml">
    <img src="https://github.com/IMMM-SFA/simplenet/actions/workflows/tests.yml/badge.svg" alt="tests"/>
  </a>
  &nbsp;
  <a href="https://github.com/IMMM-SFA/simplenet/actions/workflows/docs.yml">
    <img src="https://github.com/IMMM-SFA/simplenet/actions/workflows/docs.yml/badge.svg" alt="docs"/>
  </a>
</p>



A Python implementation of the [TAMU](https://electricgrids.engr.tamu.edu)
DC modified-Ward network reduction toolbox. **Full documentation lives at
[immm-sfa.github.io/simplenet](https://immm-sfa.github.io/simplenet/)**,
including a [side-by-side MATLAB-to-Python comparison](https://immm-sfa.github.io/simplenet/matlab-comparison/)
and the [API reference](https://immm-sfa.github.io/simplenet/api/).

`simplenet` is a from-scratch port of the MATLAB code in
`[/matlab/NetworkReduction2](matlab/NetworkReduction2)`
(originally by Yujia Zhu, ASU; later adapted by the TAMU group) that:

- Eliminates a user-specified set of "external" buses from a MATPOWER-style
power-system case.
- Produces an equivalent reduced network with new "equivalent" branches
between the boundary buses, a bus-shunt adjustment that absorbs all
branch shunts, and external generators relocated to the nearest
internal bus by shortest electrical distance.
- Redistributes loads so a DC power flow on the reduced model reproduces
the full model's bus angles.

It accepts MATPOWER `.m` files, the `matlab2*.xlsx` sheet workbook the
TAMU workflow uses, PSS/E `.RAW` v33 case files (the upstream format
of the ACTIVSg synthetic grids), or in-memory `pypower`-style
dictionaries, and can be driven from Python or from a CLI.

## Why a Python port?

The original MATLAB code requires a MATPOWER installation and license, has
not been updated in several years, and is awkward to integrate into modern
data-science pipelines. `simplenet` replaces the hand-rolled partial-LU
factorization in `PartialSymLU.m`/`PartialNumLU.m` with the equivalent
[Kron reduction](https://en.wikipedia.org/wiki/Kron_reduction) formula

 Y_{\text{red}} = Y_{ii} - Y_{ie} Y_{ee}^{-1} Y_{ei} 

implemented in scipy.sparse - identical math, but ~30 lines instead of
~1 000 of self-referential link bookkeeping, and roughly an order of
magnitude faster on the 10 000-bus WECC test case.

## Installation

```bash
pip install -e .
```

Requires Python >=3.11. The runtime dependencies are `numpy`, `scipy`,
`pandas`, `openpyxl`, and `click`.

## Quickstart - Python API

```python
from simplenet import reduce_network
from simplenet.io import load_m, load_raw, load_xlsx, load_excluded_nodes, dump_xlsx

# Option A: load a MATPOWER .m case file directly
case = load_m("matlab/NetworkReduction2/test_9bus_case.m")

# Option B: load the TAMU matlab2*.xlsx workbook
case = load_xlsx("matlab/matlab2_WECC.xlsx")

# Option C: load a PSS/E v33 .RAW (e.g. ACTIVSg10k.RAW)
case = load_raw("ACTIVSg10k.RAW")

excluded = load_excluded_nodes("matlab/expected_output/excluded_nodes.csv")

result = reduce_network(case, excluded, pf_flag=True)

print(result.summary)
dump_xlsx(result.reduced_case, "result.xlsx", summary=result.summary)
```

Result fields:


| Field                     | Description                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------------- |
| `result.reduced_case`     | `PowerCase` of the reduced network                                                              |
| `result.link`             | `Nx2` array of `[original_bus_id, mapped_bus_id]`                                               |
| `result.bcirc`            | per-branch circuit numbers (equivalent branches are tagged with `eq_bcirc_value`, typically 99) |
| `result.boundary_buses`   | bus IDs adjacent to the eliminated set                                                          |
| `result.summary`          | human-readable diary string (mirrors MATLAB's `diary` output)                                   |
| `result.preprocess_stats` | counts of isolated buses / OOS branches dropped                                                 |


## Quickstart - CLI

```bash
simplenet info matlab/matlab2_WECC.xlsx
simplenet reduce matlab/matlab2_WECC.xlsx \
    matlab/expected_output/excluded_nodes.csv \
    -o reduced.xlsx \
    --summary-txt reduced.txt
```

Without MATLAB this reproduces the workflow that `reduction_test.m`
exercises, writing the same multi-sheet output (`Summary`, `Gen`,
`Bus`, `Branch`, ...).

## API surface

```python
from simplenet import (
    PowerCase,           # core data class
    reduce_network,      # full pipeline (MPReduction.m)
    preprocess,          # PreProcessData.m
    kron_reduce,         # core Kron math (Partial{Sym,Num}LU.m)
    move_external_generators,  # MoveExGen.m
    redistribute_loads,  # LoadRedistribution.m
    run_dcpf,            # standalone DC power flow
    build_b_for_reduction,
    build_b_for_dcpf,
)
from simplenet.io import load_m, load_xlsx, load_excluded_nodes, dump_xlsx
```

`PowerCase` adapters: `PowerCase.from_pypower(d)` and `case.to_pypower()`
for interop with `pypower`-style dicts.

## Method - pipeline at a glance

1. **Preprocess.** Drop isolated buses (type 4), out-of-service
  branches (column 11 == 0), branches touching isolated buses, and
   generators on isolated buses. Update the external bus list.
2. **Build B-matrix.** Sparse DC susceptance matrix using
  `b = 1/x` per branch plus per-bus shunts and half-branch shunts on
   each endpoint.
3. **Boundary detection.** Retained buses adjacent to the external
  set.
4. **First Kron reduction.** Eliminate every external bus. Off-diagonal
  fills in the boundary-bus block become equivalent branches; the
   diagonal delta is absorbed into the bus shunts.
5. **Second Kron reduction.** Eliminate external buses that do not host
  a generator. Used solely to build the shortest-electrical-distance
   graph for generator placement.
6. **Generator relocation.** Multi-source Dijkstra
  (`scipy.sparse.csgraph.dijkstra`) over the second-reduction graph
   (parallel lines combined by inverse-sum) routes every external
   generator to its nearest retained bus.
7. **Load redistribution.** Optionally solve a DC PF on the full model,
  then set `Pd_new = P_gen - B_reduced * theta` so the reduced DC PF
   exactly reproduces the retained-bus angles. Phase shifters and HVDC
   line injections are corrected separately.
8. **Prune.** Equivalent branches with `|x| >= 10 * max(|x|_orig)` are
  dropped, matching `MPReduction.m`.

## Validation

The 9-bus regression test (`tests/test_9bus_reduction.py`) replicates the
worked example from
`[Example_9bus.m](matlab/NetworkReduction2/Example_9bus.m)`:

- 6 retained buses out of 9
- Four equivalent branches between bus pairs `(2, 7)`, `(2, 9)`,
`(4, 6)`, `(7, 9)`, all with circuit number `99`
- Generator on bus 1 moved to bus 4
- All branch shunts (column 5) zeroed in the reduced model
- DC power flow on the reduced model reproduces the full-model bus
angles up to a slack-bus shift

```bash
pytest -v
```

A smoke test on the 10 000-bus WECC case (`matlab/matlab2_WECC.xlsx`)
with 200 randomly chosen external buses completes in about one second
and emits a numerically clean reduced model.

## Citation

If you use `simplenet` in published work, please also cite the original
MATLAB toolbox and the underlying synthetic test systems:

- Y. Zhu and D. Tylavsky, "An Optimization-Based DC-Network Reduction
Method," *IEEE Trans. Power Systems*, 2018.
- A. B. Birchfield, T. Xu, K. M. Gegner, K. S. Shetye, T. J. Overbye,
"Grid Structural Characteristics as Validation Criteria for Synthetic
Networks," *IEEE Trans. Power Systems*, 2017.

## License

BSD-3-Clause.