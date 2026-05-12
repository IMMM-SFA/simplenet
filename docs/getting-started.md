# Getting Started

## Installation

```bash
git clone https://github.com/IMMM-SFA/simplenet.git
cd simplenet
pip install -e .
```

The package requires Python 3.11+. Runtime dependencies are `numpy`,
`scipy`, `pandas`, `openpyxl`, and `click`.

To run the test suite, install the dev extras:

```bash
pip install -e ".[dev]"
pytest -v
```

To build this documentation site locally, install the `docs` extras:

```bash
pip install -e ".[docs]"
mkdocs serve  # http://127.0.0.1:8000
```

## The 9-bus worked example

The MATLAB toolbox ships an Example_9bus.m demo that eliminates buses
`[1, 5, 8]` from the standard MATPOWER 9-bus case. `simplenet`
reproduces the same behavior:

```python
from simplenet import reduce_network
from simplenet.io import load_m, dump_xlsx

case = load_m("matlab/NetworkReduction2/test_9bus_case.m")
result = reduce_network(case, excluded_bus_ids=[1, 5, 8])

print(result.summary)
# Reduction process start
# Preprocess data
# ...
# 6 buses in reduced model
# 7 branches in reduced model, including 4 equivalent lines
# 3 generators in reduced model
# External generator on bus 1 is moved to 4

dump_xlsx(result.reduced_case, "case9_reduced.xlsx", summary=result.summary)
```

The reduced model has:

- 6 retained buses: `{2, 3, 4, 6, 7, 9}`
- 4 equivalent branches between bus pairs `(2,7)`, `(2,9)`, `(4,6)`,
  `(7,9)`, each carrying circuit number `99`
- Generator originally on bus 1 moved to bus 4

These expectations are baked into the regression test
[`tests/test_9bus_reduction.py`](https://github.com/IMMM-SFA/simplenet/blob/main/tests/test_9bus_reduction.py).

## Driving the TAMU WECC workflow

The TAMU instructions tell users to:

1. Populate the matrices in `matlab2.xlsx` (Bus / Gen / Branch / GenCost
   / Gentype / Genfuel / Bus Names sheets).
2. Edit `case_ACTIVSg10kCopy2.m` to point at that workbook (which the
   MATLAB code already does via `readmatrix`).
3. Run `reduction_test.m` with a chosen `excluded_nodes_<N>.csv`.

`simplenet` collapses those three steps into one call:

```python
from simplenet.io import load_xlsx, load_excluded_nodes, dump_xlsx
from simplenet import reduce_network

case = load_xlsx("matlab/matlab2_WECC.xlsx")
excluded = load_excluded_nodes("excluded_nodes_550.csv")

result = reduce_network(case, excluded, pf_flag=True)
dump_xlsx(result.reduced_case, "Result_excluded_nodes_550.xlsx",
          summary=result.summary)
```

`load_xlsx` reads the same Bus / Gen / Branch / GenCost / Gentype /
Genfuel / Bus Names sheets that `case_ACTIVSg10kCopy2.m` reads, and is
robust to optional header rows in the numeric sheets.

`load_excluded_nodes` accepts a one-column CSV with or without a
`ExcludedNodes` header row (the format used by the
[expected_output sample](https://github.com/IMMM-SFA/simplenet/blob/main/matlab/expected_output/excluded_nodes.csv)).

## Using a pypower-style dict

If you already have a case represented as a pypower-style dict you can
skip the file I/O:

```python
from simplenet import PowerCase, reduce_network

mpc = {
    "baseMVA": 100.0,
    "bus": [...],     # 13+ column numeric matrix
    "gen": [...],     # 21+ column numeric matrix
    "branch": [...],  # 13+ column numeric matrix
}
case = PowerCase.from_pypower(mpc)
result = reduce_network(case, excluded_bus_ids=[...])
reduced_dict = result.reduced_case.to_pypower()
```

## Inspecting the result

[`ReductionResult`][simplenet.pipeline.ReductionResult] has six fields:

| Field                  | Description                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| `reduced_case`         | The reduced [`PowerCase`][simplenet.case.PowerCase]                                                  |
| `link`                 | `Nx2` array of `[original_bus_id, mapped_bus_id]` for every bus that participated in gen placement   |
| `bcirc`                | Per-branch circuit numbers; equivalent branches use `eq_bcirc_value` (typically `99`)                |
| `boundary_buses`       | Retained-bus IDs adjacent to the eliminated set                                                      |
| `summary`              | Human-readable diary string mirroring MATLAB's `diary` output                                        |
| `preprocess_stats`     | Counts of isolated buses / OOS branches / generators dropped at the preprocessing stage              |

## Solving DC power flow on the reduced model

`simplenet` ships a self-contained DC power flow that you can use to
sanity-check the reduced model:

```python
from simplenet import run_dcpf
import numpy as np

reduced_pf = run_dcpf(result.reduced_case)
print(np.rad2deg(reduced_pf.theta))   # bus angles in degrees
```

For the 9-bus example the reduced-model angles match the full-model
angles for retained buses up to the slack-bus shift (the full model's
slack &mdash; bus 1 &mdash; was eliminated, so the reduced model auto-promotes
the first PV bus to slack). See [Algorithm](algorithm.md) for details.
