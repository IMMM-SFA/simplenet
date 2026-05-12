# MATLAB Comparison

This page maps the original MATLAB toolbox in
[`/matlab/NetworkReduction2`](https://github.com/IMMM-SFA/simplenet/tree/main/matlab/NetworkReduction2)
to the corresponding `simplenet` modules and documents:

1. What was ported faithfully.
2. What was reimplemented differently.
3. **Why** each design decision was made.
4. **Any expected behavioral differences** the user should be aware of.

The original code was written by Yujia Zhu (ASU, Oct. 2014) and adapted
by the TAMU group for the ACTIVSg10k (WECC) and ACTIVSg70k (East)
synthetic systems. The MATLAB sources are bundled with this repo under
`/matlab` for reference.

## Module-to-module map

| MATLAB file | simplenet module | Status |
| --- | --- | --- |
| `MPReduction.m` | [`simplenet.pipeline`][simplenet.pipeline.reduce_network] | Reimplemented |
| `PreProcessData.m` | [`simplenet.preprocess`][simplenet.preprocess.preprocess] | Direct port |
| `Initiation.m` + `BuildYMat.m` | [`simplenet.ymatrix`][simplenet.ymatrix.build_b_for_reduction] | Reimplemented (scipy.sparse) |
| `DefBoundary.m` | [`simplenet.boundary`][simplenet.boundary.find_boundary_buses] | Direct port |
| `PivotData.m` + `TinneyOne.m` | (eliminated) | Replaced by Kron reduction |
| `PartialSymLU.m` + `PartialNumLU.m` | [`simplenet.kron`][simplenet.kron.kron_reduce] | Replaced by Kron reduction |
| `RODAssignment.m` + `EQRODAssignment.m` + `SelfLink.m` | (eliminated) | Internal helpers no longer needed |
| `MakeMPCr.m` + `GenerateBCIRC.m` | [`simplenet.assemble`][simplenet.assemble.assemble_reduced] | Direct port |
| `MoveExGen.m` | [`simplenet.generators`][simplenet.generators.move_external_generators] | Reimplemented (scipy.sparse.csgraph) |
| `LoadRedistribution.m` | [`simplenet.redistribute`][simplenet.redistribute.redistribute_loads] | Direct port |
| MATPOWER `rundcpf` | [`simplenet.dcpf`][simplenet.dcpf.run_dcpf] | Reimplemented (~30 lines) |
| `MapBus.m` | (folded into the pipeline) | Implementation detail |
| `case_ACTIVSg10kCopy2.m` + `matlab2*.xlsx` | [`simplenet.io.matpower`][simplenet.io.matpower.load_m] + [`simplenet.io.xlsx`][simplenet.io.xlsx.load_xlsx] | Reimplemented |
| `reduction_test.m` | [`simplenet.cli`][simplenet.cli] (`simplenet reduce ...`) | Reimplemented |

## Reduction core: Kron vs. partial LU

**MATLAB.** `PartialSymLU.m` and `PartialNumLU.m` implement a sparse
partial LU factorization (Tinney 1 optimal ordering, self-referential
link lists, RODAssignment / EQRODAssignment helpers) of the bus
admittance matrix rows that correspond to external buses. The fills
generated during factorization in the boundary-bus block are the
equivalent branches; the diagonal updates are the equivalent shunts.
This is roughly 1 000 lines across six files.

**`simplenet`.** [`kron_reduce`][simplenet.kron.kron_reduce] computes
the equivalent admittance matrix in one line:

```python
y_red = y_ii - y_ie @ scipy.sparse.linalg.spsolve(y_ee.tocsc(), y_ei.toarray())
```

The two approaches are **mathematically identical** for the DC Ward
case: partial LU factorization of the external block is one of the
classical ways to *compute* the Schur complement that defines Kron
reduction. The MATLAB code interleaves the factorization with bookkeeping
that tracks which fills belong to boundary-bus pairs (= equivalent
branches) vs. ordinary fills. `simplenet` does the algebra directly and
then extracts equivalent branches by comparing the resulting block to
the original internal block.

| | MATLAB partial LU | `simplenet` Kron |
| --- | --- | --- |
| Lines of code | ~1 000 | ~30 |
| Wall time on WECC 10k (~500 externals) | minutes | ~1 second |
| Tinney 1 ordering | manual | not needed (`spsolve` uses SuperLU) |
| Numerical stability | hand-tuned | inherits SuperLU pivoting |
| Memory peak | small `B_ee` block + L/U factors | small `B_ee` block (dense `B_ei` is `O(n_ext × n_int)`) |

**Expected output differences.** None for typical inputs. The two
methods are exact algebra of the same Schur complement, so:

- Equivalent branch reactances should agree to floating-point
  round-off (~1e-12 relative).
- Equivalent bus shunts should agree to the same tolerance.
- Branch lists may differ in **ordering** (the MATLAB code orders
  equivalent branches in the order LU fills are produced, which
  depends on Tinney ordering; `simplenet` orders them by
  `(min(f, t), max(f, t))` via `np.triu_indices`).
- The number of equivalent branches **dropped by the `|x| >= 10 *
  max(|x|_orig)` rule** at the end of the pipeline can be very
  slightly different at the boundary of the threshold, because the
  pre-pruning value of `|x|` can differ by ~1e-12. In practice this
  only affects very weakly coupled boundary-bus pairs that have
  essentially zero impact on the reduced model's behavior.

Why the change was made: simply, the original code is unmaintainable.
With `scipy.sparse` available there is no reason to reproduce a
hand-rolled sparse factorizer.

## Boundary bus detection (`DefBoundary.m`)

**MATLAB.** Walks every branch, increments a flag array.

**`simplenet`.** Vectorized via `np.isin` and XOR
([`find_boundary_buses`][simplenet.boundary.find_boundary_buses]).

Output is identical (set of retained buses adjacent to externals).

## Generator placement (`MoveExGen.m`)

**MATLAB.** Implements a custom layered Dijkstra-like search. For each
"level" it scans every branch twice (once in each direction), updates
distances, and propagates labels outward from the internal bus set. It
also collapses parallel lines into a single equivalent impedance
beforehand using a complex-impedance combination
(`z_parallel = 1 / (1/z1 + 1/z2)`).

**`simplenet`.** Same parallel-line collapse, then a single call to
`scipy.sparse.csgraph.dijkstra(..., min_only=True)` from the full
internal-bus set as multi-source
([`move_external_generators`][simplenet.generators.move_external_generators]).
This returns per-target `sources[i]` &mdash; the nearest internal bus &mdash;
which `simplenet` reads off directly.

**Expected output differences.**

- Identical generator placement when the shortest path is **unique**.
- **Tie-breaking can differ** when multiple internal buses are
  equidistant. The MATLAB layered search visits buses in branch-list
  order; `scipy.sparse.csgraph` visits them in the order SciPy's
  binary heap pops them. The two will pick different ties.
- Both algorithms are correct shortest-path algorithms, so the
  resulting electrical distance from each generator is identical; only
  the *label* of the chosen internal bus differs.
- Practical impact: the reduced model's electrical behavior is
  unchanged.

Why the change was made: `scipy.sparse.csgraph.dijkstra` is a
well-tested binary-heap Dijkstra used by half the scientific Python
ecosystem. Multi-source min-only mode plus the existing parallel-line
collapse gives identical results in essentially every realistic case.

## DC power flow and slack handling

**MATLAB.** `LoadRedistribution.m` calls MATPOWER's `rundcpf`, which:

- Picks the bus with `type == 3` (REF) as slack.
- Solves `B'·theta = P_net`.
- Returns the case with `Va` populated.

**`simplenet`.** [`run_dcpf`][simplenet.dcpf.run_dcpf]
reimplements the same math (~30 lines) using `scipy.sparse.linalg.spsolve`
to avoid the heavyweight `pypower` / `pandapower` dependency.

There is one **important behavioral difference** to be aware of:

!!! warning "Slack-bus auto-promotion in the reduced model"
    If the full model's slack bus (`type == 3`) is in the external set
    (as is the case in the 9-bus example, where bus 1 is slack and is
    eliminated), the reduced model has **no REF bus**. MATPOWER would
    error out unless the user manually promotes a new slack.
    
    `simplenet.run_dcpf` instead **auto-promotes the first PV bus** to
    slack when no REF bus is found. This is a minor convenience &mdash; it
    matches the standard practice every power-system DC PF code uses &mdash;
    but it does mean that running DC PF on the reduced model produces
    angles that are shifted by a constant relative to the full model
    (the slack reference is different). The relative angles are
    identical, and the reduced load redistribution still works
    correctly because it operates on the full model's angles before
    the reduction.

The regression test
[`test_9bus_dc_pf_consistency`](https://github.com/IMMM-SFA/simplenet/blob/main/tests/test_9bus_reduction.py)
verifies this slack-shift behavior explicitly.

## Load redistribution (`LoadRedistribution.m`)

Direct port; same algorithm step-for-step. The only differences are
mechanical:

- Bus angles are mapped back from the full model using a dict lookup
  by bus ID instead of MATLAB's `interp1q`.
- Phase shifter and HVDC corrections use the same arithmetic as the
  MATLAB code.

Output should match to floating-point round-off.

## Preprocess (`PreProcessData.m`)

Direct port. The order of operations matches the MATLAB code exactly:

1. Sort buses by ID.
2. Sort branches by `(from, to)`.
3. Drop out-of-service branches.
4. Identify isolated buses (`type == 4`).
5. Drop branches touching isolated buses.
6. Drop isolated bus rows.
7. Drop generators on isolated buses (and corresponding gencost rows).
8. Sync external bus list.
9. Drop HVDC lines touching isolated buses.

The MATLAB code also issues an `fprintf` log line after each step;
`simplenet` collects these into the `result.summary` string and the
[`PreprocessStats`][simplenet.preprocess.PreprocessStats] dataclass.

## I/O

### MATPOWER `.m` files

**MATLAB.** Uses `loadcase('case_name')` to source-execute the function
file.

**`simplenet`.** [`load_m`][simplenet.io.matpower.load_m] is a small
state-machine parser that recognizes the assignment patterns:

```matlab
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus    = [ ... ; ... ; ... ];
mpc.branch = [ ... ];
mpc.gen    = [ ... ];
mpc.gencost = [ ... ];
mpc.gentype = { 'NG'; 'WT'; ... };
mpc.bus_name = { ... };
```

Lines that delegate to `readmatrix` / `readcell` (the pattern used in
`case_ACTIVSg10kCopy2.m`) are silently skipped &mdash; the caller is then
expected to populate those fields from the xlsx workbook. The CLI's
`--gentype-xlsx` option does this transparently.

**Expected differences.** None for inline literal cases like
`test_9bus_case.m`. For the WECC / East cases the parser hands off the
data sheets to `load_xlsx`, so the loaded matrices are identical to
what MATLAB sees.

### xlsx workbooks (`matlab2*.xlsx`)

**MATLAB.** Uses `readmatrix(file, 'Sheet', 'Bus')` for numeric sheets
(`Bus`, `Gen`, `Branch`, `GenCost`) and `readcell(... , 'Range','A2')`
for string sheets (`Gentype`, `Genfuel`, `Bus Names`). `readmatrix`
auto-skips a leading header row if present.

**`simplenet`.** [`load_xlsx`][simplenet.io.xlsx.load_xlsx] follows the
same convention. The numeric reader auto-detects a leading non-numeric
row and skips it; the string-list reader uses `header=0`. The output
schema (`PowerCase` with `bus`, `gen`, `branch`, `gencost`, `gentype`,
`genfuel`, `bus_name`) mirrors the MATPOWER struct fields exactly.

**Expected differences.** None.

### Output xlsx (`Result_excluded_nodes_*.xlsx`)

**MATLAB.** `reduction_test.m` writes four sheets: `Summary`, `Gen`,
`Bus`, `Branch` with the column headers hand-coded at lines 24-26.

**`simplenet`.** [`dump_xlsx`][simplenet.io.xlsx.dump_xlsx] writes the
same four sheets with the same headers, plus the optional `GenCost`,
`Gentype`, `Genfuel`, and `Bus Names` sheets when those fields are
present.

The numeric content is identical to what `reduction_test.m` produces
up to the algorithm tolerances discussed above.

## Inputs the MATLAB code accepts that `simplenet` does **not**

`simplenet` reads the same input formats the MATLAB workflow does
&mdash; MATPOWER `.m` files, `matlab2*.xlsx` workbooks, and PSS/E
`.RAW` v33 case files via [`load_raw`][simplenet.io.psse.load_raw].
There are no remaining input-format gaps for the DC modified-Ward
pipeline.

PSS/E sections that are present in the file but not consumed by the
DC reduction (HVDC two-terminal / VSC lines, FACTS devices, impedance
correction tables, multi-section line groupings, induction machines,
GNE records) are skipped during parsing. Three-winding transformers
are mapped to a synthetic star bus plus three equivalent branches,
which matches MATPOWER's own `psse2mpc` convention.

## Things `simplenet` does that the MATLAB code does **not**

- Round-trip to / from `pypower`-style dicts via
  [`PowerCase.from_pypower`][simplenet.case.PowerCase.from_pypower]
  and [`to_pypower`][simplenet.case.PowerCase.to_pypower].
- Auto-skip leading non-numeric rows in numeric xlsx sheets
  (useful when a workbook has been hand-edited).
- Self-contained DC PF (no MATPOWER required).
- A `--no-pf` CLI flag and `pf_flag=False` API option to skip the
  full-model DC PF before load redistribution. This is useful when the
  input case already carries a valid `Va` column and you want to skip
  the redundant solve.
- A formal regression test on the 9-bus example
  (`tests/test_9bus_reduction.py`).

## Summary table

| Aspect | Same? | Expected difference |
| --- | --- | --- |
| Equivalent branch list | Yes (set) | Branch order may differ |
| Equivalent branch reactances | Yes | ~1e-12 floating-point |
| Equivalent bus shunts | Yes | ~1e-12 floating-point |
| Generator placement | Yes | Tie-breaking may differ when multiple internal buses are equidistant |
| Pruned equivalent branches | Mostly | Borderline `|x|` ≈ `10*max` may include/exclude differently |
| DC PF angles on reduced model | Up to slack shift | Slack auto-promotion when REF bus is eliminated |
| xlsx output schema | Yes | Identical |
