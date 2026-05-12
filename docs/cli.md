# CLI Reference

`simplenet` ships a command-line entry point that mirrors the
[`reduction_test.m`](https://github.com/IMMM-SFA/simplenet/blob/main/matlab/reduction_test.m)
workflow.

```text
$ simplenet --help
Usage: simplenet [OPTIONS] COMMAND [ARGS]...

  simplenet - DC modified-Ward network reduction CLI.

Commands:
  info    Print summary statistics for CASE_PATH.
  reduce  Run network reduction on CASE_PATH using buses listed in EXCLUDED_PATH.
```

## `simplenet reduce`

Reduce a case to a smaller equivalent network.

```text
$ simplenet reduce --help
Usage: simplenet reduce [OPTIONS] CASE_PATH EXCLUDED_PATH

  Run network reduction on CASE_PATH using buses listed in EXCLUDED_PATH.

  Examples
      simplenet reduce case9.m excluded.csv -o reduced.xlsx
      simplenet reduce matlab2_WECC.xlsx excluded_550.csv -o result.xlsx

Options:
  -o, --output FILE     Path to write the reduced multi-sheet xlsx.  [required]
  --summary-txt FILE    Optional path to also dump the summary diary as a .txt
                        file.
  --pf / --no-pf        Solve a DC PF on the full model before redistributing
                        loads (default: --pf, matches Pf_flag=1).
  --gentype-xlsx FILE   If the case file is a MATPOWER .m that delegates
                        Gen/Bus/etc. data to an xlsx, point to it here.
  --help                Show this message and exit.
```

### `CASE_PATH` formats

Auto-detected by file extension:

| Extension | Loader |
| --- | --- |
| `.m` | [`load_m`][simplenet.io.matpower.load_m] |
| `.xlsx` / `.xlsm` | [`load_xlsx`][simplenet.io.xlsx.load_xlsx] |
| `.raw` / `.RAW` | [`load_raw`][simplenet.io.psse.load_raw] (PSS/E v33) |
| `.json` | `PowerCase.from_pypower(json.load(...))` |

### Examples

#### 9-bus worked example

```bash
echo "ExcludedNodes
1
5
8" > excluded.csv

simplenet reduce matlab/NetworkReduction2/test_9bus_case.m \
    excluded.csv \
    -o case9_reduced.xlsx
```

Output:

```text
Loaded 9 buses, 9 branches, 3 gens
Excluding 3 buses
Reduction process start
Preprocess data
Eliminate 0 isolated buses
Eliminate 0 branches
Eliminate 0 generators
Preprocessing complete
Convert input data model
Creating Y matrix of input full model
Do first round reduction eliminating all external buses
1 external generators are to be placed
Do second round reduction eliminating all external non-generator buses
Placing External generators
Redistribute loads
**********Reduction Summary****************
6 buses in reduced model
7 branches in reduced model, including 4 equivalent lines
3 generators in reduced model
**********Generator Placement Results**************
External generator on bus 1 is moved to 4
```

#### TAMU WECC workflow

```bash
simplenet reduce matlab/matlab2_WECC.xlsx \
    matlab/expected_output/excluded_nodes.csv \
    -o Result_excluded_nodes.xlsx \
    --summary-txt Result_excluded_nodes.txt
```

This is the direct Python analogue of the MATLAB
[`reduction_test.m`](https://github.com/IMMM-SFA/simplenet/blob/main/matlab/reduction_test.m)
script (lines 14-22).

#### Skipping the full-model DC power flow

Use `--no-pf` when the input case already has a valid `Va` column from
a previous solve. The reduction reuses those angles directly instead of
re-solving the full DC PF.

```bash
simplenet reduce case.xlsx excluded.csv -o reduced.xlsx --no-pf
```

#### Starting from a PSS/E `.RAW` file

```bash
simplenet info ACTIVSg10k.RAW
simplenet reduce ACTIVSg10k.RAW excluded.csv -o reduced.xlsx
```

`load_raw` targets the PSS/E v33 layout that the TAMU synthetic
grids (`ACTIVSg10k.RAW`, `ACTIVSg70k.RAW`) emit, but also accepts
v32 / v34 / v35. Two-winding transformers become MATPOWER branches
with their tap ratio and phase shift; three-winding transformers
expand into a synthetic star bus plus three equivalent branches (the
same convention MATPOWER's `psse2mpc` uses). Switched-shunt initial
setpoints are folded into each bus's `Bs`.

#### Using a MATPOWER `.m` that delegates to xlsx

If your case file is `case_ACTIVSg10kCopy2.m`-style (i.e. it uses
`readmatrix('matlab2.xlsx', ...)` to source bus / gen / branch data),
pass `--gentype-xlsx` to combine the `.m` scalars (`baseMVA`, `version`)
with the xlsx matrices:

```bash
simplenet reduce matlab/case_ACTIVSg10kCopy2.m excluded.csv \
    -o result.xlsx \
    --gentype-xlsx matlab/matlab2_WECC.xlsx
```

## `simplenet info`

Print a one-line summary of a case file:

```text
$ simplenet info matlab/matlab2_WECC.xlsx
baseMVA: 100.0
buses:   10000
gens:    2612
branches:12706
gencost: (2612, 7)
gentype: 2612 entries
genfuel: 2612 entries
bus_name: 10000 entries
```

## Output schema

`simplenet reduce` writes a multi-sheet xlsx with the same column
layout that MATLAB's `reduction_test.m` uses:

| Sheet | Source columns |
| --- | --- |
| `Summary` | One column of diary lines (one per `fprintf` in the MATLAB pipeline) |
| `Gen` | `Bus`, `Pg`, `Qg`, `Qmax`, `Qmin`, `Vg`, `mBase`, `status`, `Pmax`, `Pmin`, `Pc1`, `Pc2`, `Qc1min`, `Qc1max`, `Qc2min`, `Qc2max`, `ramp_agc`, `ramp_10`, `ramp_30`, `ramp_q`, `apf` (+ `mu_*`) |
| `Bus` | `bus_i`, `type`, `Pd`, `Qd`, `Gs`, `Bs`, `area`, `Vm`, `Va`, `baseKV`, `zone`, `Vmax`, `Vmin` (+ `lam_*`, `mu_*`) |
| `Branch` | `fbus`, `tbus`, `r`, `x`, `b`, `rateA`, `rateB`, `rateC`, `ratio`, `angle`, `status`, `angmin`, `angmax` (+ `Pf`, `Qf`, `Pt`, `Qt`, `mu_*`) |
| `GenCost` *(optional)* | Generator cost matrix, no header |
| `Gentype` *(optional)* | Single-column string list |
| `Genfuel` *(optional)* | Single-column string list |
| `Bus Names` *(optional)* | Single-column string list |
