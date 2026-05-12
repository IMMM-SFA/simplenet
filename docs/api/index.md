# API Reference

This section is auto-generated from the package's PEP-compliant
docstrings via [`mkdocstrings`](https://mkdocstrings.github.io/).

## Public top-level API

```python
from simplenet import (
    PowerCase,
    reduce_network,
    preprocess,
    kron_reduce,
    move_external_generators,
    redistribute_loads,
    run_dcpf,
    build_b_for_reduction,
    build_b_for_dcpf,
)
from simplenet.io import (
    load_m,
    load_xlsx,
    dump_xlsx,
    load_excluded_nodes,
)
```

## Result dataclasses

- [`PowerCase`][simplenet.case.PowerCase] - the core MATPOWER-style case
  representation.
- [`ReductionResult`][simplenet.pipeline.ReductionResult] - output of
  [`reduce_network`][simplenet.pipeline.reduce_network].
- [`PreprocessStats`][simplenet.preprocess.PreprocessStats] - counts of
  items removed during preprocessing.
- [`KronResult`][simplenet.kron.KronResult] - reduced admittance matrix
  + original internal block.
- [`AssembleResult`][simplenet.assemble.AssembleResult] - reduced case
  + branch circuit numbers.
- [`GenMoveResult`][simplenet.generators.GenMoveResult] - generator
  placement + bus mapping.
- [`DCPFResult`][simplenet.dcpf.DCPFResult] - DC power flow output.

## Modules

| Module | Purpose |
| --- | --- |
| [`simplenet.case`](case.md) | `PowerCase` data class and MATPOWER column constants |
| [`simplenet.pipeline`](pipeline.md) | End-to-end reduction pipeline (`reduce_network`) |
| [`simplenet.preprocess`](preprocess.md) | Drop isolated buses and out-of-service branches |
| [`simplenet.ymatrix`](ymatrix.md) | Sparse DC susceptance matrix builders |
| [`simplenet.boundary`](boundary.md) | Boundary bus identification |
| [`simplenet.kron`](kron.md) | Core Kron reduction math |
| [`simplenet.assemble`](assemble.md) | Build reduced case from a Kron result |
| [`simplenet.generators`](generators.md) | External generator placement (shortest electrical distance) |
| [`simplenet.dcpf`](dcpf.md) | Standalone DC power flow |
| [`simplenet.redistribute`](redistribute.md) | Load redistribution on the reduced model |
| [`simplenet.io.matpower`](io_matpower.md) | MATPOWER `.m` parser |
| [`simplenet.io.xlsx`](io_xlsx.md) | `matlab2*.xlsx` reader / writer |
| [`simplenet.io.csv_loader`](io_csv.md) | `excluded_nodes.csv` reader |
| [`simplenet.cli`](cli.md) | Command-line interface |
