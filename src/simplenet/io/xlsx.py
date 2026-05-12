"""xlsx I/O matching the ``matlab2*.xlsx`` schema used by the MATLAB workflow.

Input layout (sheets):
    - ``Bus``        bus matrix (13-17 columns)
    - ``Gen``        gen matrix (21-25 columns)
    - ``Branch``     branch matrix (13-21 columns)
    - ``GenCost``    optional gencost matrix
    - ``Gentype``    optional 1-column string list (header row, then values)
    - ``Genfuel``    optional 1-column string list
    - ``Bus Names``  optional 1-column string list

The MATLAB code uses ``readmatrix`` / ``readcell`` with ``'Range','A2'``
for the string sheets - i.e. it skips the header row. Numeric sheets
have no header.

Output (``dump_xlsx``) mirrors ``reduction_test.m``'s ``Result_*.xlsx``:
sheets ``Summary``, ``Gen``, ``Bus``, ``Branch`` with the column headers
defined there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simplenet.case import (
    BRANCH_FULL_HEADER,
    BUS_FULL_HEADER,
    GEN_FULL_HEADER,
    PowerCase,
)


def _read_matrix(path: Path, sheet: str) -> np.ndarray | None:
    """Read a numeric sheet into a 2-D array.

    Returns ``None`` if the sheet does not exist. Leading non-numeric
    rows (e.g. a column-header row added when the workbook was edited
    by hand) are auto-detected and skipped to mimic MATLAB's
    ``readmatrix`` behavior.
    """

    try:
        df = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    except ValueError:
        return None
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        return np.zeros((0, 0))
    skip = 0
    for _, row in df.iterrows():
        try:
            row.astype(float)
            break
        except (TypeError, ValueError):
            skip += 1
    if skip:
        df = df.iloc[skip:]
        if df.empty:
            return np.zeros((0, 0))
    return df.to_numpy(dtype=float, copy=True)


def _read_string_list(path: Path, sheet: str) -> list[str] | None:
    """Read a single-column string sheet (with a header row in row 1)."""

    try:
        df = pd.read_excel(path, sheet_name=sheet, header=0, engine="openpyxl")
    except ValueError:
        return None
    if df.empty:
        return []
    col = df.iloc[:, 0]
    return [str(v) for v in col.dropna().tolist()]


def load_xlsx(path: str | Path) -> PowerCase:
    """Load a ``matlab2*.xlsx`` style workbook into a :class:`PowerCase`.

    The ``baseMVA`` is fixed to 100 to match
    ``case_ACTIVSg10kCopy2.m``. Override ``case.base_mva`` after loading
    if the source case uses a different base.
    """

    p = Path(path)
    bus = _read_matrix(p, "Bus")
    gen = _read_matrix(p, "Gen")
    branch = _read_matrix(p, "Branch")
    if bus is None or gen is None or branch is None:
        raise ValueError(
            f"{path}: expected Bus/Gen/Branch sheets, got {bus is not None}/"
            f"{gen is not None}/{branch is not None}"
        )

    gencost = _read_matrix(p, "GenCost")
    if gencost is not None and gencost.size == 0:
        gencost = None

    gentype = _read_string_list(p, "Gentype")
    genfuel = _read_string_list(p, "Genfuel")
    bus_name = _read_string_list(p, "Bus Names")
    if bus_name is None:
        bus_name = _read_string_list(p, "Bus_Names")

    return PowerCase(
        base_mva=100.0,
        bus=bus,
        gen=gen,
        branch=branch,
        gencost=gencost,
        gentype=gentype,
        genfuel=genfuel,
        bus_name=bus_name,
    )


def _trim_header(header: list[str], ncols: int) -> list[str]:
    if ncols <= len(header):
        return header[:ncols]
    extras = [f"col_{i}" for i in range(len(header), ncols)]
    return header + extras


def dump_xlsx(
    reduced: PowerCase,
    path: str | Path,
    *,
    summary: str | list[str] | None = None,
) -> None:
    """Write a reduced :class:`PowerCase` to a multi-sheet xlsx.

    Mirrors the output of ``reduction_test.m``: sheets ``Summary`` (one
    column of text lines), ``Gen``, ``Bus``, ``Branch`` (with header
    rows from the original MATLAB script). Generator type / fuel /
    bus-name lists, when present, are appended as their own sheets.
    """

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if summary is None:
        summary_lines: list[str] = []
    elif isinstance(summary, str):
        summary_lines = summary.splitlines()
    else:
        summary_lines = list(summary)
    summary_df = pd.DataFrame({"Summary": summary_lines})

    bus_header = _trim_header(BUS_FULL_HEADER, reduced.bus.shape[1])
    gen_header = _trim_header(GEN_FULL_HEADER, reduced.gen.shape[1])
    branch_header = _trim_header(BRANCH_FULL_HEADER, reduced.branch.shape[1])

    bus_df = pd.DataFrame(reduced.bus, columns=bus_header)
    gen_df = pd.DataFrame(reduced.gen, columns=gen_header)
    branch_df = pd.DataFrame(reduced.branch, columns=branch_header)

    with pd.ExcelWriter(p, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        gen_df.to_excel(writer, sheet_name="Gen", index=False)
        bus_df.to_excel(writer, sheet_name="Bus", index=False)
        branch_df.to_excel(writer, sheet_name="Branch", index=False)
        if reduced.gencost is not None and reduced.gencost.size:
            pd.DataFrame(reduced.gencost).to_excel(writer, sheet_name="GenCost", index=False, header=False)
        if reduced.gentype:
            pd.DataFrame({"Gentype": list(reduced.gentype)}).to_excel(writer, sheet_name="Gentype", index=False)
        if reduced.genfuel:
            pd.DataFrame({"Genfuel": list(reduced.genfuel)}).to_excel(writer, sheet_name="Genfuel", index=False)
        if reduced.bus_name:
            pd.DataFrame({"Bus Names": list(reduced.bus_name)}).to_excel(writer, sheet_name="Bus Names", index=False)


def dump_excluded_template(path: str | Path, bus_ids: list[Any]) -> None:
    """Write a one-column ``excluded_nodes.csv``-style template file."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ExcludedNodes": list(bus_ids)}).to_csv(p, index=False)
