"""Round-trip tests for the xlsx I/O adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simplenet.case import PowerCase
from simplenet.io.csv_loader import load_excluded_nodes
from simplenet.io.xlsx import dump_xlsx, load_xlsx


def _stub_case() -> PowerCase:
    rng = np.random.default_rng(0)
    bus = rng.standard_normal((5, 13))
    bus[:, 0] = [1, 2, 3, 4, 5]
    bus[:, 1] = [3, 1, 2, 1, 1]
    gen = rng.standard_normal((2, 21))
    gen[:, 0] = [1, 2]
    branch = rng.standard_normal((4, 13))
    branch[:, 0] = [1, 2, 3, 4]
    branch[:, 1] = [2, 3, 4, 5]
    branch[:, 10] = 1
    gencost = np.array([[2, 0, 0, 3, 0.1, 5, 150], [2, 0, 0, 3, 0.2, 1.2, 600]])
    return PowerCase(bus=bus, branch=branch, gen=gen, gencost=gencost, gentype=["NG", "WT"], genfuel=["ng", "wind"], bus_name=["a", "b", "c", "d", "e"])


def test_xlsx_input_round_trip(tmp_path: Path) -> None:
    case = _stub_case()
    out = tmp_path / "input.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(case.bus).to_excel(w, sheet_name="Bus", header=False, index=False)
        pd.DataFrame(case.gen).to_excel(w, sheet_name="Gen", header=False, index=False)
        pd.DataFrame(case.branch).to_excel(w, sheet_name="Branch", header=False, index=False)
        pd.DataFrame(case.gencost).to_excel(w, sheet_name="GenCost", header=False, index=False)
        pd.DataFrame({"Gentype": case.gentype}).to_excel(w, sheet_name="Gentype", index=False)
        pd.DataFrame({"Genfuel": case.genfuel}).to_excel(w, sheet_name="Genfuel", index=False)
        pd.DataFrame({"Bus Names": case.bus_name}).to_excel(w, sheet_name="Bus Names", index=False)

    loaded = load_xlsx(out)
    np.testing.assert_allclose(loaded.bus, case.bus)
    np.testing.assert_allclose(loaded.gen, case.gen)
    np.testing.assert_allclose(loaded.branch, case.branch)
    np.testing.assert_allclose(loaded.gencost, case.gencost)
    assert loaded.gentype == case.gentype
    assert loaded.genfuel == case.genfuel
    assert loaded.bus_name == case.bus_name


def test_xlsx_output(tmp_path: Path) -> None:
    case = _stub_case()
    out = tmp_path / "output.xlsx"
    dump_xlsx(case, out, summary=["Reduction summary line one", "line two"])

    written = pd.ExcelFile(out, engine="openpyxl")
    assert set(written.sheet_names) >= {"Summary", "Gen", "Bus", "Branch"}
    summary_df = pd.read_excel(out, sheet_name="Summary")
    assert summary_df.iloc[0, 0] == "Reduction summary line one"


def test_excluded_nodes_csv(tmp_path: Path) -> None:
    csv_with_header = tmp_path / "with_header.csv"
    csv_with_header.write_text("ExcludedNodes\n1\n5\n8\n")
    arr = load_excluded_nodes(csv_with_header)
    np.testing.assert_array_equal(arr, [1, 5, 8])

    csv_no_header = tmp_path / "no_header.csv"
    csv_no_header.write_text("1\n5\n8\n")
    arr = load_excluded_nodes(csv_no_header)
    np.testing.assert_array_equal(arr, [1, 5, 8])
