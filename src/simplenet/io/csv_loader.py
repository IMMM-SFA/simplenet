"""Loader for ``excluded_nodes.csv``.

The MATLAB workflow uses ``readmatrix`` on the file
``excluded_nodes_<N>.csv`` (see ``reduction_test.m`` line 18).
``readmatrix`` happily ignores a leading text header (``ExcludedNodes``
in the expected-output sample) and returns the numeric column. We
emulate that here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_excluded_nodes(path: str | Path) -> np.ndarray:
    """Read an ``excluded_nodes.csv`` file into a 1-D integer array.

    Accepts files that either have a single ``ExcludedNodes`` header row
    or no header at all.
    """

    p = Path(path)
    first_line = p.read_text(encoding="utf-8", errors="replace").splitlines()[0] if p.exists() else ""
    try:
        float(first_line.strip().split(",")[0])
        header: int | None = None
    except (ValueError, IndexError):
        header = 0

    df = pd.read_csv(p, header=header)
    if df.shape[1] == 0:
        return np.zeros(0, dtype=np.int64)
    col = df.iloc[:, 0].dropna()
    arr = np.asarray(col, dtype=float)
    int_arr = arr.astype(np.int64)
    if not np.all(int_arr == arr):
        return arr
    return int_arr
