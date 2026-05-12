"""Parser for MATPOWER ``.m`` case files.

Handles the inline assignment style used in ``test_9bus_case.m`` and
``case9.m`` files - ``mpc.bus = [ ... ];`` matrix literals plus the
single-line ``mpc.baseMVA = 100;`` scalar assignment. Lines whose RHS
calls ``readmatrix`` / ``readcell`` are skipped silently so case files
such as ``case_ACTIVSg10kCopy2.m`` (which sources data from
``matlab2.xlsx``) can still be parsed - the caller is then expected to
populate those fields via the xlsx loader.
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import numpy as np

from simplenet.case import PowerCase

_ASSIGN_RE = re.compile(r"""
    ^\s*mpc\.(?P<field>\w+)\s*=\s*(?P<rhs>.*?)$
""", re.VERBOSE)

_COMMENT_RE = re.compile(r"%.*$")


def _strip_comment(line: str) -> str:
    return _COMMENT_RE.sub("", line)


def _parse_matrix(text: str) -> np.ndarray:
    """Parse ``[ ... ]`` matrix text into a 2-D numpy array.

    MATLAB uses ``;`` (or newlines) for row separators and whitespace or
    ``,`` for column separators. We accept either.
    """

    text = text.strip()
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    rows: list[list[float]] = []
    for raw_row in text.split(";"):
        cleaned = raw_row.replace(",", " ").strip()
        if not cleaned:
            continue
        for sub in cleaned.split("\n"):
            sub_clean = sub.strip()
            if not sub_clean:
                continue
            row = np.fromstring(sub_clean, sep=" ")
            if row.size:
                rows.append(row.tolist())
    if not rows:
        return np.zeros((0, 0))
    width = max(len(r) for r in rows)
    out = np.zeros((len(rows), width))
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def _parse_cell(text: str) -> list[str]:
    """Parse a MATLAB cell-array literal ``{ 'a'; 'b' }`` into a list of strings."""

    text = text.strip()
    if text.startswith("{"):
        text = text[1:]
    if text.endswith("}"):
        text = text[:-1]
    out: list[str] = []
    for raw in text.replace("\n", ";").split(";"):
        item = raw.strip().strip(",").strip()
        if not item:
            continue
        item = item.strip("'").strip('"')
        out.append(item)
    return out


def load_m(path: str | Path) -> PowerCase:
    """Load a MATPOWER ``.m`` case file into a :class:`PowerCase`.

    Lines that source data from external files (``readmatrix``,
    ``readcell``, etc.) are ignored and leave the corresponding field
    empty so the caller can fill it in (typically via the xlsx
    loader).
    """

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    case = PowerCase()

    pos = 0
    n = len(text)
    while pos < n:
        next_nl = text.find("\n", pos)
        if next_nl == -1:
            next_nl = n
        raw_line = text[pos:next_nl]
        pos = next_nl + 1
        line = _strip_comment(raw_line)
        m = _ASSIGN_RE.match(line)
        if not m:
            continue
        field_name = m.group("field")
        rhs = m.group("rhs").strip()
        if not rhs:
            continue
        if rhs.endswith(";"):
            rhs = rhs[:-1].rstrip()

        if rhs.startswith("[") and not rhs.endswith("]"):
            buf = StringIO()
            buf.write(rhs + "\n")
            depth = rhs.count("[") - rhs.count("]")
            while depth > 0 and pos < n:
                nxt_nl = text.find("\n", pos)
                if nxt_nl == -1:
                    nxt_nl = n
                segment = _strip_comment(text[pos:nxt_nl])
                pos = nxt_nl + 1
                buf.write(segment + "\n")
                depth += segment.count("[") - segment.count("]")
            rhs = buf.getvalue().rstrip().rstrip(";").rstrip()
        elif rhs.startswith("{") and not rhs.endswith("}"):
            buf = StringIO()
            buf.write(rhs + "\n")
            depth = rhs.count("{") - rhs.count("}")
            while depth > 0 and pos < n:
                nxt_nl = text.find("\n", pos)
                if nxt_nl == -1:
                    nxt_nl = n
                segment = _strip_comment(text[pos:nxt_nl])
                pos = nxt_nl + 1
                buf.write(segment + "\n")
                depth += segment.count("{") - segment.count("}")
            rhs = buf.getvalue().rstrip().rstrip(";").rstrip()

        if "readmatrix" in rhs or "readcell" in rhs or "readtable" in rhs:
            continue

        if field_name == "version":
            case.version = rhs.strip().strip("'").strip('"')
        elif field_name == "baseMVA":
            try:
                case.base_mva = float(rhs.strip())
            except ValueError:
                continue
        elif field_name == "bus":
            case.bus = _parse_matrix(rhs)
        elif field_name == "gen":
            case.gen = _parse_matrix(rhs)
        elif field_name == "branch":
            case.branch = _parse_matrix(rhs)
        elif field_name == "gencost":
            case.gencost = _parse_matrix(rhs)
        elif field_name == "dcline":
            case.dcline = _parse_matrix(rhs)
        elif field_name in ("gentype",):
            case.gentype = _parse_cell(rhs)
        elif field_name in ("genfuel",):
            case.genfuel = _parse_cell(rhs)
        elif field_name in ("bus_name",):
            case.bus_name = _parse_cell(rhs)

    return case
