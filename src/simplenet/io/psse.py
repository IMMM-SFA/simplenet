"""Parser for PSS/E ``.RAW`` (Power Flow Raw Data) case files.

Targets the v33 layout that the TAMU synthetic grids
(``ACTIVSg10k.RAW``, ``ACTIVSg70k.RAW``, ...) ship in, but is tolerant
of v32 / v34 / v35 for the sections used by ``simplenet``:

* Bus
* Load
* Fixed shunt
* Generator
* Non-transformer branch
* Two-winding transformer
* Three-winding transformer (mapped to a synthetic star bus + three
  equivalent branches, MATPOWER's standard conversion)
* Switched shunt (its initial setpoint is folded into ``Bs``)

The result is a :class:`simplenet.case.PowerCase` using MATPOWER v2
column conventions, so the rest of the pipeline (``preprocess``, the
DC Y-matrix, Kron reduction, ...) accepts it unchanged.

Only the fields the DC modified-Ward pipeline cares about are
populated. Items the format carries but the pipeline does not need
(area / zone / owner / FACTS / DC links / cost data) are parsed
defensively and otherwise ignored. HVDC lines are not converted -
they appear instead as zero-injection buses unless you handle them
upstream.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from simplenet.case import (
    BRANCH_COLUMNS,
    BUS_COLUMNS,
    GEN_COLUMNS,
    ISOLATED_BUS,
    PQ_BUS,
    PV_BUS,
    REF_BUS,
    PowerCase,
)

# Section names in PSS/E v33 order. Sections we do not consume are
# still listed so the section index stays aligned with the file.
_SECTION_ORDER: tuple[str, ...] = (
    "bus",
    "load",
    "fixed_shunt",
    "generator",
    "branch",
    "transformer",
    "area",
    "two_terminal_dc",
    "vsc_dc",
    "imp_correction",
    "multi_terminal_dc",
    "multi_section_line",
    "zone",
    "interarea_transfer",
    "owner",
    "facts",
    "switched_shunt",
    "gne",
    "induction_machine",
)

# A section terminator in PSS/E is a record whose first field is the
# bare integer 0 (optionally followed by a "/ end of ..." comment).
_SECTION_TERMINATOR_RE = re.compile(r"^\s*0\s*(?:[,/]|$)")


def _strip_psse_comment(line: str) -> str:
    """Strip the ``/`` PSS/E inline-comment tail, respecting quoted strings."""

    in_quote = False
    for i, ch in enumerate(line):
        if ch == "'":
            in_quote = not in_quote
        elif ch == "/" and not in_quote:
            return line[:i]
    return line


def _split_row(line: str) -> list[str]:
    """Comma-split a PSS/E record, ignoring commas inside ``'...'`` strings."""

    parts: list[str] = []
    buf: list[str] = []
    in_quote = False
    for ch in line:
        if ch == "'":
            in_quote = not in_quote
            buf.append(ch)
        elif ch == "," and not in_quote:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().strip("'").strip('"'))
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip().strip("'").strip('"')))
    except (TypeError, ValueError):
        return default


def _get(row: list[str], idx: int, default: str = "") -> str:
    return row[idx] if idx < len(row) else default


def _parse_sections(lines: list[str]) -> dict[str, list[list[list[str]]]]:
    """Walk the file body and group records into the ordered sections.

    Returns a mapping from section name (see ``_SECTION_ORDER``) to a
    list of records. Most records are a single comma-split row; the
    transformer section yields a list of rows per record (4 rows for a
    two-winding, 5 for a three-winding transformer).
    """

    sections: dict[str, list[list[list[str]]]] = {name: [] for name in _SECTION_ORDER}
    section_idx = 0
    current_section: str | None = _SECTION_ORDER[0]

    cursor = 0
    n = len(lines)
    while cursor < n:
        raw = lines[cursor]
        line = _strip_psse_comment(raw).rstrip()
        if not line.strip():
            cursor += 1
            continue

        if _SECTION_TERMINATOR_RE.match(line):
            section_idx += 1
            current_section = (
                _SECTION_ORDER[section_idx] if section_idx < len(_SECTION_ORDER) else None
            )
            cursor += 1
            continue

        if current_section is None:
            cursor += 1
            continue

        if current_section == "transformer":
            first_row = _split_row(line)
            cursor += 1
            k_bus = _as_int(_get(first_row, 2), 0)
            n_extra = 3 if k_bus == 0 else 4
            record = [first_row]
            for _ in range(n_extra):
                if cursor >= n:
                    break
                inner = _strip_psse_comment(lines[cursor]).rstrip()
                cursor += 1
                if not inner.strip():
                    continue
                record.append(_split_row(inner))
            sections["transformer"].append(record)
            continue

        sections[current_section].append([_split_row(line)])
        cursor += 1

    return sections


def _delta_to_star(
    z12: complex, z23: complex, z31: complex
) -> tuple[complex, complex, complex]:
    """Convert delta-connected pairwise impedances to star-equivalent.

    Returns ``(z_1, z_2, z_3)`` such that a wye / star network with
    those leg impedances is electrically equivalent to a delta with
    the supplied pair impedances. This is the standard PSS/E
    three-winding-to-MATPOWER conversion.
    """

    z1 = 0.5 * (z12 + z31 - z23)
    z2 = 0.5 * (z12 + z23 - z31)
    z3 = 0.5 * (z23 + z31 - z12)
    return z1, z2, z3


def _convert_xfmr_impedance(
    r: float, x: float, sbase_xf: float, sbase_sys: float, cz: int
) -> tuple[float, float]:
    """Return (r, x) on the system MVA base for the given PSS/E CZ code.

    * ``CZ == 1`` - already on system per-unit (no conversion).
    * ``CZ == 2`` - on the transformer's own ``sbase_xf`` per-unit.
    * ``CZ == 3`` - load-loss in watts and X in p.u. on ``sbase_xf``;
      we approximate the conversion as well as possible.
    """

    if cz == 1 or sbase_xf <= 0.0:
        return r, x
    if cz == 2:
        scale = sbase_sys / sbase_xf
        return r * scale, x * scale
    if cz == 3:
        # R from load-loss watts -> p.u. = (W / 1e6) / SBASE12, then to system base
        r_pu_xf = (r / 1.0e6) / sbase_xf
        scale = sbase_sys / sbase_xf
        return r_pu_xf * scale, x * scale
    return r, x


def _convert_tap_ratio(windv1: float, windv2: float, cw: int) -> float:
    """Return MATPOWER off-nominal tap ratio for the given CW convention.

    * ``CW == 1`` - ``WINDV`` is the off-nominal turns ratio in p.u.
      of the bus base, so the MATPOWER ratio is just ``WINDV1/WINDV2``.
    * ``CW == 2`` - ``WINDV`` is the actual winding voltage in kV;
      MATPOWER's ratio still works out to ``WINDV1/WINDV2`` because
      MATPOWER bakes the bus-base ratio in separately.
    * ``CW == 3`` - ``WINDV`` is in p.u. of the winding nominal kV
      ``NOMV``; we still approximate with ``WINDV1/WINDV2`` (this is
      the convention MATPOWER's ``psse2mpc`` uses when ``NOMV`` is 0).
    """

    if windv2 == 0.0:
        return 1.0
    return windv1 / windv2


def _build_case(
    sections: dict[str, list[list[list[str]]]], base_mva: float
) -> PowerCase:
    """Assemble the parsed sections into a :class:`PowerCase`."""

    # --- buses -----------------------------------------------------------
    bus_records = [rec[0] for rec in sections["bus"]]
    n_bus = len(bus_records)
    if n_bus == 0:
        raise ValueError("PSS/E raw file contained no bus records")

    bus = np.zeros((n_bus, BUS_COLUMNS))
    bus_index_by_id: dict[int, int] = {}
    ide_to_mp = {1: PQ_BUS, 2: PV_BUS, 3: REF_BUS, 4: ISOLATED_BUS}
    for i, row in enumerate(bus_records):
        bus_i = _as_int(_get(row, 0))
        basekv = _as_float(_get(row, 2), 0.0)
        ide = _as_int(_get(row, 3), 1)
        area = _as_int(_get(row, 4), 1)
        zone = _as_int(_get(row, 5), 1)
        vm = _as_float(_get(row, 7), 1.0)
        va = _as_float(_get(row, 8), 0.0)
        vmax = _as_float(_get(row, 9), 1.1)
        vmin = _as_float(_get(row, 10), 0.9)
        bus[i, 0] = bus_i
        bus[i, 1] = ide_to_mp.get(ide, PQ_BUS)
        bus[i, 6] = area
        bus[i, 7] = vm
        bus[i, 8] = va
        bus[i, 9] = basekv
        bus[i, 10] = zone
        bus[i, 11] = vmax if vmax > 0 else 1.1
        bus[i, 12] = vmin if vmin > 0 else 0.9
        bus_index_by_id[bus_i] = i

    # --- loads (aggregate Pd/Qd per bus) ---------------------------------
    for record in sections["load"]:
        row = record[0]
        bus_i = _as_int(_get(row, 0))
        status = _as_int(_get(row, 2), 1)
        if status == 0 or bus_i not in bus_index_by_id:
            continue
        idx = bus_index_by_id[bus_i]
        # Constant power + constant current (at 1 pu V) + constant admittance
        # all collapse to one MW/Mvar number under DC assumptions.
        pl = _as_float(_get(row, 5), 0.0)
        ql = _as_float(_get(row, 6), 0.0)
        ip = _as_float(_get(row, 7), 0.0)
        iq = _as_float(_get(row, 8), 0.0)
        yp = _as_float(_get(row, 9), 0.0)
        yq = _as_float(_get(row, 10), 0.0)
        bus[idx, 2] += pl + ip + yp
        bus[idx, 3] += ql + iq + yq

    # --- fixed shunts ----------------------------------------------------
    for record in sections["fixed_shunt"]:
        row = record[0]
        bus_i = _as_int(_get(row, 0))
        status = _as_int(_get(row, 2), 1)
        if status == 0 or bus_i not in bus_index_by_id:
            continue
        idx = bus_index_by_id[bus_i]
        gl = _as_float(_get(row, 3), 0.0)
        bl = _as_float(_get(row, 4), 0.0)
        bus[idx, 4] += gl
        bus[idx, 5] += bl

    # --- switched shunts (initial setpoint only) -------------------------
    for record in sections["switched_shunt"]:
        row = record[0]
        bus_i = _as_int(_get(row, 0))
        stat = _as_int(_get(row, 3), 1)
        if stat == 0 or bus_i not in bus_index_by_id:
            continue
        idx = bus_index_by_id[bus_i]
        binit = _as_float(_get(row, 9), 0.0)
        bus[idx, 5] += binit

    # --- generators ------------------------------------------------------
    gen_list: list[np.ndarray] = []
    for record in sections["generator"]:
        row = record[0]
        bus_i = _as_int(_get(row, 0))
        if bus_i not in bus_index_by_id:
            continue
        pg = _as_float(_get(row, 2), 0.0)
        qg = _as_float(_get(row, 3), 0.0)
        qt = _as_float(_get(row, 4), 9999.0)
        qb = _as_float(_get(row, 5), -9999.0)
        vs = _as_float(_get(row, 6), 1.0)
        mbase = _as_float(_get(row, 8), base_mva)
        stat = _as_int(_get(row, 14), 1)
        pt = _as_float(_get(row, 16), 0.0)
        pb = _as_float(_get(row, 17), 0.0)

        gen_row = np.zeros(GEN_COLUMNS)
        gen_row[0] = bus_i
        gen_row[1] = pg
        gen_row[2] = qg
        gen_row[3] = qt
        gen_row[4] = qb
        gen_row[5] = vs
        gen_row[6] = mbase if mbase > 0 else base_mva
        gen_row[7] = stat
        gen_row[8] = pt
        gen_row[9] = pb
        gen_list.append(gen_row)
    gen = np.vstack(gen_list) if gen_list else np.zeros((0, GEN_COLUMNS))

    # --- non-transformer branches ----------------------------------------
    branch_list: list[np.ndarray] = []
    for record in sections["branch"]:
        row = record[0]
        i_bus = _as_int(_get(row, 0))
        j_bus = _as_int(_get(row, 1))
        if i_bus not in bus_index_by_id or j_bus not in bus_index_by_id:
            continue
        r = _as_float(_get(row, 3), 0.0)
        x = _as_float(_get(row, 4), 0.0)
        b = _as_float(_get(row, 5), 0.0)
        rate_a = _as_float(_get(row, 6), 0.0)
        rate_b = _as_float(_get(row, 7), 0.0)
        rate_c = _as_float(_get(row, 8), 0.0)
        st = _as_int(_get(row, 13), 1)
        br = np.zeros(BRANCH_COLUMNS)
        br[0] = abs(i_bus)
        br[1] = abs(j_bus)
        br[2] = r
        br[3] = x
        br[4] = b
        br[5] = rate_a
        br[6] = rate_b
        br[7] = rate_c
        br[10] = st
        br[11] = -360.0
        br[12] = 360.0
        branch_list.append(br)

    # --- transformers (2- and 3-winding) ---------------------------------
    star_bus_rows: list[np.ndarray] = []
    next_star_id = (max(bus_index_by_id, default=0) // 1000 + 1) * 1000 + 1
    for record in sections["transformer"]:
        l1 = record[0] if len(record) > 0 else []
        l2 = record[1] if len(record) > 1 else []
        l3 = record[2] if len(record) > 2 else []
        l4 = record[3] if len(record) > 3 else []
        l5 = record[4] if len(record) > 4 else []

        i_bus = _as_int(_get(l1, 0))
        j_bus = _as_int(_get(l1, 1))
        k_bus = _as_int(_get(l1, 2), 0)
        cw = _as_int(_get(l1, 4), 1)
        cz = _as_int(_get(l1, 5), 1)
        stat = _as_int(_get(l1, 11), 1)

        if k_bus == 0:
            # Two-winding transformer
            if i_bus not in bus_index_by_id or j_bus not in bus_index_by_id:
                continue
            r12 = _as_float(_get(l2, 0), 0.0)
            x12 = _as_float(_get(l2, 1), 0.0)
            sbase12 = _as_float(_get(l2, 2), base_mva)
            r_pu, x_pu = _convert_xfmr_impedance(r12, x12, sbase12, base_mva, cz)

            windv1 = _as_float(_get(l3, 0), 1.0)
            ang1 = _as_float(_get(l3, 2), 0.0)
            rata1 = _as_float(_get(l3, 3), 0.0)
            ratb1 = _as_float(_get(l3, 4), 0.0)
            ratc1 = _as_float(_get(l3, 5), 0.0)
            windv2 = _as_float(_get(l4, 0), 1.0)
            ratio = _convert_tap_ratio(windv1, windv2, cw)

            br = np.zeros(BRANCH_COLUMNS)
            br[0] = i_bus
            br[1] = j_bus
            br[2] = r_pu
            br[3] = x_pu
            br[5] = rata1
            br[6] = ratb1
            br[7] = ratc1
            br[8] = ratio if ratio != 1.0 else ratio  # MATPOWER allows 0 == 1.0
            br[9] = ang1
            br[10] = 1 if stat >= 1 else 0
            br[11] = -360.0
            br[12] = 360.0
            branch_list.append(br)
            continue

        # Three-winding transformer: add a star bus + three branches.
        if (
            i_bus not in bus_index_by_id
            or j_bus not in bus_index_by_id
            or k_bus not in bus_index_by_id
        ):
            continue
        r12 = _as_float(_get(l2, 0), 0.0)
        x12 = _as_float(_get(l2, 1), 0.0)
        sbase12 = _as_float(_get(l2, 2), base_mva)
        r23 = _as_float(_get(l2, 3), 0.0)
        x23 = _as_float(_get(l2, 4), 0.0)
        sbase23 = _as_float(_get(l2, 5), base_mva)
        r31 = _as_float(_get(l2, 6), 0.0)
        x31 = _as_float(_get(l2, 7), 0.0)
        sbase31 = _as_float(_get(l2, 8), base_mva)
        vmstar = _as_float(_get(l2, 9), 1.0)
        anstar = _as_float(_get(l2, 10), 0.0)

        r12s, x12s = _convert_xfmr_impedance(r12, x12, sbase12, base_mva, cz)
        r23s, x23s = _convert_xfmr_impedance(r23, x23, sbase23, base_mva, cz)
        r31s, x31s = _convert_xfmr_impedance(r31, x31, sbase31, base_mva, cz)
        z1, z2, z3 = _delta_to_star(
            complex(r12s, x12s), complex(r23s, x23s), complex(r31s, x31s)
        )

        windv1 = _as_float(_get(l3, 0), 1.0)
        ang1 = _as_float(_get(l3, 2), 0.0)
        rata1 = _as_float(_get(l3, 3), 0.0)
        ratb1 = _as_float(_get(l3, 4), 0.0)
        ratc1 = _as_float(_get(l3, 5), 0.0)
        windv2 = _as_float(_get(l4, 0), 1.0)
        ang2 = _as_float(_get(l4, 2), 0.0)
        rata2 = _as_float(_get(l4, 3), 0.0)
        ratb2 = _as_float(_get(l4, 4), 0.0)
        ratc2 = _as_float(_get(l4, 5), 0.0)
        windv3 = _as_float(_get(l5, 0), 1.0)
        ang3 = _as_float(_get(l5, 2), 0.0)
        rata3 = _as_float(_get(l5, 3), 0.0)
        ratb3 = _as_float(_get(l5, 4), 0.0)
        ratc3 = _as_float(_get(l5, 5), 0.0)

        while next_star_id in bus_index_by_id:
            next_star_id += 1
        star_id = next_star_id
        next_star_id += 1
        star_row = np.zeros(BUS_COLUMNS)
        star_row[0] = star_id
        star_row[1] = PQ_BUS
        star_row[6] = bus[bus_index_by_id[i_bus], 6]
        star_row[7] = vmstar
        star_row[8] = anstar
        star_row[10] = bus[bus_index_by_id[i_bus], 10]
        star_row[11] = 1.1
        star_row[12] = 0.9
        star_bus_rows.append(star_row)
        bus_index_by_id[star_id] = -1

        for other_bus, z, windv, ang, rata, ratb, ratc in (
            (i_bus, z1, windv1, ang1, rata1, ratb1, ratc1),
            (j_bus, z2, windv2, ang2, rata2, ratb2, ratc2),
            (k_bus, z3, windv3, ang3, rata3, ratb3, ratc3),
        ):
            ratio = windv if windv != 0 else 1.0
            br = np.zeros(BRANCH_COLUMNS)
            br[0] = other_bus
            br[1] = star_id
            br[2] = z.real
            br[3] = z.imag
            br[5] = rata
            br[6] = ratb
            br[7] = ratc
            br[8] = ratio
            br[9] = ang
            br[10] = 1 if stat >= 1 else 0
            br[11] = -360.0
            br[12] = 360.0
            branch_list.append(br)

    if star_bus_rows:
        bus = np.vstack([bus, np.array(star_bus_rows)])

    branch = np.vstack(branch_list) if branch_list else np.zeros((0, BRANCH_COLUMNS))

    return PowerCase(
        base_mva=float(base_mva) if base_mva > 0 else 100.0,
        bus=bus,
        gen=gen,
        branch=branch,
    )


def load_raw(path: str | Path) -> PowerCase:
    """Load a PSS/E ``.RAW`` v33-style case file into a :class:`PowerCase`.

    The header line supplies ``SBASE`` (which maps to ``base_mva``)
    and ``REV`` (the format version). Versions 32 / 33 / 34 / 35 are
    all accepted; older versions emit a warning if structural
    differences cause sections to be misaligned.

    Parameters
    ----------
    path
        Filesystem path to a ``.raw`` / ``.RAW`` file.

    Returns
    -------
    PowerCase
        Bus / gen / branch matrices populated using MATPOWER v2 column
        conventions. Three-winding transformers are exploded into a
        synthetic star bus plus three equivalent branches.

    Notes
    -----
    Only sections used by the DC modified-Ward pipeline are converted.
    Cost data (``gencost``), generator type / fuel labels, and bus
    names are not present in PSS/E and therefore stay ``None``.
    HVDC / VSC / FACTS / multi-section / multi-terminal records are
    ignored.
    """

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    raw_lines = text.splitlines()
    if len(raw_lines) < 3:
        raise ValueError(f"{path}: too short to be a PSS/E .RAW file")

    header_row = _split_row(_strip_psse_comment(raw_lines[0]))
    sbase = _as_float(_get(header_row, 1), 100.0)
    rev = _as_int(_get(header_row, 2), 33)
    if rev and rev < 32:
        warnings.warn(
            f"{path}: PSS/E REV={rev} is older than v32; section layout "
            "may be misaligned, parser is tuned for v33.",
            stacklevel=2,
        )

    sections = _parse_sections(raw_lines[3:])
    return _build_case(sections, base_mva=sbase)
