"""PowerCase data model and column index constants.

The constants follow MATPOWER's case format (version 2) - see
``case_ACTIVSg10kCopy2.m`` headers and ``test_9bus_case.m`` for the
canonical column orderings:

    bus:    bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
            (+ lam_P lam_Q mu_Vmax mu_Vmin if present)
    gen:    bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin Pc1 Pc2 Qc1min
            Qc1max Qc2min Qc2max ramp_agc ramp_10 ramp_30 ramp_q apf
            (+ mu_Pmax mu_Pmin mu_Qmax mu_Qmin if present)
    branch: fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax
            (+ Pf Qf Pt Qt mu_Sf mu_St mu_angmin mu_angmax if present)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt

BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
BUS_AREA = 6
VM = 7
VA = 8
BASE_KV = 9
ZONE = 10
VMAX = 11
VMIN = 12

GEN_BUS = 0
PG = 1
QG = 2
QMAX = 3
QMIN = 4
VG = 5
MBASE = 6
GEN_STATUS = 7
PMAX = 8
PMIN = 9
PC1 = 10
PC2 = 11
QC1MIN = 12
QC1MAX = 13
QC2MIN = 14
QC2MAX = 15
RAMP_AGC = 16
RAMP_10 = 17
RAMP_30 = 18
RAMP_Q = 19
APF = 20

F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
RATE_A = 5
RATE_B = 6
RATE_C = 7
TAP = 8
SHIFT = 9
BR_STATUS = 10
ANGMIN = 11
ANGMAX = 12
PF = 13
QF = 14
PT = 15
QT = 16

PQ_BUS = 1
PV_BUS = 2
REF_BUS = 3
ISOLATED_BUS = 4

BUS_COLUMNS = 13
GEN_COLUMNS = 21
BRANCH_COLUMNS = 13

BUS_FULL_HEADER = [
    "bus_i", "type", "Pd", "Qd", "Gs", "Bs", "area", "Vm", "Va",
    "baseKV", "zone", "Vmax", "Vmin", "lam_P", "lam_Q", "mu_Vmax", "mu_Vmin",
]
GEN_FULL_HEADER = [
    "Bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin",
    "Pc1", "Pc2", "Qc1min", "Qc1max", "Qc2min", "Qc2max", "ramp_agc", "ramp_10",
    "ramp_30", "ramp_q", "apf", "mu_Pmax", "mu_Pmin", "mu_Qmax", "mu_Qmin",
]
BRANCH_FULL_HEADER = [
    "fbus", "tbus", "r", "x", "b", "rateA", "rateB", "rateC", "ratio", "angle",
    "status", "angmin", "angmax", "Pf", "Qf", "Pt", "Qt", "mu_Sf", "mu_St",
    "mu_angmin", "mu_angmax",
]


def _to_2d_float(a: Any) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


@dataclass
class PowerCase:
    """In-memory MATPOWER case (version 2).

    Matrices use numpy float arrays, 0-indexed. ``base_mva`` matches
    MATPOWER's ``mpc.baseMVA``. Optional fields are ``None`` when absent.
    """

    base_mva: float = 100.0
    bus: npt.NDArray[np.floating] = field(default_factory=lambda: np.zeros((0, BUS_COLUMNS)))
    gen: npt.NDArray[np.floating] = field(default_factory=lambda: np.zeros((0, GEN_COLUMNS)))
    branch: npt.NDArray[np.floating] = field(default_factory=lambda: np.zeros((0, BRANCH_COLUMNS)))
    gencost: npt.NDArray[np.floating] | None = None
    gentype: list[str] | None = None
    genfuel: list[str] | None = None
    bus_name: list[str] | None = None
    dcline: npt.NDArray[np.floating] | None = None
    version: str = "2"

    def copy(self) -> PowerCase:
        """Return a deep copy that is safe to mutate."""

        return replace(
            self,
            bus=self.bus.copy(),
            gen=self.gen.copy(),
            branch=self.branch.copy(),
            gencost=None if self.gencost is None else self.gencost.copy(),
            gentype=None if self.gentype is None else list(self.gentype),
            genfuel=None if self.genfuel is None else list(self.genfuel),
            bus_name=None if self.bus_name is None else list(self.bus_name),
            dcline=None if self.dcline is None else self.dcline.copy(),
        )

    @classmethod
    def from_pypower(cls, mpc: dict[str, Any]) -> PowerCase:
        """Build a ``PowerCase`` from a pypower-style dict.

        Accepts either pypower's ``baseMVA`` (camelCase) or
        ``base_mva`` (snake_case). Missing optional fields are left
        unset.
        """

        base = mpc.get("baseMVA", mpc.get("base_mva", 100.0))
        bus = _to_2d_float(mpc["bus"])
        gen = _to_2d_float(mpc["gen"])
        branch = _to_2d_float(mpc["branch"])
        gencost = _to_2d_float(mpc["gencost"]) if "gencost" in mpc and mpc["gencost"] is not None else None
        dcline = _to_2d_float(mpc["dcline"]) if "dcline" in mpc and mpc["dcline"] is not None else None
        gentype = list(mpc["gentype"]) if "gentype" in mpc and mpc["gentype"] is not None else None
        genfuel = list(mpc["genfuel"]) if "genfuel" in mpc and mpc["genfuel"] is not None else None
        bus_name = list(mpc["bus_name"]) if "bus_name" in mpc and mpc["bus_name"] is not None else None
        return cls(
            base_mva=float(base),
            bus=bus,
            gen=gen,
            branch=branch,
            gencost=gencost,
            dcline=dcline,
            gentype=gentype,
            genfuel=genfuel,
            bus_name=bus_name,
            version=str(mpc.get("version", "2")),
        )

    def to_pypower(self) -> dict[str, Any]:
        """Export to a pypower-style dict."""

        out: dict[str, Any] = {
            "version": self.version,
            "baseMVA": float(self.base_mva),
            "bus": self.bus.copy(),
            "gen": self.gen.copy(),
            "branch": self.branch.copy(),
        }
        if self.gencost is not None:
            out["gencost"] = self.gencost.copy()
        if self.gentype is not None:
            out["gentype"] = list(self.gentype)
        if self.genfuel is not None:
            out["genfuel"] = list(self.genfuel)
        if self.bus_name is not None:
            out["bus_name"] = list(self.bus_name)
        if self.dcline is not None:
            out["dcline"] = self.dcline.copy()
        return out

    def n_bus(self) -> int:
        return int(self.bus.shape[0])

    def n_gen(self) -> int:
        return int(self.gen.shape[0])

    def n_branch(self) -> int:
        return int(self.branch.shape[0])


def pad_to_columns(arr: np.ndarray, ncols: int) -> np.ndarray:
    """Right-pad a 2-D numeric array with zero columns up to ``ncols``."""

    if arr.size == 0:
        return np.zeros((arr.shape[0], ncols))
    if arr.shape[1] >= ncols:
        return arr
    pad = np.zeros((arr.shape[0], ncols - arr.shape[1]))
    return np.hstack([arr, pad])
