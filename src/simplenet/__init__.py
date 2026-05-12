"""simplenet - DC modified-Ward network reduction in Python.

Quickstart
----------

>>> from simplenet import PowerCase, reduce_network
>>> from simplenet.io import load_m, load_xlsx, dump_xlsx, load_excluded_nodes
>>> case = load_m("matlab/NetworkReduction2/test_9bus_case.m")
>>> result = reduce_network(case, [1, 5, 8])
>>> print(result.summary)
>>> dump_xlsx(result.reduced_case, "case9_reduced.xlsx", summary=result.summary)
"""

from simplenet.case import PowerCase
from simplenet.dcpf import DCPFResult, run_dcpf
from simplenet.generators import GenMoveResult, move_external_generators
from simplenet.kron import KronResult, kron_reduce
from simplenet.pipeline import ReductionResult, reduce_network
from simplenet.preprocess import PreprocessStats, preprocess
from simplenet.redistribute import redistribute_loads
from simplenet.ymatrix import build_b_for_dcpf, build_b_for_reduction

__all__ = [
    "DCPFResult",
    "GenMoveResult",
    "KronResult",
    "PowerCase",
    "PreprocessStats",
    "ReductionResult",
    "build_b_for_dcpf",
    "build_b_for_reduction",
    "kron_reduce",
    "move_external_generators",
    "preprocess",
    "redistribute_loads",
    "reduce_network",
    "run_dcpf",
]

__version__ = "0.1.1a0"
