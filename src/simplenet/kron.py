"""Kron reduction core.

Mathematically equivalent to the partial-LU implementation in
``matlab/NetworkReduction2/PartialSymLU.m`` and ``PartialNumLU.m``: given
a partition of buses into *external* (to be eliminated) and *internal*
(retained), the equivalent admittance matrix seen from the internal
buses is:

.. math::

    Y_{\\text{red}} = Y_{ii} - Y_{ie} \\, Y_{ee}^{-1} \\, Y_{ei}

This module returns the dense reduced internal block (typically far
smaller than the full system) so that downstream code can extract
equivalent branches and shunts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class KronResult:
    """Output of :func:`kron_reduce`.

    Attributes
    ----------
    y_red
        Dense reduced admittance matrix of shape
        ``(len(internal_idx), len(internal_idx))``.
    y_ii_orig
        Original internal block (dense), useful for distinguishing
        native off-diagonal entries from new fills (equivalent
        branches).
    """

    y_red: np.ndarray
    y_ii_orig: np.ndarray


def kron_reduce(
    y_full: sp.spmatrix,
    external_idx: np.ndarray,
    internal_idx: np.ndarray,
) -> KronResult:
    """Compute :math:`Y_{ii} - Y_{ie} Y_{ee}^{-1} Y_{ei}` densely.

    Parameters
    ----------
    y_full
        Square sparse matrix (any sparse format), typically built by
        :func:`simplenet.ymatrix.build_b_for_reduction`.
    external_idx
        0-indexed array of external bus rows/columns.
    internal_idx
        0-indexed array of internal bus rows/columns; together with
        ``external_idx`` it must partition ``range(y_full.shape[0])``.
    """

    if y_full.shape[0] != y_full.shape[1]:
        raise ValueError("y_full must be square")

    external_idx = np.asarray(external_idx, dtype=np.int64)
    internal_idx = np.asarray(internal_idx, dtype=np.int64)

    y_full_csc = y_full.tocsc()
    y_ii_sp = y_full_csc[internal_idx, :][:, internal_idx]
    y_ii_dense = y_ii_sp.toarray()

    if external_idx.size == 0:
        return KronResult(y_red=y_ii_dense.copy(), y_ii_orig=y_ii_dense)

    y_ee = y_full_csc[external_idx, :][:, external_idx]
    y_ei = y_full_csc[external_idx, :][:, internal_idx]
    y_ie = y_full_csc[internal_idx, :][:, external_idx]

    y_ei_dense = y_ei.toarray()
    x = spla.spsolve(y_ee.tocsc(), y_ei_dense)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    correction = y_ie @ x
    if sp.issparse(correction):
        correction = correction.toarray()
    y_red = y_ii_dense - correction
    return KronResult(y_red=y_red, y_ii_orig=y_ii_dense)
