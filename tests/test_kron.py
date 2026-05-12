"""Unit tests for the Kron-reduction core."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from simplenet.kron import kron_reduce


def test_kron_matches_explicit_formula() -> None:
    """``kron_reduce`` matches Y_ii - Y_ie Y_ee^-1 Y_ei explicitly."""

    rng = np.random.default_rng(42)
    n = 6
    a = rng.standard_normal((n, n))
    y = a @ a.T + 5.0 * np.eye(n)

    sparse_y = sp.csc_matrix(y)
    ext = np.array([0, 3])
    intl = np.array([1, 2, 4, 5])

    result = kron_reduce(sparse_y, ext, intl)

    expected = (
        y[np.ix_(intl, intl)]
        - y[np.ix_(intl, ext)] @ np.linalg.solve(y[np.ix_(ext, ext)], y[np.ix_(ext, intl)])
    )

    np.testing.assert_allclose(result.y_red, expected, atol=1e-10)
    np.testing.assert_allclose(result.y_ii_orig, y[np.ix_(intl, intl)])


def test_kron_with_no_externals_is_identity() -> None:
    """When no buses are excluded the Kron result equals the input."""

    n = 4
    y = np.diag([3.0, 4.0, 5.0, 6.0]) - 1.0
    sp_y = sp.csc_matrix(y)
    result = kron_reduce(sp_y, np.zeros(0, dtype=np.int64), np.arange(n))

    np.testing.assert_allclose(result.y_red, y)


def test_kron_star_to_mesh() -> None:
    """Star-to-mesh (Y-Delta): eliminating a single hub bus connected
    to N spokes with reactances ``x_i`` produces an N-clique where
    ``Y_red[i, j] = -1 / (x_i * x_j * sum(1/x_k))``."""

    n = 4
    x = np.array([1.0, 2.0, 0.5])
    b = 1.0 / x
    y = np.zeros((n, n))
    y[0, 0] = b.sum()
    for k in range(3):
        y[k + 1, k + 1] = b[k]
        y[0, k + 1] = -b[k]
        y[k + 1, 0] = -b[k]

    result = kron_reduce(sp.csc_matrix(y), np.array([0]), np.array([1, 2, 3]))

    sum_b = b.sum()
    for i in range(3):
        for j in range(i + 1, 3):
            expected_off = -1.0 / (x[i] * x[j] * sum_b)
            np.testing.assert_allclose(result.y_red[i, j], expected_off, atol=1e-10)
            np.testing.assert_allclose(result.y_red[j, i], expected_off, atol=1e-10)
