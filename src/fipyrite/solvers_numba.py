"""
Numba-accelerated numerical kernels for direct assembled reactive transport solvers.
"""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def solve_tridiagonal_thomas(
    ab: np.ndarray,
    b: np.ndarray,
    out: np.ndarray,
    c_prime: np.ndarray,
) -> None:
    """
    O(N) In-place Thomas algorithm (TDMA) for tridiagonal systems in SciPy banded (1, 1) layout.

    Matrix layout in ab (shape 3, N):
      - row 0: superdiagonal (ab[0, 1:] contains A[i, i+1] for i = 0..N-2)
      - row 1: main diagonal  (ab[1, :] contains A[i, i] for i = 0..N-1)
      - row 2: subdiagonal   (ab[2, :-1] contains A[i+1, i] for i = 0..N-2)

    b:       RHS vector of shape (N,)
    out:     solution vector of shape (N,), updated in-place
    c_prime: work scratch array of shape (N - 1,)
    """
    n = len(b)
    denom0 = ab[1, 0]
    c_prime[0] = ab[0, 1] / denom0
    out[0] = b[0] / denom0

    # Forward elimination
    for i in range(1, n - 1):
        ai = ab[2, i - 1]
        denom = ab[1, i] - ai * c_prime[i - 1]
        c_prime[i] = ab[0, i + 1] / denom
        out[i] = (b[i] - ai * out[i - 1]) / denom

    ai_last = ab[2, n - 2]
    denom_last = ab[1, n - 1] - ai_last * c_prime[n - 2]
    out[n - 1] = (b[n - 1] - ai_last * out[n - 2]) / denom_last

    # Back substitution
    for i in range(n - 2, -1, -1):
        out[i] -= c_prime[i] * out[i + 1]


@nb.njit(fastmath=True)
def compute_species_residual_wrms(
    curr_val: np.ndarray,
    prev_val: np.ndarray,
    inner_tol: float,
    atol: float,
) -> float:
    """
    Computes WRMS norm for a single species:
        sqrt( (1/N) * sum( ( |c_curr - c_prev| / (inner_tol * |c_curr| + atol) )^2 ) )
    """
    n = len(curr_val)
    sum_sq = 0.0
    for i in range(n):
        c = curr_val[i]
        diff = abs(c - prev_val[i])
        scale = inner_tol * abs(c) + atol
        ratio = diff / scale
        if np.isnan(ratio) or np.isinf(ratio):
            return np.inf
        sum_sq += ratio * ratio
    return np.sqrt(sum_sq / n)


@nb.njit(fastmath=True)
def compute_species_residual_linf(
    curr_val: np.ndarray,
    prev_val: np.ndarray,
    inner_tol: float,
    atol: float,
) -> float:
    """
    Computes L-infinity norm for a single species:
        max( |c_curr - c_prev| / (inner_tol * |c_curr| + atol) )
    """
    n = len(curr_val)
    max_ratio = 0.0
    for i in range(n):
        c = curr_val[i]
        diff = abs(c - prev_val[i])
        scale = inner_tol * abs(c) + atol
        ratio = diff / scale
        if np.isnan(ratio) or np.isinf(ratio):
            return np.inf
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio

