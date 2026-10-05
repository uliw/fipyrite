"""
Unit tests for Numba-accelerated solvers in fipyrite.solvers_numba.
"""

import numpy as np
import pytest
from scipy.linalg import solve_banded

from fipyrite.solvers_numba import solve_tridiagonal_thomas


def test_solve_tridiagonal_thomas_accuracy():
    """Verify solve_tridiagonal_thomas produces identical results to scipy.linalg.solve_banded."""
    rng = np.random.default_rng(42)

    for N in [3, 10, 100, 1108]:
        ab = np.zeros((3, N), dtype=np.float64)
        # Off-diagonals (diffusion/advection)
        ab[0, 1:] = -rng.uniform(0.1, 2.0, N - 1)
        ab[2, :-1] = -rng.uniform(0.1, 2.0, N - 1)
        # Diagonally dominant main diagonal
        diag_base = np.zeros(N)
        diag_base[:-1] += np.abs(ab[0, 1:])
        diag_base[1:] += np.abs(ab[2, :-1])
        ab[1, :] = diag_base + rng.uniform(0.5, 5.0, N)

        b = rng.uniform(0.1, 10.0, N)

        out = np.empty(N, dtype=np.float64)
        c_prime = np.empty(N - 1, dtype=np.float64)

        solve_tridiagonal_thomas(ab, b, out, c_prime)
        expected = solve_banded((1, 1), ab, b)

        max_abs_err = np.max(np.abs(out - expected))
        max_rel_err = np.max(np.abs(out - expected) / np.abs(expected))

        assert max_abs_err < 1e-12, f"Failed for N={N}: max_abs_err={max_abs_err}"
        assert max_rel_err < 1e-12, f"Failed for N={N}: max_rel_err={max_rel_err}"


def test_solve_tridiagonal_thomas_reproducibility():
    """Verify calling solve_tridiagonal_thomas repeatedly with reused buffers is deterministic."""
    N = 100
    rng = np.random.default_rng(123)

    ab = np.zeros((3, N), dtype=np.float64)
    ab[0, 1:] = -1.0
    ab[2, :-1] = -1.0
    ab[1, :] = 3.0
    b = rng.uniform(1.0, 5.0, N)

    out1 = np.empty(N, dtype=np.float64)
    out2 = np.empty(N, dtype=np.float64)
    c_prime = np.empty(N - 1, dtype=np.float64)

    solve_tridiagonal_thomas(ab, b, out1, c_prime)
    solve_tridiagonal_thomas(ab, b, out2, c_prime)

    np.testing.assert_array_equal(out1, out2)


def test_compute_species_residual_wrms_accuracy():
    """Verify compute_species_residual_wrms produces identical result to NumPy reference."""
    from fipyrite.solvers_numba import compute_species_residual_wrms, compute_species_residual_linf

    rng = np.random.default_rng(99)
    N = 500
    curr = rng.uniform(0.1, 10.0, N)
    prev = curr + rng.normal(0, 0.05, N)
    inner_tol = 1e-4
    atol = 1e-6

    # Reference WRMS
    diff = np.abs(curr - prev)
    scale = inner_tol * np.abs(curr) + atol
    ratio = diff / scale
    expected_wrms = float(np.sqrt(np.mean(ratio**2)))
    expected_linf = float(np.max(ratio))

    val_wrms = compute_species_residual_wrms(curr, prev, inner_tol, atol)
    val_linf = compute_species_residual_linf(curr, prev, inner_tol, atol)

    assert abs(val_wrms - expected_wrms) < 1e-12
    assert abs(val_linf - expected_linf) < 1e-12


def test_solve_block_tridiagonal_thomas_accuracy():
    """Verify solve_block_tridiagonal_thomas matches dense scipy.linalg.solve to < 1e-12."""
    from scipy.linalg import solve
    from fipyrite.solvers_numba import solve_block_tridiagonal_thomas

    rng = np.random.default_rng(42)

    for N, S in [(3, 2), (5, 4), (50, 5), (100, 10)]:
        A = -rng.uniform(0.1, 1.0, (N, S))
        C = -rng.uniform(0.1, 1.0, (N, S))
        B = rng.uniform(-0.2, 0.2, (N, S, S))
        for i in range(N):
            for r in range(S):
                B[i, r, r] += 4.0 + abs(A[i, r]) + abs(C[i, r])
        D = rng.uniform(1.0, 5.0, (N, S))

        out = np.zeros((N, S), dtype=np.float64)
        C_prime = np.zeros((N, S, S), dtype=np.float64)
        D_prime = np.zeros((N, S), dtype=np.float64)

        solve_block_tridiagonal_thomas(A, B, C, D, out, C_prime, D_prime)

        # Assemble full (N*S, N*S) dense matrix for comparison
        K = np.zeros((N * S, N * S), dtype=np.float64)
        RHS_full = np.zeros(N * S, dtype=np.float64)
        for i in range(N):
            K[i * S : (i + 1) * S, i * S : (i + 1) * S] = B[i]
            if i > 0:
                for r in range(S):
                    K[i * S + r, (i - 1) * S + r] = A[i, r]
            if i < N - 1:
                for r in range(S):
                    K[i * S + r, (i + 1) * S + r] = C[i, r]
            RHS_full[i * S : (i + 1) * S] = D[i]

        expected = solve(K, RHS_full).reshape((N, S))
        max_err = np.max(np.abs(out - expected))
        assert max_err < 1e-11, f"Failed for N={N}, S={S}: max_err={max_err}"
