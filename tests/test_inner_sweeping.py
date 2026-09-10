import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fipy import CellVariable, Grid1D
from fipyrite.diff_lib import data_container
from fipyrite.solver_calls import (
    run_non_steady_state_solver_coupled,
    _compute_inner_residual,
)


class MockMP:
    def __init__(self):
        self.plot_name = "test_inner_sweep_run"
        self.max_steps = 3
        self.start_time = 0.0
        self.t_end = 1000.0
        self.dt_min = 1.0
        self.dt_init = 10.0
        self.dt_max = 100.0
        self.backup_step = 100
        self.report_step = 100
        self.isotopes = False
        self.title = "test"
        self.enable_rate_adaptation = False
        self.phi = 0.8
        self.solver_backend = "default"
        self.tolerance = 1e-12
        self.dt_tolerance = -1.0
        self.adaptive_solver_tolerance = False
        
        # Inner sweeping settings
        self.enable_inner_sweeping = True
        self.max_inner_sweeps = 5
        self.inner_tol = 1e-3
        self.inner_relaxation = 1.0
        self.inner_sweep_equilibrium = False
        self.adaptive_sweeps_dt = True
        self.sweep_target_optimal = 3
        self.sweep_max_acceptable = 4


class MockD:
    def __init__(self):
        self.FeS = 0.0
        self.TS2 = 0.0
        self.D_bio = 0.0
        self.D_irr = 0.0


@pytest.fixture
def solver_setup():
    mesh = Grid1D(nx=3)
    c = data_container({
        "FeS": CellVariable(mesh=mesh, value=1.0, hasOld=True),
        "TS2": CellVariable(mesh=mesh, value=1.0, hasOld=True),
    })
    mp = MockMP()
    k = data_container()
    D_mol = MockD()
    bc_map = {
        "FeS": {"type": "solid", "top": 0.0},
        "TS2": {"type": "dissolved", "top": 0.0}
    }
    z = np.array([0.0, 1.0, 2.0, 3.0])
    return mp, c, k, mesh, D_mol, bc_map, z


def test_compute_inner_residual():
    mesh = Grid1D(nx=3)
    var = CellVariable(mesh=mesh, value=1.0)
    species_struct = [{"name": "A", "var": var}]
    
    # Identical values: error should be 0.0
    prev_iterate = {"A": np.array([1.0, 1.0, 1.0])}
    err = _compute_inner_residual(species_struct, prev_iterate, inner_tol=1e-4)
    assert err == 0.0
    
    # Change is below atol (1e-6) + rtol (1e-4) -> err <= 1.0
    prev_iterate = {"A": np.array([1.0 - 1e-5, 1.0, 1.0])}
    err = _compute_inner_residual(species_struct, prev_iterate, inner_tol=1e-4, atol_default=1e-4)
    assert err <= 1.0
    
    # Large change: diff = 0.5, scale = 1e-4 * 1.0 + 1e-6 -> err >> 1.0
    prev_iterate = {"A": np.array([0.5, 1.0, 1.0])}
    err = _compute_inner_residual(species_struct, prev_iterate, inner_tol=1e-4)
    assert err > 1.0

    # WRMS vs Linf comparison: single-cell outlier
    # In Linf, the single outlier dictates the norm. In WRMS, it is averaged across cells.
    prev_iterate_outlier = {"A": np.array([0.5, 1.0, 1.0])}
    err_linf = _compute_inner_residual(species_struct, prev_iterate_outlier, inner_tol=1e-4, inner_norm="linf")
    err_wrms = _compute_inner_residual(species_struct, prev_iterate_outlier, inner_tol=1e-4, inner_norm="wrms")
    assert err_linf > err_wrms
    assert err_wrms == pytest.approx(err_linf / np.sqrt(3.0), rel=1e-3)

    # NaN check: should safely return inf
    prev_iterate = {"A": np.array([np.nan, 1.0, 1.0])}
    err = _compute_inner_residual(species_struct, prev_iterate, inner_tol=1e-4)
    assert np.isinf(err)


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_inner_sweeping_convergence(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that inner sweeping iterates until residual is met and advances time."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 5
    mp.inner_tol = 1e-3
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}
    
    sweep_counts = []
    def sweep_side_effect(dt, solver):
        sweep_counts.append(1)
        if len(sweep_counts) == 1:
            c["TS2"].setValue(1.1)
        elif len(sweep_counts) == 2:
            c["TS2"].setValue(1.105)
        else:
            c["TS2"].setValue(1.1050001)
        return 0.0
    
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    assert step == 1
    assert len(sweep_counts) == 3  # Converged in 3 sweeps!


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_inner_sweeping_failure_and_rollback(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that failure to converge within max_inner_sweeps triggers rollback and cuts dt."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 3
    mp.inner_tol = 1e-6
    mp.dt_init = 20.0
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}
    
    attempt = {"count": 0}
    sweep_dts = []
    
    def sweep_side_effect(dt, solver):
        sweep_dts.append(dt)
        if dt > 15.0:
            attempt["count"] += 1
            c["TS2"].setValue(attempt["count"] * 2.0)
        else:
            c["TS2"].setValue(1.0)
        return 0.0
    
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    assert step == 1
    assert sweep_dts[0] == 20.0
    assert sweep_dts[1] == 20.0
    assert sweep_dts[2] == 20.0
    assert sweep_dts[3] < 20.0


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_inner_sweeping_dt_growth(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that when steps converge in <= sweep_target_optimal sweeps, dt grows."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 3
    mp.dt_init = 10.0
    mp.sweep_target_optimal = 3
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}
    
    dts_at_step_start = []
    def sweep_side_effect(dt, solver):
        dts_at_step_start.append(dt)
        c["TS2"].setValue(c["TS2"].old.value)
        return 0.0
    
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    assert step == 3
    assert dts_at_step_start[0] == pytest.approx(10.0)
    assert dts_at_step_start[1] == pytest.approx(12.0)
    assert dts_at_step_start[2] == pytest.approx(14.4)


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_inner_sweeping_adaptive_damping(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that adaptive damping engages when residual increases and drives convergence."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 8
    mp.inner_tol = 1e-3
    mp.enable_adaptive_damping = True
    mp.inner_relaxation = 1.0

    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}

    sweeps = []
    def sweep_side_effect(dt, solver):
        sweeps.append(len(sweeps) + 1)
        # Sweep 1: small change
        if len(sweeps) == 1:
            c["TS2"].setValue(1.05)
        # Sweep 2: sudden overshoot -> large change (residual grows)
        elif len(sweeps) == 2:
            c["TS2"].setValue(1.8)
        # Sweep 3+: stabilizes as damping factor reduces
        elif len(sweeps) == 3:
            c["TS2"].setValue(1.3)
        else:
            c["TS2"].setValue(1.3000001)
        return 0.0

    mock_coupled_eq.sweep.side_effect = sweep_side_effect

    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )

    assert step == 1
    assert len(sweeps) <= 6


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_legacy_mode_when_inner_sweeping_disabled(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that setting enable_inner_sweeping=False strictly preserves legacy single-sweep behavior."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 3
    mp.dt_init = 10.0
    mp.enable_inner_sweeping = False
    mp.enable_rate_adaptation = True

    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}

    sweep_call_count = 0
    def sweep_side_effect(dt, solver):
        nonlocal sweep_call_count
        sweep_call_count += 1
        # Set a different concentration; in legacy mode this must NOT trigger additional sweeps
        c["TS2"].setValue(c["TS2"].value + 0.5)
        return 0.0

    mock_coupled_eq.sweep.side_effect = sweep_side_effect

    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )

    assert step == 3
    # Exactly one sweep per step (total 3 sweeps across 3 steps)
    assert sweep_call_count == 3


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_early_bailout_stagnant_residual(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that inner sweeping bails out at early_bailout_iter (5) instead of burning all max_inner_sweeps (15)."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 15
    mp.inner_tol = 1e-3
    mp.dt_init = 20.0
    mp.enable_early_bailout = True
    mp.early_bailout_iter = 5
    mp.early_bailout_err = 3.0

    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}

    attempt = 0
    sweep_count_attempt1 = 0
    sweep_dts = []

    def sweep_side_effect(dt, solver):
        nonlocal attempt, sweep_count_attempt1
        sweep_dts.append(dt)
        if dt > 10.0:
            # Attempt 1: Keep residual stagnant and high (~5.0)
            sweep_count_attempt1 += 1
            # Change by 0.005 each sweep -> scaled error ~ 5.0 / 1e-3
            c["TS2"].setValue(c["TS2"].value + 0.005)
        else:
            # Attempt 2: Converge immediately
            c["TS2"].setValue(c["TS2"].old.value)
        return 0.0

    mock_coupled_eq.sweep.side_effect = sweep_side_effect

    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )

    assert step == 1
    # Attempt 1 must have bailed out at sweep 5 instead of continuing to 15!
    assert sweep_count_attempt1 == 5
    # Retry dt should have been cut aggressively (20.0 * 0.25 = 5.0)
    assert sweep_dts[5] == pytest.approx(5.0)


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_early_bailout_diverging_residual(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that inner sweeping bails out early when residual grows/diverges."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 15
    mp.inner_tol = 1e-3
    mp.dt_init = 20.0
    mp.enable_early_bailout = True
    mp.early_bailout_iter = 5

    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}

    sweep_count_attempt1 = 0

    def sweep_side_effect(dt, solver):
        nonlocal sweep_count_attempt1
        if dt > 10.0:
            sweep_count_attempt1 += 1
            # Error grows exponentially: 0.002, 0.004, 0.008, 0.016, 0.032
            c["TS2"].setValue(c["TS2"].value + 0.002 * (2 ** sweep_count_attempt1))
        else:
            c["TS2"].setValue(c["TS2"].old.value)
        return 0.0

    mock_coupled_eq.sweep.side_effect = sweep_side_effect

    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )

    assert step == 1
    # Bailed out at sweep 5 due to divergence
    assert sweep_count_attempt1 == 5


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_aggressive_vs_standard_cut(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that residual > 3.0 triggers 0.25x cut while residual < 3.0 uses standard 0.5x cut."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 3
    mp.dt_init = 20.0
    mp.enable_early_bailout = False  # disable early bailout so max_inner_sweeps dictates end
    mp.enable_aggressive_cut = True
    mp.aggressive_cut_threshold = 3.0
    mp.aggressive_cut_factor = 0.25
    mp.dt_cut_factor = 0.5
    mp.inner_tol = 1e-3

    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}

    # Test 1: residual is small (~1.5) on failure -> standard cut 0.5x -> dt = 10.0
    sweep_dts = []
    def sweep_side_effect_mild(dt, solver):
        sweep_dts.append(dt)
        if dt > 15.0:
            # Change relative to previous iterate is 0.0015 -> scaled error ~1.5 (< 3.0)
            c["TS2"].setValue(c["TS2"].value + 0.0015)
        else:
            c["TS2"].setValue(c["TS2"].old.value)
        return 0.0

    mock_coupled_eq.sweep.side_effect = sweep_side_effect_mild

    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )

    assert step == 1
    # 3 sweeps at 20.0, then retry at 10.0 (0.5x)
    assert sweep_dts[0] == 20.0
    assert sweep_dts[1] == 20.0
    assert sweep_dts[2] == 20.0
    assert sweep_dts[3] == pytest.approx(10.0)


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_aggressive_cut_on_high_residual(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that residual > 3.0 triggers aggressive 0.25x cut."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 3
    mp.dt_init = 20.0
    mp.enable_early_bailout = False
    mp.enable_aggressive_cut = True
    mp.aggressive_cut_threshold = 3.0
    mp.aggressive_cut_factor = 0.25
    mp.dt_cut_factor = 0.5
    mp.inner_tol = 1e-3

    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}

    sweep_dts = []
    def sweep_side_effect_high(dt, solver):
        sweep_dts.append(dt)
        if dt > 10.0:
            # Change relative to previous iterate is 0.005 -> scaled error ~5.0 (> 3.0)
            c["TS2"].setValue(c["TS2"].value + 0.005)
        else:
            c["TS2"].setValue(c["TS2"].old.value)
        return 0.0

    mock_coupled_eq.sweep.side_effect = sweep_side_effect_high

    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )

    assert step == 1
    # 3 sweeps at 20.0, then aggressive retry at 5.0 (0.25x)
    assert sweep_dts[0] == 20.0
    assert sweep_dts[1] == 20.0
    assert sweep_dts[2] == 20.0
    assert sweep_dts[3] == pytest.approx(5.0)


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_near_convergence_acceptance(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that a step within near_convergence_tol (e.g. 1.25) at max_inner_sweeps is accepted."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 3
    mp.enable_early_bailout = False
    mp.inner_tol = 1e-3
    mp.near_convergence_tol = 1.25
    mp.dt_init = 10.0

    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}

    sweep_count = 0
    def sweep_near_conv(dt, solver):
        nonlocal sweep_count
        sweep_count += 1
        # Change by 1.1e-3 -> scaled error = 1.1e-3 / 1e-3 = 1.10 (between 1.00 and 1.25)
        c["TS2"].setValue(c["TS2"].value + 1.1e-3)
        return 0.0

    mock_coupled_eq.sweep.side_effect = sweep_near_conv

    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )

    # Must succeed in 1 step without retries/failures because near_convergence_tol accepted it!
    assert step == 1
    assert sweep_count == 3


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_graceful_acceptance(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that a step between near_convergence_tol and graceful_acceptance_tol is gracefully accepted."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 3
    mp.enable_early_bailout = False
    mp.inner_tol = 1e-3
    mp.near_convergence_tol = 1.25
    mp.graceful_acceptance_tol = 5.0
    mp.dt_init = 10.0

    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}

    sweep_count = 0
    def sweep_graceful(dt, solver):
        nonlocal sweep_count
        sweep_count += 1
        # Change by 2.5e-3 -> scaled error = 2.50 (between near_tol 1.25 and graceful_tol 5.0)
        c["TS2"].setValue(c["TS2"].value + 2.5e-3)
        return 0.0

    mock_coupled_eq.sweep.side_effect = sweep_graceful

    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )

    # Must succeed in 1 step without raising failure/retry!
    assert step == 1
    assert sweep_count == 3


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_single_sweep_early_exit(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_setup):
    """Test that when enable_single_sweep_exit=True and error <= sweep1_exit_tol, it exits on sweep 1."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_setup
    mp.max_steps = 1
    mp.max_inner_sweeps = 4
    mp.inner_tol = 1e-3
    mp.enable_single_sweep_exit = True
    mp.sweep1_exit_tol = 3.0
    mp.dt_init = 10.0

    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    mock_update_coeffs.return_value = {"FeS": np.zeros(3), "TS2": np.zeros(3)}

    sweep_count = 0
    def sweep_smooth(dt, solver):
        nonlocal sweep_count
        sweep_count += 1
        # Change by 2.0e-3 -> scaled error = 2.00 <= sweep1_exit_tol 3.0
        c["TS2"].setValue(c["TS2"].value + 2.0e-3)
        return 0.0

    mock_coupled_eq.sweep.side_effect = sweep_smooth

    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )

    # Must succeed in 1 step with exactly 1 sweep!
    assert step == 1
    assert sweep_count == 1


def test_mttf_growth_governor():
    """Test that MTTF governor throttles growth factor on rapid failures and restores it on stability."""
    from fipyrite.solver_calls import AdaptiveDT

    dt_ctrl = AdaptiveDT(
        dt_min=1.0,
        dt_max=100.0,
        dt_initial=10.0,
        growth_factor=1.20,
        cut_factor=0.5,
        enable_mttf_governor=True,
    )
    assert dt_ctrl.growth_factor == 1.20
    assert dt_ctrl.steps_since_failure == 0

    # Rapid failure (steps_since_failure = 2 < 5)
    dt_ctrl.record_success()
    dt_ctrl.record_success()
    assert dt_ctrl.steps_since_failure == 2

    # Register failure: excess growth should be reduced by mttf_x1 (60% cut on 0.20 -> 0.08)
    dt_ctrl.register_failure(10.0)
    assert dt_ctrl.steps_since_failure == 0
    assert dt_ctrl.growth_factor == pytest.approx(1.0 + 0.20 * 0.40, rel=1e-3)

    # 45 stable steps: should recover towards base growth factor (1.20)
    for _ in range(45):
        dt_ctrl.record_success()
    assert dt_ctrl.growth_factor > 1.10


