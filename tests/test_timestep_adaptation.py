import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fipy import CellVariable, Grid1D
from fipyrite.diff_lib import data_container
from fipyrite.solver_calls import run_non_steady_state_solver_coupled

class MockMP:
    def __init__(self):
        self.plot_name = "test_run"
        self.max_steps = 3
        self.start_time = 0.0
        self.t_end = 100.0
        self.dt_min = 1.0
        self.dt_init = 5.0
        self.dt_max = 50.0
        self.backup_step = 100
        self.report_step = 100
        self.isotopes = False
        self.title = "test"
        self.enable_rate_adaptation = True
        self.monitored_rate_species = ["FeS", "TS2"]
        self.rate_threshold = 1e-8
        self.phi = 0.8
        self.solver_backend = "default"
        self.tolerance = 1e-12
        self.dt_tolerance = -1.0
        self.adaptive_solver_tolerance = False
        self.rate_adaptation_start_step = 2
        self.enable_rate_magnitude_check = True

class MockD:
    def __init__(self):
        self.FeS = 0.0
        self.TS2 = 0.0
        self.D_bio = 0.0
        self.D_irr = 0.0

@pytest.fixture
def solver_inputs():
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

@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_rate_adaptation_success(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that a step with valid rate changes is accepted and dt grows."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    
    # Mock setup static equations
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    # Configure mock update rates:
    # 2 calls per step (pre-sweep, post-sweep) for 3 steps
    rates_step1 = {"FeS": np.array([1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    rates_step2 = {"FeS": np.array([2e-7, 2e-7, 2e-7]), "TS2": np.array([2e-7, 2e-7, 2e-7])}
    mock_update_coeffs.side_effect = [
        rates_step1, rates_step1,  # Step 1
        rates_step2, rates_step2,  # Step 2
        rates_step2, rates_step2   # Step 3
    ]
    
    # Call the solver
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    # The solver should successfully run up to max_steps (3 steps here)
    assert step == 3

@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_rate_adaptation_sign_change_rollback(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that consecutive sign changes in rates (oscillation) triggers rollback and dt reduction."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    mp.max_steps = 4
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    # RATES sequence (consecutive sign flips required for rejection):
    # 1. Step 1 -> positive rate (2 calls) -> accepted
    # 2. Step 2 -> negative rate (2 calls) -> first flip -> accepted
    # 3. Step 3 tentative -> positive rate (2 calls) -> second consecutive flip -> rejected & rollback
    # 4. Step 3 retry -> negative rate (2 calls) -> accepted
    # 5. Step 4 -> negative rate (2 calls) -> accepted
    rates_pos = {"FeS": np.array([1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    rates_neg = {"FeS": np.array([-1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    
    mock_update_coeffs.side_effect = [
        rates_pos, rates_pos,          # Step 1
        rates_neg, rates_neg,          # Step 2
        rates_pos, rates_pos,          # Step 3 tentative -> rejected (consecutive flip)
        rates_neg, rates_neg,          # Step 3 retry -> accepted
        rates_neg, rates_neg           # Step 4
    ]
    
    # Track dt values to see if it cut
    dts = []
    def sweep_side_effect(dt, solver):
        dts.append(dt)
        return 0.0
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    # Verify rejection and rollback happened at Step 3
    # dts should contain:
    # 1. Step 1: 5.0 (dt_init)
    # 2. Step 2: 6.0 (5.0 * 1.2 growth)
    # 3. Step 3 tentative: 7.2 (6.0 * 1.2 growth)
    # 4. Step 3 retry: 3.6 (7.2 * 0.5 cut)
    # 5. Step 4: 4.32 (3.6 * 1.2 growth)
    assert len(dts) >= 5
    assert dts[3] == pytest.approx(3.6)

@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_rate_adaptation_magnitude_rollback(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that a >1-order-of-magnitude change in rates triggers rollback and dt reduction."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    # RATES sequence:
    # 1. Step 1 -> accepted (2 calls)
    # 2. Step 2 tentative -> >10x increase -> rejected (2 calls)
    # 3. Step 2 retry -> accepted (2 calls)
    # 4. Step 3 -> accepted (2 calls)
    rates_init = {"FeS": np.array([1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    rates_huge_jump = {"FeS": np.array([2e-6, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    rates_retry = {"FeS": np.array([1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    
    mock_update_coeffs.side_effect = [
        rates_init, rates_init,
        rates_huge_jump, rates_huge_jump,
        rates_retry, rates_retry,
        rates_retry, rates_retry
    ]
    
    # Track dt values to see if it cut
    dts = []
    def sweep_side_effect(dt, solver):
        dts.append(dt)
        return 0.0
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    assert len(dts) >= 4
    assert dts[2] == pytest.approx(3.0)

@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_rate_adaptation_noise_ignored(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that rate changes below the noise threshold are ignored."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    rates_init = {"FeS": np.array([1e-9, 1e-9, 1e-9]), "TS2": np.array([1e-9, 1e-9, 1e-9])}
    rates_noise_oscillating = {"FeS": np.array([-1e-9, 1e-9, 1e-9]), "TS2": np.array([1e-9, 1e-9, 1e-9])}
    
    mock_update_coeffs.side_effect = [
        rates_init, rates_init,
        rates_noise_oscillating, rates_noise_oscillating,
        rates_noise_oscillating, rates_noise_oscillating
    ]
    
    dts = []
    def sweep_side_effect(dt, solver):
        dts.append(dt)
        return 0.0
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    assert step == 3
    assert len(dts) == 3
    assert dts[1] == pytest.approx(6.0)
    assert dts[2] == pytest.approx(7.2)

@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_isotope_dt_limiter(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that the dynamic isotope dt limiter caps dt once species concentration exceeds threshold."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    
    mp.isotopes = True
    mp.enable_isotope_dt_limiter = True
    mp.isotope_limiter_species = "FeS"
    mp.isotope_onset_threshold = 1e-5
    mp.reaction_zone_spacing = 0.0001
    mp.w = 1e-5
    mp.dt_max_isotope = 4.0  # Explicitly set limit for test
    
    c["FeS"].setValue(0.0)
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    rates = {"FeS": np.array([1e-8, 1e-8, 1e-8]), "TS2": np.array([1e-8, 1e-8, 1e-8])}
    mock_update_coeffs.return_value = rates
    
    dts = []
    def sweep_side_effect(dt, solver):
        dts.append(dt)
        if len(dts) == 2:
            c["FeS"].setValue(2e-5)
        return 0.0
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    # Step 1: FeS = 0.0 -> next dt grows to 6.0 s
    # Step 2: FeS = 2e-5 -> onset detected! next dt grows to 7.2 s but capped to dt_max_isotope = 4.0 s
    # Step 3: dt = 4.0 s
    assert len(dts) == 3
    assert dts[0] == pytest.approx(5.0)
    assert dts[1] == pytest.approx(6.0)
    assert dts[2] == pytest.approx(4.0)


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_rate_sign_min_change_filtered(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that sign flips with rate difference smaller than rate_sign_min_change are ignored."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    mp.max_steps = 4
    mp.rate_threshold = 1e-12
    mp.rate_sign_min_change = 2e-8  # require >= 2e-8 change in rate
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    # Rates flip between +5e-9 and -5e-9 (rate change is 1e-8 < 2e-8)
    rates_pos = {"FeS": np.array([5e-9, 5e-9, 5e-9]), "TS2": np.array([5e-9, 5e-9, 5e-9])}
    rates_neg = {"FeS": np.array([-5e-9, 5e-9, 5e-9]), "TS2": np.array([5e-9, 5e-9, 5e-9])}
    
    mock_update_coeffs.side_effect = [
        rates_pos, rates_pos,          # Step 1
        rates_neg, rates_neg,          # Step 2
        rates_pos, rates_pos,          # Step 3 tentative -> rate change is only 1e-8 (< 2e-8) -> accepted without rollback!
        rates_neg, rates_neg           # Step 4
    ]
    
    dts = []
    def sweep_side_effect(dt, solver):
        dts.append(dt)
        return 0.0
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    # Step 3 should NOT have been rejected because rate change is below 2e-8
    assert step == 4
    assert len(dts) == 4
    assert dts[2] == pytest.approx(7.2)  # continues growth uninterrupted


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_rate_sign_min_consecutive_cells_filtered(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq, solver_inputs):
    """Test that sign flips affecting fewer than rate_sign_min_consecutive_cells are ignored."""
    mp, c, k, mesh, D_mol, bc_map, z = solver_inputs
    mp.max_steps = 4
    mp.rate_threshold = 1e-12
    mp.rate_sign_min_change = 2e-8
    mp.rate_sign_min_consecutive_cells = 2  # require >= 2 consecutive cells
    
    mock_coupled_eq = MagicMock()
    mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
    
    # Only cell 0 flips (1 cell < 2 cells)
    rates_pos = {"FeS": np.array([1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    rates_neg = {"FeS": np.array([-1e-7, 1e-7, 1e-7]), "TS2": np.array([1e-7, 1e-7, 1e-7])}
    
    mock_update_coeffs.side_effect = [
        rates_pos, rates_pos,          # Step 1
        rates_neg, rates_neg,          # Step 2
        rates_pos, rates_pos,          # Step 3 tentative -> only 1 cell oscillating (< 2) -> accepted without rollback!
        rates_neg, rates_neg           # Step 4
    ]
    
    dts = []
    def sweep_side_effect(dt, solver):
        dts.append(dt)
        return 0.0
    mock_coupled_eq.sweep.side_effect = sweep_side_effect
    
    step, rms = run_non_steady_state_solver_coupled(
        mp, c, ["FeS", "TS2"], ["FeS", "TS2"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
    )
    
    # Step 3 should NOT have been rejected because only 1 cell flipped, but 2 were required
    assert step == 4
    assert len(dts) == 4
    assert dts[2] == pytest.approx(7.2)


def test_adaptive_dt_failure_ceiling():
    """Test that AdaptiveDT correctly caps dt at the failure ceiling and holds it."""
    from fipyrite.solver_calls import AdaptiveDT
    
    controller = AdaptiveDT(
        dt_min=1.0,
        dt_max=100.0,
        dt_initial=30.0,
        enable_failure_ceiling=True,
        failure_ceiling_factor=0.7,
        failure_hold_steps=3,
        ceiling_growth_factor=1.1,
    )
    
    assert controller._dt == 30.0
    
    # Simulate failure at dt = 35.0
    cut_dt = controller.register_failure(failed_dt=35.0)
    assert cut_dt == pytest.approx(17.5)
    assert controller._dt_ceiling == pytest.approx(24.5)  # 35.0 * 0.7
    assert controller._steps_at_ceiling == 0
    
    # During recovery ramp (_dt < 24.5): steps_at_ceiling remains 0
    controller._dt = 10.0
    eff_max = controller.get_effective_max()
    assert eff_max == pytest.approx(24.5)
    assert controller._steps_at_ceiling == 0
    
    # Arrived at ceiling: _dt = 24.5
    controller._dt = 24.5
    # Step 1 at ceiling (steps_at_ceiling 1 < 3): ceiling remains 24.5
    eff_max_1 = controller.get_effective_max()
    assert eff_max_1 == pytest.approx(24.5)
    assert controller._steps_at_ceiling == 1
    
    # Step 2 at ceiling (steps_at_ceiling 2 < 3): ceiling remains 24.5
    eff_max_2 = controller.get_effective_max()
    assert eff_max_2 == pytest.approx(24.5)
    assert controller._steps_at_ceiling == 2
    
    # Step 3 at ceiling (steps_at_ceiling 3 >= 3): hold period finishes, ceiling raises!
    eff_max_3 = controller.get_effective_max()
    assert eff_max_3 == pytest.approx(24.5 * 1.1)  # 26.95
    assert controller._dt_ceiling == pytest.approx(26.95)
    assert controller._steps_at_ceiling == 0  # Counter reset for the new ceiling


def test_porewater_depletion_governor():
    from fipyrite.solver_calls import apply_porewater_depletion_governor

    mesh = Grid1D(nx=5)
    # TS2 (dissolved): drops from 2.0 to 1.0 (50% drop at cell 2)
    var_ts2 = CellVariable(mesh=mesh, value=2.0, hasOld=True)
    var_ts2.old.value = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    var_ts2.value = np.array([2.0, 2.0, 1.0, 2.0, 2.0])  # 50% change at cell 2

    # FeS (solid): drops from 100.0 to 10.0 (90% drop)
    var_fes = CellVariable(mesh=mesh, value=100.0, hasOld=True)
    var_fes.old.value = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    var_fes.value = np.array([100.0, 100.0, 10.0, 100.0, 100.0])

    species_struct = [
        {"name": "TS2", "var": var_ts2},
        {"name": "FeS", "var": var_fes},
    ]
    bc_map = {
        "TS2": {"type": "dissolved", "top": 0.0},
        "FeS": {"type": "solid", "top": 0.0},
    }

    # Case 1: Over-depletion (50% change vs 25% target)
    adapted_dt, obs_rel, lim_sp, lim_cell = apply_porewater_depletion_governor(
        species_struct=species_struct,
        bc_map=bc_map,
        current_dt=10.0,
        proposed_dt=12.0,
        max_rel_change=0.25,
        conc_scale=1e-4,
    )
    # Only TS2 (dissolved) is monitored, not FeS (solid)
    assert lim_sp == "TS2"
    assert lim_cell == 2
    assert obs_rel == pytest.approx(1.0 / (1.0 + 1e-4), rel=1e-3)
    # Factor is (0.25 / ~0.50)^0.5 ~ 0.707 -> adapted_dt ~ 7.07 < 10.0
    assert adapted_dt < 10.0
    assert adapted_dt == pytest.approx(10.0 * (0.25 / obs_rel) ** 0.5, rel=1e-3)

    # Case 2: Smooth change (TS2 only changes by 5%)
    var_ts2.value = np.array([2.0, 2.0, 1.9, 2.0, 2.0])  # 5% change
    adapted_dt, obs_rel, lim_sp, lim_cell = apply_porewater_depletion_governor(
        species_struct=species_struct,
        bc_map=bc_map,
        current_dt=10.0,
        proposed_dt=12.0,
        max_rel_change=0.25,
        conc_scale=1e-4,
    )
    assert obs_rel < 0.10
    # Allows growth up to proposed_dt
    assert adapted_dt >= 12.0

    # Case 3: Substance with sub-floor residual solver noise (e.g. O2 in anoxic zone)
    # Even if noise relative change is large (e.g. from 1e-4 to 2e-4), it must be ignored
    var_o2 = CellVariable(mesh=mesh, value=1e-4, hasOld=True)
    var_o2.old.value = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    var_o2.value = np.array([2e-4, 1e-4, 1e-4, 1e-4, 1e-4])  # 50% relative to denom
    species_struct_with_noise = [
        {"name": "TS2", "var": var_ts2},  # 5% change
        {"name": "O2", "var": var_o2},    # noise at 1e-4 mmol/L (< conc_presence_floor = 1e-3)
    ]
    bc_map_with_o2 = {
        "TS2": {"type": "dissolved", "top": 0.0},
        "O2": {"type": "dissolved", "top": 0.0},
    }
    adapted_dt, obs_rel, lim_sp, lim_cell = apply_porewater_depletion_governor(
        species_struct=species_struct_with_noise,
        bc_map=bc_map_with_o2,
        current_dt=10.0,
        proposed_dt=12.0,
        max_rel_change=0.25,
        conc_scale=1e-4,
        conc_presence_floor=1e-3,
    )
    # O2 noise must be filtered out; TS2 remains the limiting species
    assert lim_sp == "TS2"
    assert obs_rel < 0.10
    assert adapted_dt >= 12.0




