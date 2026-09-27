import sys
from pathlib import Path
import numpy as np
import pytest
from fipyrite.diff_lib import VariableArray as CellVariable, Mesh1D as Grid1D

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "nbk" / "experiments"))

from fipyrite.diff_lib import data_container
import reactions_new as rn


class MockMP(data_container):
    def __init__(self):
        super().__init__()
        self.phi = 0.8
        self.Fe2_diss = 0.95
        self.hs_frac = 0.6
        self.h2s_frac = 0.4
        self.h2s_hs_alpha = 1.000
        self.VCDT = 0.044163
        self.isotopes = False
        self.in_solver = False
        self.fes_picard_weight_fe = "auto"
        self.fes_smooth_epsilon = 0.05
        self.current_dt = 3600.0


class MockK:
    def __init__(self):
        self.Hplus = 1e-7
        self.FeS_sp = 1e-3
        self.FeS_isp = 1e-3  # Precipitation rate constant
        self.FeS_isd = 1e-4  # Dissolution rate constant


def make_test_containers(mesh, fe2_val, ts2_val, fes_val, isotopes=False):
    c_dict = {
        "Fe2_total": CellVariable(mesh=mesh, value=fe2_val, hasOld=True),
        "TS2": CellVariable(mesh=mesh, value=ts2_val, hasOld=True),
        "FeS": CellVariable(mesh=mesh, value=fes_val, hasOld=True),
    }
    if isotopes:
        c_dict["TS2_32"] = CellVariable(mesh=mesh, value=ts2_val * 0.95, hasOld=True)
        c_dict["FeS_32"] = CellVariable(mesh=mesh, value=fes_val * 0.95, hasOld=True)
    c = data_container(c_dict)

    species_list = list(c_dict.keys())
    LHS = {s: np.zeros(mesh.numberOfCells) for s in species_list}
    RHS = {s: np.zeros(mesh.numberOfCells) for s in species_list}
    RATES = {s: np.zeros(mesh.numberOfCells) for s in species_list}
    CROSS = {s: [] for s in species_list}
    lim = {}
    return c, LHS, RHS, RATES, CROSS, lim


def test_smooth_transition_import_and_aliases():
    """Verify function exists and alias is defined."""
    assert hasattr(rn, "FeS_precipitation_dissolution_smooth_transition")
    assert hasattr(rn, "FeS_precipitation_dissolution_smooth")
    assert rn.FeS_precipitation_dissolution_smooth is rn.FeS_precipitation_dissolution_smooth_transition


def test_smooth_transition_matches_raw_kinetics_far_from_equilibrium():
    """Far from Omega=1 (|Omega-1| >= epsilon), smooth transition matches symmetrical Picard exactly."""
    mesh = Grid1D(nx=3)
    k = MockK()
    mp = MockMP()

    # Case 1: Well supersaturated (Omega >> 1 + epsilon)
    c1, LHS1, RHS1, RATES1, CROSS1, lim1 = make_test_containers(mesh, fe2_val=1e-2, ts2_val=1e-2, fes_val=1e-3)
    rn.FeS_precipitation_dissolution_smooth_transition(c1, k, lim1, LHS1, RHS1, RATES1, CROSS1, mp)

    c2, LHS2, RHS2, RATES2, CROSS2, lim2 = make_test_containers(mesh, fe2_val=1e-2, ts2_val=1e-2, fes_val=1e-3)
    rn.FeS_precipitation_dissolution_symmetrical_picard(c2, k, lim2, LHS2, RHS2, RATES2, CROSS2, mp)

    np.testing.assert_allclose(RATES1["FeS"], RATES2["FeS"], rtol=1e-5)
    np.testing.assert_allclose(LHS1["TS2"], LHS2["TS2"], rtol=1e-5)


def test_smooth_transition_rate_tapers_to_zero_at_equilibrium():
    """As Omega -> 1, rate smoothly approaches 0 with zero derivative jump."""
    mesh = Grid1D(nx=1)
    k = MockK()
    mp = MockMP()
    mp.fes_smooth_epsilon = 0.05

    # omega_den = 1e-7 * 1e-3 = 1e-10
    # Fe2_pw = Fe2 * 0.95, hs = TS2 * 0.6 => Fe2_pw * hs = Fe2 * TS2 * 0.57
    # For Omega = 1.0: Fe2 * TS2 = 1e-10 / 0.57 = 1.75438596e-10
    target_prod = 1e-10 / 0.57
    fe2_base = np.sqrt(target_prod)

    omegas = [1.0, 1.001, 1.01, 1.05, 1.1]
    rates = []
    for om in omegas:
        fe2 = fe2_base * np.sqrt(om)
        ts2 = fe2_base * np.sqrt(om)
        c, LHS, RHS, RATES, CROSS, lim = make_test_containers(mesh, fe2_val=fe2, ts2_val=ts2, fes_val=1e-3)
        rn.FeS_precipitation_dissolution_smooth_transition(c, k, lim, LHS, RHS, RATES, CROSS, mp)
        rates.append(RATES["FeS"][0])

    # Rate at exactly Omega=1 should be 0 (within machine precision)
    assert abs(rates[0]) < 1e-30
    # Rate should be monotonically increasing with Omega above 1
    assert rates[1] > rates[0]
    assert rates[2] > rates[1]
    assert rates[3] > rates[2]
    assert rates[4] > rates[3]


def test_smooth_transition_thermodynamic_bounds():
    """Verify zero dissolution when Omega >= 1, and zero precipitation when Omega < 1."""
    mesh = Grid1D(nx=1)
    k = MockK()
    mp = MockMP()

    # 1. Supersaturated: Omega > 1
    c1, LHS1, RHS1, RATES1, CROSS1, lim1 = make_test_containers(mesh, fe2_val=1e-3, ts2_val=1e-3, fes_val=1e-1)
    rn.FeS_precipitation_dissolution_smooth_transition(c1, k, lim1, LHS1, RHS1, RATES1, CROSS1, mp)
    assert RATES1["FeS"][0] > 0.0
    fe2_fes_coeffs = [coeff for src, coeff in CROSS1["Fe2_total"] if src == "FeS"]
    assert np.all(fe2_fes_coeffs[0] == 0.0)

    # 2. Undersaturated: Omega < 1
    c2, LHS2, RHS2, RATES2, CROSS2, lim2 = make_test_containers(mesh, fe2_val=1e-7, ts2_val=1e-7, fes_val=1e-1)
    rn.FeS_precipitation_dissolution_smooth_transition(c2, k, lim2, LHS2, RHS2, RATES2, CROSS2, mp)
    assert RATES2["FeS"][0] < 0.0
    fes_prec_coeffs = [coeff for src, coeff in CROSS2["FeS"] if src in ("Fe2_total", "TS2")]
    assert all(np.all(coeff == 0.0) for coeff in fes_prec_coeffs)


def test_smooth_transition_independent_of_dt():
    """Unlike flux-limited scheme, smooth transition does not depend on current_dt."""
    mesh = Grid1D(nx=3)
    k = MockK()

    mp1 = MockMP()
    mp1.current_dt = 1.0  # 1 second
    c1, LHS1, RHS1, RATES1, CROSS1, lim1 = make_test_containers(mesh, fe2_val=1e-3, ts2_val=1e-3, fes_val=1e-2)
    rn.FeS_precipitation_dissolution_smooth_transition(c1, k, lim1, LHS1, RHS1, RATES1, CROSS1, mp1)

    mp2 = MockMP()
    mp2.current_dt = 365.0 * 86400.0  # 1 year
    c2, LHS2, RHS2, RATES2, CROSS2, lim2 = make_test_containers(mesh, fe2_val=1e-3, ts2_val=1e-3, fes_val=1e-2)
    rn.FeS_precipitation_dissolution_smooth_transition(c2, k, lim2, LHS2, RHS2, RATES2, CROSS2, mp2)

    np.testing.assert_allclose(RATES1["FeS"], RATES2["FeS"], rtol=1e-12)
    np.testing.assert_allclose(LHS1["TS2"], LHS2["TS2"], rtol=1e-12)


def test_smooth_transition_stoichiometry_and_conservation():
    """Verify exact 1:1 stoichiometry and zero explicit residual."""
    mesh = Grid1D(nx=3)
    k = MockK()
    mp = MockMP()

    c, LHS, RHS, RATES, CROSS, lim = make_test_containers(mesh, fe2_val=1e-2, ts2_val=1e-2, fes_val=1e-3)
    rn.FeS_precipitation_dissolution_smooth_transition(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    np.testing.assert_allclose(RATES["Fe2_total"], RATES["TS2"], rtol=1e-12)
    np.testing.assert_allclose(RATES["Fe2_total"], -RATES["FeS"], rtol=1e-12)
    assert np.all(RHS["Fe2_total"] == 0.0)
    assert np.all(RHS["TS2"] == 0.0)
    assert np.all(RHS["FeS"] == 0.0)
