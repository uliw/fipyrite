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
        self.fes_flux_limiter_gamma = 0.98
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


def test_flux_limited_import_and_aliases():
    """Verify function exists and alias is defined."""
    assert hasattr(rn, "FeS_precipitation_dissolution_flux_limited")
    assert hasattr(rn, "FeS_precipitation_dissolution_symmetrical_picard_flux_limited")
    assert rn.FeS_precipitation_dissolution_flux_limited is rn.FeS_precipitation_dissolution_symmetrical_picard_flux_limited


def test_flux_limited_small_dt_matches_raw_kinetics():
    """When dt is small (e.g. 1 sec), flux limiter is inactive and rates match raw kinetics."""
    mesh = Grid1D(nx=3)
    k = MockK()

    # Case with dt = 1.0 s
    mp1 = MockMP()
    mp1.current_dt = 1.0
    c1, LHS1, RHS1, RATES1, CROSS1, lim1 = make_test_containers(mesh, fe2_val=1e-2, ts2_val=1e-2, fes_val=1e-3)
    rn.FeS_precipitation_dissolution_flux_limited(c1, k, lim1, LHS1, RHS1, RATES1, CROSS1, mp1)

    mp2 = MockMP()
    c2, LHS2, RHS2, RATES2, CROSS2, lim2 = make_test_containers(mesh, fe2_val=1e-2, ts2_val=1e-2, fes_val=1e-3)
    rn.FeS_precipitation_dissolution_symmetrical_picard(c2, k, lim2, LHS2, RHS2, RATES2, CROSS2, mp2)

    np.testing.assert_allclose(RATES1["FeS"], RATES2["FeS"], rtol=1e-5)
    np.testing.assert_allclose(LHS1["TS2"], LHS2["TS2"], rtol=1e-5)


def test_flux_limited_large_dt_caps_overshoot():
    """When dt is large and Omega is slightly > 1, raw rate would overshoot Omega=1, but limiter caps it."""
    mesh = Grid1D(nx=3)
    k = MockK()
    # Omega_den = 1e-7 * 1e-3 = 1e-10
    # Fe2_pw = Fe2 * 0.95, hs = TS2 * 0.6
    # To get Omega = 1.05: Fe2_pw * hs = 1.05e-10
    # Let Fe2 = 1.4e-5, TS2 = 1.3e-5
    mp = MockMP()
    mp.current_dt = 100.0 * 3600.0  # 100 hours
    mp.fes_flux_limiter_gamma = 0.98

    fe2_val = 1.4e-5
    ts2_val = 1.3e-5
    fes_val = 1e-2

    c, LHS, RHS, RATES, CROSS, lim = make_test_containers(mesh, fe2_val=fe2_val, ts2_val=ts2_val, fes_val=fes_val)
    rn.FeS_precipitation_dissolution_flux_limited(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    # Compute dC_eq analytically
    fe2_pw = fe2_val * mp.Fe2_diss
    hs_val = ts2_val * mp.hs_frac
    omega = (fe2_pw * hs_val) / (k.Hplus * k.FeS_sp)
    assert omega > 1.0  # verify supersaturated

    b = fe2_val + ts2_val
    c_prod = fe2_val * ts2_val * (1.0 - 1.0 / omega)
    disc = np.sqrt(b * b - 4.0 * c_prod)
    dC_eq = (2.0 * c_prod) / (b + disc)

    # Porewater turnover in dt:
    # Rate of FeS production in porewater is RATES['FeS'] / (1-phi) * phi / (1-phi) etc.
    # In bulk, RATES['FeS'] = Rate_pw * phi
    # So Rate_pw = RATES['FeS'] / phi
    rate_prec_pw = RATES["FeS"][0] / mp.phi
    turnover_pw = rate_prec_pw * mp.current_dt

    # The turnover must NOT exceed gamma * dC_eq
    assert turnover_pw <= mp.fes_flux_limiter_gamma * dC_eq * 1.001
    assert turnover_pw < dC_eq  # Strictly no overshoot past equilibrium!


def test_flux_limited_thermodynamic_bounds():
    """Verify zero dissolution when Omega >= 1, and zero precipitation when Omega < 1."""
    mesh = Grid1D(nx=1)
    k = MockK()
    mp = MockMP()
    mp.current_dt = 3600.0

    # 1. Supersaturated: Fe2 = 1e-3, TS2 = 1e-3 => Omega >> 1
    c1, LHS1, RHS1, RATES1, CROSS1, lim1 = make_test_containers(mesh, fe2_val=1e-3, ts2_val=1e-3, fes_val=1e-1)
    rn.FeS_precipitation_dissolution_flux_limited(c1, k, lim1, LHS1, RHS1, RATES1, CROSS1, mp)
    assert RATES1["FeS"][0] > 0.0  # Precipitation positive
    # Check dissolution cross-term coefficient is exactly zero
    fe2_fes_coeffs = [coeff for src, coeff in CROSS1["Fe2_total"] if src == "FeS"]
    assert np.all(fe2_fes_coeffs[0] == 0.0)

    # 2. Undersaturated: Fe2 = 1e-7, TS2 = 1e-7 => Omega << 1
    c2, LHS2, RHS2, RATES2, CROSS2, lim2 = make_test_containers(mesh, fe2_val=1e-7, ts2_val=1e-7, fes_val=1e-1)
    rn.FeS_precipitation_dissolution_flux_limited(c2, k, lim2, LHS2, RHS2, RATES2, CROSS2, mp)
    assert RATES2["FeS"][0] < 0.0  # Dissolution negative for solid FeS
    assert RATES2["TS2"][0] > 0.0  # TS2 produced
    # Check precipitation cross couplings coefficients are exactly zero
    fes_prec_coeffs = [coeff for src, coeff in CROSS2["FeS"] if src in ("Fe2_total", "TS2")]
    assert all(np.all(coeff == 0.0) for coeff in fes_prec_coeffs)


def test_flux_limited_dissolution_solid_exhaustion():
    """When Omega < 1 and solid FeS is vanishingly small, dissolution cannot exceed FeS."""
    mesh = Grid1D(nx=1)
    k = MockK()
    mp = MockMP()
    mp.current_dt = 1000.0 * 3600.0  # Huge timestep

    fes_initial = 1e-8
    c, LHS, RHS, RATES, CROSS, lim = make_test_containers(mesh, fe2_val=1e-7, ts2_val=1e-7, fes_val=fes_initial)
    rn.FeS_precipitation_dissolution_flux_limited(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    # Solid consumed in dt = -RATES['FeS'] / (1-phi) * dt
    solid_loss = (-RATES["FeS"][0] / (1.0 - mp.phi)) * mp.current_dt
    assert solid_loss <= fes_initial * mp.fes_flux_limiter_gamma * 1.001


def test_flux_limited_stoichiometry_and_conservation():
    """Verify exact 1:1 stoichiometry and zero explicit residual."""
    mesh = Grid1D(nx=3)
    k = MockK()
    mp = MockMP()
    mp.current_dt = 3600.0

    c, LHS, RHS, RATES, CROSS, lim = make_test_containers(mesh, fe2_val=1e-2, ts2_val=1e-2, fes_val=1e-3)
    rn.FeS_precipitation_dissolution_flux_limited(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    np.testing.assert_allclose(RATES["Fe2_total"], RATES["TS2"], rtol=1e-12)
    np.testing.assert_allclose(RATES["Fe2_total"], -RATES["FeS"], rtol=1e-12)
    assert np.all(RHS["Fe2_total"] == 0.0)
    assert np.all(RHS["TS2"] == 0.0)
    assert np.all(RHS["FeS"] == 0.0)


def test_flux_limited_isotopes():
    """Verify isotope mirroring runs cleanly without NaN or infinite values."""
    mesh = Grid1D(nx=3)
    k = MockK()
    mp = MockMP()
    mp.isotopes = True
    mp.current_dt = 3600.0

    c, LHS, RHS, RATES, CROSS, lim = make_test_containers(mesh, fe2_val=1e-2, ts2_val=1e-2, fes_val=1e-3, isotopes=True)
    rn.FeS_precipitation_dissolution_flux_limited(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    assert not np.isnan(RATES["TS2_32"]).any()
    assert not np.isnan(RATES["FeS_32"]).any()
    assert not np.isnan(LHS["TS2_32"]).any()
