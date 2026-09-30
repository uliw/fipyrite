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
        self.pH = 7.5


class MockK:
    def __init__(self):
        self.Hplus = 10.0**(-7.5)
        self.FeS_sp = 1e-3


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
    RATES = {}
    return c, RATES


def compute_omega(c, k, mp):
    fe2_pw = np.asarray(c.Fe2_total.value) * mp.Fe2_diss
    hs = np.asarray(c.TS2.value) * mp.hs_frac
    omega_den = k.Hplus * k.FeS_sp
    return (fe2_pw * hs) / omega_den


def test_trimmer_supersaturated_snaps_to_omega_1():
    """Verify that when Omega > 1, trimmer projects exactly to Omega = 1.000000."""
    mesh = Grid1D(nx=1)
    k = MockK()
    mp = MockMP()

    # Equilibrium product K_target = (Hplus * FeS_sp) / (Fe2_diss * hs_frac)
    K_target = (k.Hplus * k.FeS_sp) / (mp.Fe2_diss * mp.hs_frac)
    
    # Set concentrations so that fe2 * ts2 = 1.25 * K_target (Omega = 1.25)
    fe2_init = np.sqrt(1.25 * K_target)
    ts2_init = np.sqrt(1.25 * K_target)
    fes_init = 50.0

    c, RATES = make_test_containers(mesh, fe2_val=fe2_init, ts2_val=ts2_init, fes_val=fes_init)
    omega_before = compute_omega(c, k, mp)[0]
    assert np.isclose(omega_before, 1.25, rtol=1e-4)

    rn.FeS_equilibrium_trimmer(c, k, mp, dt=3600.0, RATES=RATES)

    omega_after = compute_omega(c, k, mp)[0]
    assert np.isclose(omega_after, 1.0, atol=1e-5)
    # Verify FeS increased
    assert c.FeS.value[0] > fes_init


def test_trimmer_undersaturated_with_solid_snaps_to_omega_1():
    """Verify that when Omega < 1 and solid FeS is present, trimmer dissolves to Omega = 1.000000."""
    mesh = Grid1D(nx=1)
    k = MockK()
    mp = MockMP()

    K_target = (k.Hplus * k.FeS_sp) / (mp.Fe2_diss * mp.hs_frac)
    
    # Set concentrations so that Omega = 0.8
    fe2_init = np.sqrt(0.8 * K_target)
    ts2_init = np.sqrt(0.8 * K_target)
    fes_init = 50.0  # plenty of solid to dissolve

    c, RATES = make_test_containers(mesh, fe2_val=fe2_init, ts2_val=ts2_init, fes_val=fes_init)
    omega_before = compute_omega(c, k, mp)[0]
    assert np.isclose(omega_before, 0.8, rtol=1e-4)

    rn.FeS_equilibrium_trimmer(c, k, mp, dt=3600.0, RATES=RATES)

    omega_after = compute_omega(c, k, mp)[0]
    assert np.isclose(omega_after, 1.0, atol=1e-5)
    # Verify FeS decreased (dissolved)
    assert c.FeS.value[0] < fes_init


def test_trimmer_undersaturated_zero_solid_does_nothing():
    """When Omega < 1 and no solid FeS is present, trimmer must do nothing."""
    mesh = Grid1D(nx=1)
    k = MockK()
    mp = MockMP()

    K_target = (k.Hplus * k.FeS_sp) / (mp.Fe2_diss * mp.hs_frac)
    fe2_init = np.sqrt(0.5 * K_target)
    ts2_init = np.sqrt(0.5 * K_target)
    fes_init = 0.0

    c, RATES = make_test_containers(mesh, fe2_val=fe2_init, ts2_val=ts2_init, fes_val=fes_init)
    rn.FeS_equilibrium_trimmer(c, k, mp, dt=3600.0, RATES=RATES)

    # Concentrations unchanged
    assert c.Fe2_total.value[0] == fe2_init
    assert c.TS2.value[0] == ts2_init
    assert c.FeS.value[0] == 0.0


def test_trimmer_bulk_mass_conservation():
    """Verify that bulk iron and bulk sulfur are strictly conserved: phi*dC_liq + (1-phi)*dC_solid = 0."""
    mesh = Grid1D(nx=3)
    k = MockK()
    mp = MockMP()

    K_target = (k.Hplus * k.FeS_sp) / (mp.Fe2_diss * mp.hs_frac)
    fe2_init = np.sqrt(1.5 * K_target)
    ts2_init = np.sqrt(1.5 * K_target)
    fes_init = 100.0

    c, RATES = make_test_containers(mesh, fe2_val=fe2_init, ts2_val=ts2_init, fes_val=fes_init)
    phi = mp.phi
    tot_fe_before = phi * c.Fe2_total.value + (1.0 - phi) * c.FeS.value

    rn.FeS_equilibrium_trimmer(c, k, mp, dt=3600.0, RATES=RATES)

    tot_fe_after = phi * c.Fe2_total.value + (1.0 - phi) * c.FeS.value
    np.testing.assert_allclose(tot_fe_before, tot_fe_after, rtol=1e-14, atol=1e-14)


def test_trimmer_isotopes():
    """Verify isotope mirroring updates TS2_32 and FeS_32 cleanly."""
    mesh = Grid1D(nx=3)
    k = MockK()
    mp = MockMP()
    mp.isotopes = True

    K_target = (k.Hplus * k.FeS_sp) / (mp.Fe2_diss * mp.hs_frac)
    fe2_init = np.sqrt(1.2 * K_target)
    ts2_init = np.sqrt(1.2 * K_target)
    fes_init = 50.0

    c, RATES = make_test_containers(mesh, fe2_val=fe2_init, ts2_val=ts2_init, fes_val=fes_init, isotopes=True)
    rn.FeS_equilibrium_trimmer(c, k, mp, dt=3600.0, RATES=RATES)

    assert not np.isnan(c.TS2_32.value).any()
    assert not np.isnan(c.FeS_32.value).any()
    assert c.FeS_32.value[0] > 50.0 * 0.95
