import sys
from pathlib import Path
import numpy as np
import pytest
from fipyrite.diff_lib import VariableArray as CellVariable, Mesh1D as Grid1D

# Ensure experiments and src paths are importable
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
        self.fes_picard_weight_fe = 0.5


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


def test_precipitation_stoichiometry_and_conservation():
    """Verify that precipitation strictly conserves mass across Fe2, TS2, and FeS."""
    mesh = Grid1D(nx=3)
    mp = MockMP()
    k = MockK()
    # High concentrations ensuring Omega >> 1 (precipitation)
    c, LHS, RHS, RATES, CROSS, lim = make_test_containers(mesh, fe2_val=1e-2, ts2_val=1e-2, fes_val=1e-3)

    rn.FeS_precipitation_dissolution_symmetrical_picard(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    phi = mp.phi
    # 1. Check RATES mass conservation: -RATES[Fe2] == -RATES[TS2] == +RATES[FeS]
    assert np.all(RATES["Fe2_total"] < 0.0)
    assert np.all(RATES["TS2"] < 0.0)
    assert np.all(RATES["FeS"] > 0.0)

    np.testing.assert_allclose(RATES["Fe2_total"], RATES["TS2"], rtol=1e-12)
    np.testing.assert_allclose(RATES["Fe2_total"], -RATES["FeS"], rtol=1e-12)

    # 2. Check that RHS has zero explicit residual
    assert np.all(RHS["Fe2_total"] == 0.0)
    assert np.all(RHS["TS2"] == 0.0)
    assert np.all(RHS["FeS"] == 0.0)


def test_precipitation_symmetrical_diagonals():
    """Verify that when [Fe2] == [TS2] and w_Fe == 0.5, LHS diagonal sinks are exactly identical."""
    mesh = Grid1D(nx=3)
    mp = MockMP()
    mp.fes_picard_weight_fe = 0.5
    k = MockK()
    c, LHS, RHS, RATES, CROSS, lim = make_test_containers(mesh, fe2_val=5e-3, ts2_val=5e-3, fes_val=1e-3)

    rn.FeS_precipitation_dissolution_symmetrical_picard(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    # Both reactants must have identical negative diagonal entries in LHS
    assert np.all(LHS["Fe2_total"] < 0.0)
    assert np.all(LHS["TS2"] < 0.0)
    np.testing.assert_allclose(LHS["Fe2_total"], LHS["TS2"], rtol=1e-12)

    # Cross couplings must be symmetric
    cross_fe2 = [entry for entry in CROSS["Fe2_total"] if entry[0] == "TS2"]
    cross_ts2 = [entry for entry in CROSS["TS2"] if entry[0] == "Fe2_total"]
    assert len(cross_fe2) == 1
    assert len(cross_ts2) == 1
    np.testing.assert_allclose(cross_fe2[0][1], cross_ts2[0][1], rtol=1e-12)


def test_limiting_reactant_diagonal_damping():
    """Verify that depleted reactant receives proportional high diagonal damping."""
    mesh = Grid1D(nx=1)
    mp = MockMP()
    k = MockK()

    # Case A: Fe2 is trace (1e-6) and TS2 is abundant (1e-2)
    c_A, LHS_A, RHS_A, RATES_A, CROSS_A, lim_A = make_test_containers(mesh, fe2_val=1e-6, ts2_val=1e-2, fes_val=1e-3)
    rn.FeS_precipitation_dissolution_symmetrical_picard(c_A, k, lim_A, LHS_A, RHS_A, RATES_A, CROSS_A, mp)

    # Fe2 diagonal sink magnitude must be much larger than TS2 diagonal sink magnitude
    # because prec_coeff_Fe2 = 0.5 * R / Fe2, while prec_coeff_TS2 = 0.5 * R / TS2
    assert abs(LHS_A["Fe2_total"][0]) > 1000 * abs(LHS_A["TS2"][0])

    # Case B: TS2 is trace (1e-6) and Fe2 is abundant (1e-2)
    c_B, LHS_B, RHS_B, RATES_B, CROSS_B, lim_B = make_test_containers(mesh, fe2_val=1e-2, ts2_val=1e-6, fes_val=1e-3)
    rn.FeS_precipitation_dissolution_symmetrical_picard(c_B, k, lim_B, LHS_B, RHS_B, RATES_B, CROSS_B, mp)

    assert abs(LHS_B["TS2"][0]) > 1000 * abs(LHS_B["Fe2_total"][0])


def test_custom_reactant_weighting():
    """Verify that user-configured fes_picard_weight_fe partitions diagonal damping accordingly."""
    mesh = Grid1D(nx=1)
    mp = MockMP()
    mp.fes_picard_weight_fe = 0.8  # 80% on Fe2, 20% on TS2
    k = MockK()

    c, LHS, RHS, RATES, CROSS, lim = make_test_containers(mesh, fe2_val=2e-3, ts2_val=2e-3, fes_val=1e-3)
    rn.FeS_precipitation_dissolution_symmetrical_picard(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    # LHS[Fe2] / LHS[TS2] should be 0.8 / 0.2 = 4.0
    ratio = LHS["Fe2_total"][0] / LHS["TS2"][0]
    np.testing.assert_allclose(ratio, 4.0, rtol=1e-10)

    # But total bulk rates remain 1:1!
    np.testing.assert_allclose(RATES["Fe2_total"], RATES["TS2"], rtol=1e-12)


def test_dissolution_branch():
    """Verify dissolution branch (Omega < 1) stoichiometry and unimolecular FeS coupling."""
    mesh = Grid1D(nx=3)
    mp = MockMP()
    k = MockK()
    # Trace solute concentrations ensuring Omega << 1 (dissolution)
    c, LHS, RHS, RATES, CROSS, lim = make_test_containers(mesh, fe2_val=1e-9, ts2_val=1e-9, fes_val=1e-2)

    rn.FeS_precipitation_dissolution_symmetrical_picard(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    # Dissolution consumes FeS, produces Fe2 and TS2
    assert np.all(RATES["FeS"] < 0.0)
    assert np.all(RATES["Fe2_total"] > 0.0)
    assert np.all(RATES["TS2"] > 0.0)

    # 1:1 bulk stoichiometry:
    np.testing.assert_allclose(RATES["Fe2_total"], -RATES["FeS"], rtol=1e-12)
    np.testing.assert_allclose(RATES["TS2"], -RATES["FeS"], rtol=1e-12)

    # FeS has diagonal sink on LHS
    assert np.all(LHS["FeS"] < 0.0)


def test_isotopes_support():
    """Verify that isotopes partitioning runs cleanly without errors and preserves mass."""
    mesh = Grid1D(nx=3)
    mp = MockMP()
    mp.isotopes = True
    k = MockK()

    # Precipitation with isotopes
    c_p, LHS_p, RHS_p, RATES_p, CROSS_p, lim_p = make_test_containers(
        mesh, fe2_val=1e-2, ts2_val=1e-2, fes_val=1e-3, isotopes=True
    )
    rn.FeS_precipitation_dissolution_symmetrical_picard(c_p, k, lim_p, LHS_p, RHS_p, RATES_p, CROSS_p, mp)

    assert "TS2_32" in RATES_p
    assert "FeS_32" in RATES_p
    assert np.all(RATES_p["TS2_32"] < 0.0)
    assert np.all(RATES_p["FeS_32"] > 0.0)
    np.testing.assert_allclose(RATES_p["TS2_32"], -RATES_p["FeS_32"], rtol=1e-12)

    # Dissolution with isotopes
    c_d, LHS_d, RHS_d, RATES_d, CROSS_d, lim_d = make_test_containers(
        mesh, fe2_val=1e-9, ts2_val=1e-9, fes_val=1e-2, isotopes=True
    )
    rn.FeS_precipitation_dissolution_symmetrical_picard(c_d, k, lim_d, LHS_d, RHS_d, RATES_d, CROSS_d, mp)
    assert np.all(RATES_d["FeS_32"] < 0.0)
    assert np.all(RATES_d["TS2_32"] > 0.0)
    np.testing.assert_allclose(RATES_d["TS2_32"], -RATES_d["FeS_32"], rtol=1e-12)


def test_alias_equivalence():
    """Verify that FeS_precipitation_dissolution_symmetrical is an exact alias."""
    assert rn.FeS_precipitation_dissolution_symmetrical is rn.FeS_precipitation_dissolution_symmetrical_picard


def test_coupled_matrix_assembly_and_update():
    """Verify end-to-end compatibility with DirectAssembledSystem."""
    from fipyrite.direct_assembled_solver import DirectAssembledSystem

    mesh = Grid1D(nx=3)
    mp = MockMP()
    k = MockK()
    mp.diagenetic_reactions = [[rn.FeS_precipitation_dissolution_symmetrical_picard, k]]
    mp.K_O2 = 1e-4
    mp.K_TS2 = 1e-4
    mp.K_SO4 = 1e-4
    mp.K_Fe3 = 1e-4
    mp.K_Fe3_diss_red = 1e-4
    mp.K_O2_TS2 = 1e-4
    mp.K_epsilon_msr = 1e-4
    mp.K_epsilon_TS2_O2 = 1e-4

    all_species = ["Fe2_total", "TS2", "FeS", "SO4", "O2", "Fe3"]
    class MockD:
        def __init__(self):
            for s in all_species:
                setattr(self, s, 1e-9)
            self.FeS = 0.0
            self.Fe3 = 0.0
            self.D_bio = 0.0
            self.D_irr = 0.0

    D_mol = MockD()
    species_list = ["Fe2_total", "TS2", "FeS"]
    c = data_container({
        s: CellVariable(mesh=mesh, value=1e-3, hasOld=True) for s in all_species
    })
    bc_map = {
        "Fe2_total": {"type": "dissolved", "top": 1e-3},
        "TS2": {"type": "dissolved", "top": 1e-3},
        "FeS": {"type": "solid", "top": 0.0},
        "SO4": {"type": "dissolved", "top": 1e-3},
        "O2": {"type": "dissolved", "top": 1e-3},
        "Fe3": {"type": "solid", "top": 0.0},
    }

    species_struct = [{"name": s, "var": c[s]} for s in species_list]
    assembled_system = DirectAssembledSystem(
        species_struct=species_struct,
        mesh=mesh,
        mp=mp,
        bc_map=bc_map,
        D_mol=D_mol,
        z=mesh.cellCenters[0],
    )

    assert assembled_system is not None
    assert "Fe2_total" in assembled_system.ab_transport
    assert "TS2" in assembled_system.ab_transport
    assert "FeS" in assembled_system.ab_transport


def test_patankar_auto_weighting_regimes():
    """Verify Patankar auto-weighting behavior across limiting reactant regimes."""
    mesh = Grid1D(nx=1)
    mp = MockMP()
    mp.fes_picard_weight_fe = "auto"
    k = MockK()

    # Regime 1: Fe2 >> TS2 (e.g. Velde at depth: Fe2 = 2.0, TS2 = 0.005)
    c_1, LHS_1, RHS_1, RATES_1, CROSS_1, lim_1 = make_test_containers(
        mesh, fe2_val=2.0, ts2_val=0.005, fes_val=1.0
    )
    rn.FeS_precipitation_dissolution_symmetrical_picard(c_1, k, lim_1, LHS_1, RHS_1, RATES_1, CROSS_1, mp)
    # TS2 is limiting: it receives virtually all diagonal damping, and Fe2 cross-term is nearly zero
    assert abs(LHS_1["TS2"][0]) > 1e4 * abs(LHS_1["Fe2_total"][0])
    # Cross sink on TS2 from Fe2 should be negligible
    cross_ts2_fe2 = [entry for entry in CROSS_1["TS2"] if entry[0] == "Fe2_total"][0][1]
    assert abs(cross_ts2_fe2) < 1e-4 * abs(LHS_1["TS2"][0])

    # Regime 2: TS2 >> Fe2 (Fe2 is limiting)
    c_2, LHS_2, RHS_2, RATES_2, CROSS_2, lim_2 = make_test_containers(
        mesh, fe2_val=0.005, ts2_val=2.0, fes_val=1.0
    )
    rn.FeS_precipitation_dissolution_symmetrical_picard(c_2, k, lim_2, LHS_2, RHS_2, RATES_2, CROSS_2, mp)
    assert abs(LHS_2["Fe2_total"][0]) > 1e4 * abs(LHS_2["TS2"][0])

    # Regime 3: Fe2 == TS2 (equimolar) -> exactly 50/50 symmetric
    c_3, LHS_3, RHS_3, RATES_3, CROSS_3, lim_3 = make_test_containers(
        mesh, fe2_val=0.1, ts2_val=0.1, fes_val=1.0
    )
    rn.FeS_precipitation_dissolution_symmetrical_picard(c_3, k, lim_3, LHS_3, RHS_3, RATES_3, CROSS_3, mp)
    np.testing.assert_allclose(LHS_3["Fe2_total"], LHS_3["TS2"], rtol=1e-12)


def test_isotopes_operator_mirroring_and_delta_stability():
    """Verify that TS2_32 operator matches bulk TS2 exactly, ensuring zero artificial d34S drift."""
    from fipyrite.diff_lib import get_delta

    mesh = Grid1D(nx=1)
    mp = MockMP()
    mp.isotopes = True
    mp.h2s_hs_alpha = 1.000  # No equilibrium fractionation to isolate numerical operator effect
    mp.fes_picard_weight_fe = "auto"
    k = MockK()

    # Initial concentrations
    fe2_val = 1.84
    ts2_val = 0.005
    fes_val = 250.0
    c, LHS, RHS, RATES, CROSS, lim = make_test_containers(
        mesh, fe2_val=fe2_val, ts2_val=ts2_val, fes_val=fes_val, isotopes=True
    )
    # Set natural isotopic abundance
    c.TS2_32.setValue(ts2_val / (1.0 + mp.VCDT))
    c.FeS_32.setValue(fes_val / (1.0 + mp.VCDT))

    rn.FeS_precipitation_dissolution_symmetrical_picard(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    # 1. Check that TS2_32 has off-diagonal coupling to Fe2_total
    cross_32_fe2 = [entry for entry in CROSS["TS2_32"] if entry[0] == "Fe2_total"]
    assert len(cross_32_fe2) == 1, "TS2_32 must have cross coupling to Fe2_total"

    # 2. Check that FeS_32 has cross coupling to both TS2_32 and Fe2_total
    cross_fes32_sources = [entry[0] for entry in CROSS["FeS_32"]]
    assert "TS2_32" in cross_fes32_sources
    assert "Fe2_total" in cross_fes32_sources

    # 3. Mass conservation of 32S
    np.testing.assert_allclose(RATES["TS2_32"], -RATES["FeS_32"], rtol=1e-12)

    # 4. Simulate a single 40h linear step to verify zero delta drift
    phi = mp.phi
    dt = 40.0 * 3600.0  # 40 hours
    diag_TS2 = phi / dt - LHS["TS2"][0]
    cross_ts2_fe2_coeff = [c_entry[1] for c_entry in CROSS["TS2"] if c_entry[0] == "Fe2_total"][0]
    ts2_next = ((phi / dt) * ts2_val + cross_ts2_fe2_coeff * fe2_val) / diag_TS2

    diag_32 = phi / dt - LHS["TS2_32"][0]
    cross_32_fe2_coeff = cross_32_fe2[0][1]
    ts2_32_next = ((phi / dt) * c.TS2_32.value[0] + cross_32_fe2_coeff * fe2_val) / diag_32

    # Direct delta calculation:
    ratio = (ts2_next - ts2_32_next) / ts2_32_next
    ratio_val = ratio[0] if hasattr(ratio, "__len__") else ratio
    d34S = float(1000.0 * (ratio_val - mp.VCDT) / mp.VCDT)
    np.testing.assert_allclose(d34S, 0.0, atol=1e-3)


def test_coupled_matrix_assembly_with_isotopes():
    """Verify coupled matrix assembly and update with isotopes enabled."""
    from fipyrite.direct_assembled_solver import DirectAssembledSystem

    mesh = Grid1D(nx=3)
    mp = MockMP()
    mp.isotopes = True
    k = MockK()
    mp.diagenetic_reactions = [[rn.FeS_precipitation_dissolution_symmetrical_picard, k]]
    mp.K_O2 = 1e-4
    mp.K_TS2 = 1e-4
    mp.K_SO4 = 1e-4
    mp.K_Fe3 = 1e-4
    mp.K_Fe3_diss_red = 1e-4
    mp.K_O2_TS2 = 1e-4
    mp.K_epsilon_msr = 1e-4
    mp.K_epsilon_TS2_O2 = 1e-4

    all_species = ["Fe2_total", "TS2", "FeS", "SO4", "O2", "Fe3", "TS2_32", "FeS_32"]
    class MockD:
        def __init__(self):
            for s in all_species:
                setattr(self, s, 1e-9)
            self.FeS = 0.0
            self.FeS_32 = 0.0
            self.Fe3 = 0.0
            self.D_bio = 0.0
            self.D_irr = 0.0

    D_mol = MockD()
    species_list = ["Fe2_total", "TS2", "FeS", "TS2_32", "FeS_32"]
    c = data_container({
        s: CellVariable(mesh=mesh, value=1e-3, hasOld=True) for s in all_species
    })
    bc_map = {
        "Fe2_total": {"type": "dissolved", "top": 1e-3},
        "TS2": {"type": "dissolved", "top": 1e-3},
        "FeS": {"type": "solid", "top": 0.0},
        "SO4": {"type": "dissolved", "top": 1e-3},
        "O2": {"type": "dissolved", "top": 1e-3},
        "Fe3": {"type": "solid", "top": 0.0},
        "TS2_32": {"type": "dissolved", "top": 1e-3},
        "FeS_32": {"type": "solid", "top": 0.0},
    }

    species_struct = [{"name": s, "var": c[s]} for s in species_list]
    assembled_system = DirectAssembledSystem(
        species_struct=species_struct,
        mesh=mesh,
        mp=mp,
        bc_map=bc_map,
        D_mol=D_mol,
        z=mesh.cellCenters[0],
    )

    assert assembled_system is not None
    assert "TS2_32" in assembled_system.ab_transport
    assert "FeS_32" in assembled_system.ab_transport
    assert "TS2_32" in assembled_system.b_transport
    assert "FeS_32" in assembled_system.b_transport


