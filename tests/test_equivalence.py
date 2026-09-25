import sys
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parents[1]
experiments_dir = repo_root / "nbk" / "experiments"
if not experiments_dir.exists():
    experiments_dir = repo_root / "experiments"
if str(experiments_dir) not in sys.path:
    sys.path.insert(0, str(experiments_dir))

import reactions_new as rn
import generated_equations as gen_rn

from fipyrite.diff_lib import data_container

class MockVariable:
    def __init__(self, value):
        self.value = value
    def __array__(self, dtype=None):
        return self.value
    def __getattr__(self, name):
        return getattr(self.value, name)
    def __getitem__(self, item):
        return self.value[item]
    def __len__(self):
        return len(self.value)
    def __add__(self, other):
        return self.value + getattr(other, "value", other)
    def __radd__(self, other):
        return getattr(other, "value", other) + self.value
    def __sub__(self, other):
        return self.value - getattr(other, "value", other)
    def __rsub__(self, other):
        return getattr(other, "value", other) - self.value
    def __mul__(self, other):
        return self.value * getattr(other, "value", other)
    def __rmul__(self, other):
        return getattr(other, "value", other) * self.value
    def __truediv__(self, other):
        return self.value / getattr(other, "value", other)
    def __rtruediv__(self, other):
        return getattr(other, "value", other) / self.value
    def __pow__(self, other):
        return self.value ** getattr(other, "value", other)

class MockC:
    def __init__(self):
        self.SO4 = MockVariable(np.array([28.2, 25.0, 20.0]))
        self.SO4_32 = MockVariable(np.array([28.1, 24.9, 19.9]))
        self.TS2 = MockVariable(np.array([0.1, 0.5, 1.2]))
        self.TS2_32 = MockVariable(np.array([0.09, 0.49, 1.18]))
        self.O2 = MockVariable(np.array([0.28, 0.1, 0.0]))
        self.POC_fast = MockVariable(np.array([1000.0, 950.0, 900.0]))
        self.POC_slow = MockVariable(np.array([500.0, 480.0, 460.0]))
        self.Fe3 = MockVariable(np.array([50.0, 45.0, 40.0]))
        self.Fe2_total = MockVariable(np.array([0.0, 0.5, 1.0]))
        self.S0 = MockVariable(np.array([0.0, 0.1, 0.2]))
        self.S0_32 = MockVariable(np.array([0.0, 0.09, 0.18]))
        self.FeS = MockVariable(np.array([15.0, 14.0, 13.0]))
        self.FeS_32 = MockVariable(np.array([14.9, 13.9, 12.9]))

class MockMP:
    def __init__(self):
        self.phi = np.array([0.8, 0.8, 0.8])
        self.POC_O2_ratio = 1.0
        self.msr_alpha = 1.07
        self.TS2_O2_alpha = 0.995
        self.h2s_hs_alpha = 1.002
        self.hs_frac = 0.9
        self.h2s_frac = 0.1
        self.isotopes = True
        self.K_O2 = 0.001 * 0.8
        self.K_O2_TS2 = 0.001
        self.monod_scheme = "picard"
        self.k = data_container({
            "POC_fast": 1.2e-9,
            "POC_slow": 1.2e-10,
            "TS2_O2": 1.5e-6,
            "S0_O2": 2.0e-7,
            "Fe3_hs": 3.0e-8,
            "Fe2_O2": 4.0e-7,
            "FeS_O2": 5.0e-7,
        })

@pytest.fixture
def mock_inputs():
    c = MockC()
    k = data_container({"poc_species": "POC_fast", "poc_k": "POC_fast"})
    lim = {
        "O2_implicit": MockVariable(1.0 / (c.O2 + 0.001)),
        "O2_implicit_TS2": MockVariable(1.0 / (c.O2 + 0.001)),
        "O2_inhibit": MockVariable(0.001 / (c.O2 + 0.001)),
        "TS2": MockVariable(0.1 / (c.TS2 + 0.1)),
        "SO4_implicit": MockVariable(1.0 / (c.SO4 + 0.9)),
        "SO4_explicit": MockVariable(c.SO4 / (c.SO4 + 0.9)),
        "SO4_alpha_explicit": MockVariable(c.SO4 / (c.SO4 + 0.2)),
        "TS2_alpha_explicit": MockVariable(c.TS2 / (c.TS2 + 0.01)),
        "Fe3_implicit": MockVariable(1.0 / (c.Fe3 + 0.001)),
        "Fe3_diss_red_implicit": MockVariable(1.0 / (c.Fe3 + 10.4)),
        "Fe3_diss_red_inhib": MockVariable(10.4 / (c.Fe3 + 10.4)),
    }
    mp = MockMP()
    return c, k, lim, mp


def reset_accumulators():
    species_list = ["SO4", "SO4_32", "TS2", "TS2_32", "O2", "POC_fast", "POC_slow", "Fe3", "Fe2_total", "S0", "S0_32", "FeS", "FeS_32"]
    LHS = {s: np.zeros(3) for s in species_list}
    RHS = {s: np.zeros(3) for s in species_list}
    RATES = {s: np.zeros(3) for s in species_list}
    CROSS = {s: [] for s in species_list}
    return LHS, RHS, RATES, CROSS

def assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen):
    for s in LHS_orig:
        assert np.allclose(LHS_orig[s], LHS_gen[s]), f"LHS mismatch for species {s}"
    for s in RATES_orig:
        assert np.allclose(RATES_orig[s], RATES_gen[s]), f"RATES mismatch for species {s}"
    for s in CROSS_orig:
        assert len(CROSS_orig[s]) == len(CROSS_gen[s]), f"CROSS list length mismatch for species {s}"
        for idx in range(len(CROSS_orig[s])):
            orig_item = CROSS_orig[s][idx]
            gen_item = CROSS_gen[s][idx]
            assert orig_item[0] == gen_item[0], f"CROSS source mismatch for species {s} at idx {idx}"
            assert np.allclose(orig_item[1], gen_item[1]), f"CROSS coeff mismatch for species {s} at idx {idx}"

def test_aerobic_respiration(mock_inputs):
    c, k, lim, mp = mock_inputs
    LHS_orig, RHS_orig, RATES_orig, CROSS_orig = reset_accumulators()
    rn.aerobic_respiration(c, k, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)
    LHS_gen, RHS_gen, RATES_gen, CROSS_gen = reset_accumulators()
    gen_rn.aerobic_respiration(c, k, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)
    assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen)

def test_dissimilatory_iron_reduction(mock_inputs):
    c, k, lim, mp = mock_inputs
    LHS_orig, RHS_orig, RATES_orig, CROSS_orig = reset_accumulators()
    rn.dissimilatory_iron_reduction(c, k, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)
    LHS_gen, RHS_gen, RATES_gen, CROSS_gen = reset_accumulators()
    gen_rn.dissimilatory_iron_reduction(c, k, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)
    assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen)

def test_sulfate_reduction(mock_inputs):
    c, k, lim, mp = mock_inputs
    LHS_orig, RHS_orig, RATES_orig, CROSS_orig = reset_accumulators()
    rn.sulfate_reduction(c, k, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)
    LHS_gen, RHS_gen, RATES_gen, CROSS_gen = reset_accumulators()
    gen_rn.sulfate_reduction(c, k, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)
    assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen)

def test_hs_oxidation(mock_inputs):
    c, k, lim, mp = mock_inputs
    k_inst = mp.k
    LHS_orig, RHS_orig, RATES_orig, CROSS_orig = reset_accumulators()
    rn.hs_oxidation(c, k_inst, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)
    LHS_gen, RHS_gen, RATES_gen, CROSS_gen = reset_accumulators()
    gen_rn.hs_oxidation(c, k_inst, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)
    assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen)

def test_hs_oxidation_velde(mock_inputs):
    c, k, lim, mp = mock_inputs
    k_inst = mp.k
    LHS_orig, RHS_orig, RATES_orig, CROSS_orig = reset_accumulators()
    rn.hs_oxidation_velde(c, k_inst, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)
    LHS_gen, RHS_gen, RATES_gen, CROSS_gen = reset_accumulators()
    gen_rn.hs_oxidation_velde(c, k_inst, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)
    assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen)

def test_elemental_sulfur_oxidation(mock_inputs):
    c, k, lim, mp = mock_inputs
    k_inst = mp.k
    LHS_orig, RHS_orig, RATES_orig, CROSS_orig = reset_accumulators()
    rn.elemental_sulfur_oxidation(c, k_inst, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)
    LHS_gen, RHS_gen, RATES_gen, CROSS_gen = reset_accumulators()
    gen_rn.elemental_sulfur_oxidation(c, k_inst, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)
    assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen)

def test_sulfide_mediated_iron_reduction(mock_inputs):
    c, k, lim, mp = mock_inputs
    k_inst = mp.k
    LHS_orig, RHS_orig, RATES_orig, CROSS_orig = reset_accumulators()
    rn.sulfide_mediated_iron_reduction(c, k_inst, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)
    LHS_gen, RHS_gen, RATES_gen, CROSS_gen = reset_accumulators()
    gen_rn.sulfide_mediated_iron_reduction(c, k_inst, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)
    assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen)

def test_sulfide_mediated_iron_reduction_velde(mock_inputs):
    c, k, lim, mp = mock_inputs
    k_inst = mp.k
    LHS_orig, RHS_orig, RATES_orig, CROSS_orig = reset_accumulators()
    rn.sulfide_mediated_iron_reduction_velde(c, k_inst, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)
    LHS_gen, RHS_gen, RATES_gen, CROSS_gen = reset_accumulators()
    gen_rn.sulfide_mediated_iron_reduction_velde(c, k_inst, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)
    assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen)

def test_Fe2_oxidation(mock_inputs):
    c, k, lim, mp = mock_inputs
    k_inst = mp.k
    
    # Original
    LHS_orig, RHS_orig, RATES_orig, CROSS_orig = reset_accumulators()
    rn.Fe2_oxidation(c, k_inst, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)
    
    # Generated
    LHS_gen, RHS_gen, RATES_gen, CROSS_gen = reset_accumulators()
    gen_rn.Fe2_oxidation(c, k_inst, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)
    
    # Assert equivalence
    # Note: the original used add_implicit_coupling_new(CROSS, ..., "Fe3", "Fe2_total", ...)
    # which is mathematically equivalent to add_coupled_reaction(CROSS, ..., master_species="Fe2_total", products={"Fe3": 4.0}, ...).
    # Since they populate the same matrix elements, we check the equivalence.
    assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen)

def test_FeS_oxidation(mock_inputs):
    c, k, lim, mp = mock_inputs
    k_inst = mp.k
    LHS_orig, RHS_orig, RATES_orig, CROSS_orig = reset_accumulators()
    rn.FeS_oxidation(c, k_inst, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)
    LHS_gen, RHS_gen, RATES_gen, CROSS_gen = reset_accumulators()
    gen_rn.FeS_oxidation(c, k_inst, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)
    assert_accumulators_match(LHS_orig, RHS_orig, RATES_orig, CROSS_orig, LHS_gen, RHS_gen, RATES_gen, CROSS_gen)
