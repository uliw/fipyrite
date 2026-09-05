import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure experiments and package directories are in sys.path
experiments_dir = Path(__file__).resolve().parents[1] / "experiments"
src_dir = Path(__file__).resolve().parents[1] / "src"
for p in [str(experiments_dir), str(src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fipyrite.diff_lib import add_implicit_sink, add_monod_sink, data_container
from fipyrite.generate_equations import create_reaction, load_reactions_from_py
import reactions_new as rn


class MockMP:
    def __init__(self):
        self.phi = 0.8
        self.K_O2 = 0.001 * 0.8  # so K_O2 / phi = 0.001
        self.K_O2_TS2 = 0.001
        self.hs_frac = 0.9
        self.isotopes = False
        self.monod_scheme = "hybrid"
        self.k = data_container({
            "POC_fast": 1.2e-9,
            "TS2_O2": 1.5e-6,
        })


def test_add_monod_sink_picard_equivalence():
    """Verify that scheme='picard' in add_monod_sink is numerically identical to add_implicit_sink."""
    mp = MockMP()
    C = np.array([0.28, 0.05, 0.001, 1e-6])
    K_m = 0.001
    R_max = 1.2e-9 * 1000.0  # k_val * POC

    # Legacy calculation with add_implicit_sink
    LHS_legacy = {"O2": np.zeros_like(C)}
    RHS_legacy = {"O2": np.zeros_like(C)}
    RATES_legacy = {"O2": np.zeros_like(C)}
    
    coeff_legacy = R_max / (C + K_m)
    rate_legacy = R_max * (C / (C + K_m))
    add_implicit_sink(LHS_legacy, RATES_legacy, "O2", coeff_legacy, rate_legacy, mp=mp, has_solid=False)

    # New add_monod_sink with scheme="picard"
    LHS_new = {"O2": np.zeros_like(C)}
    RHS_new = {"O2": np.zeros_like(C)}
    RATES_new = {"O2": np.zeros_like(C)}
    add_monod_sink(
        LHS_new, RHS_new, RATES_new,
        species="O2",
        conc=C,
        K_m=K_m,
        R_max=R_max,
        mp=mp,
        has_solid=False,
        scheme="picard",
    )

    assert np.allclose(LHS_legacy["O2"], LHS_new["O2"])
    assert np.allclose(RHS_legacy["O2"], RHS_new["O2"])
    assert np.allclose(RATES_legacy["O2"], RATES_new["O2"])
    assert np.allclose(RHS_new["O2"], 0.0)  # RHS must remain 0 in Picard mode


def test_add_monod_sink_newton():
    """Verify analytical Newton Jacobian derivative and explicit remainder."""
    mp = MockMP()
    C = np.array([0.28, 0.05, 0.001, 1e-6])
    K_m = 0.001
    R_max = 1.0e-6

    LHS = {"O2": np.zeros_like(C)}
    RHS = {"O2": np.zeros_like(C)}
    RATES = {"O2": np.zeros_like(C)}
    add_monod_sink(
        LHS, RHS, RATES,
        species="O2",
        conc=C,
        K_m=K_m,
        R_max=R_max,
        mp=mp,
        has_solid=False,
        scheme="newton",
    )

    phi = mp.phi
    denom = C + K_m
    expected_J = R_max * (K_m / (denom * denom))
    expected_rem = - R_max * ((C / denom) ** 2)
    expected_rate = R_max * (C / denom) * phi

    assert np.allclose(LHS["O2"], -expected_J * phi)
    assert np.allclose(RHS["O2"], expected_rem * phi)
    assert np.allclose(RATES["O2"], -expected_rate)

    # Identity check: J * C + R_rem = - R_physical
    linearized_total = expected_J * C + (-expected_rem)
    assert np.allclose(linearized_total, R_max * (C / denom))


def test_add_monod_sink_hybrid_branching():
    """Verify that hybrid scheme uses Newton for C > K_m and Picard for C <= K_m."""
    mp = MockMP()
    K_m = 0.01
    C = np.array([0.1, 0.02, 0.01, 0.005, 1e-6])  # First 2 are > K_m, last 3 are <= K_m
    R_max = 1.0e-6

    LHS = {"O2": np.zeros_like(C)}
    RHS = {"O2": np.zeros_like(C)}
    RATES = {"O2": np.zeros_like(C)}
    add_monod_sink(
        LHS, RHS, RATES,
        species="O2",
        conc=C,
        K_m=K_m,
        R_max=R_max,
        mp=mp,
        has_solid=False,
        scheme="hybrid",
    )

    phi = mp.phi
    denom = C + K_m
    J = R_max * (K_m / (denom * denom))
    R_rem = - R_max * ((C / denom) ** 2)
    coeff_picard = R_max / denom

    # For C > K_m (indices 0, 1): must match Newton
    assert np.allclose(LHS["O2"][:2], -J[:2] * phi)
    assert np.allclose(RHS["O2"][:2], R_rem[:2] * phi)

    # For C <= K_m (indices 2, 3, 4): must match Picard (RHS=0)
    assert np.allclose(LHS["O2"][2:], -coeff_picard[2:] * phi)
    assert np.allclose(RHS["O2"][2:], 0.0)


def test_generator_picard_matches_legacy():
    """Verify that create_reaction with monod_scheme='picard' produces the exact legacy code."""
    configs = load_reactions_from_py(experiments_dir / "chemical_equations.py")
    aerobic_cfg = next(c for c in configs if c["reaction_name"] == "aerobic_respiration")

    # Generate with picard (default)
    code_picard = create_reaction(**aerobic_cfg, monod_scheme="picard")
    
    assert "add_implicit_sink(LHS, RATES, 'O2'" in code_picard
    assert "add_monod_sink" not in code_picard


def test_generator_hybrid_emits_monod_sink():
    """Verify that create_reaction with monod_scheme='hybrid' emits add_monod_sink."""
    configs = load_reactions_from_py(experiments_dir / "chemical_equations.py")
    aerobic_cfg = next(c for c in configs if c["reaction_name"] == "aerobic_respiration")

    # Generate with hybrid
    code_hybrid = create_reaction(**aerobic_cfg, monod_scheme="hybrid")
    
    assert "add_monod_sink(" in code_hybrid
    assert "species='O2'" in code_hybrid
    assert "scheme=getattr(mp, 'monod_scheme', 'hybrid')" in code_hybrid
    assert "add_implicit_sink(LHS, RATES, poc_species" in code_hybrid


def test_generated_hybrid_runtime_equivalence():
    """Verify that the generated hybrid function with mp.monod_scheme='picard' gives identical results to legacy."""
    configs = load_reactions_from_py(experiments_dir / "chemical_equations.py")
    aerobic_cfg = next(c for c in configs if c["reaction_name"] == "aerobic_respiration")
    code_hybrid = create_reaction(**aerobic_cfg, monod_scheme="hybrid")

    # Compile the generated hybrid function into a temporary namespace
    ns = {
        "add_coupled_reaction": None,
        "add_implicit_sink": add_implicit_sink,
        "add_monod_sink": add_monod_sink,
        "calculate_fractionated_coeff_32": None,
        "partition_equilibrium_isotope_32": None,
    }
    exec(code_hybrid, ns)
    aerobic_respiration_hybrid = ns["aerobic_respiration"]

    # Setup inputs
    mp = MockMP()
    mp.monod_scheme = "picard"  # Test that runtime picard produces identical output
    k = data_container({"poc_species": "POC_fast", "poc_k": "POC_fast"})
    
    from test_equivalence import MockVariable

    c = data_container({
        "O2": MockVariable(np.array([0.28, 0.05, 0.001])),
        "POC_fast": MockVariable(np.array([1000.0, 950.0, 900.0])),
    })
    K_O2 = mp.K_O2 / mp.phi
    lim = {"O2_implicit": MockVariable(1.0 / (c.O2.value + K_O2))}

    # Run legacy
    LHS_orig = {"O2": np.zeros(3), "POC_fast": np.zeros(3)}
    RHS_orig = {"O2": np.zeros(3), "POC_fast": np.zeros(3)}
    RATES_orig = {"O2": np.zeros(3), "POC_fast": np.zeros(3)}
    CROSS_orig = {"O2": [], "POC_fast": []}
    rn.aerobic_respiration(c, k, lim, LHS_orig, RHS_orig, RATES_orig, CROSS_orig, mp)

    # Run generated hybrid in picard mode
    LHS_gen = {"O2": np.zeros(3), "POC_fast": np.zeros(3)}
    RHS_gen = {"O2": np.zeros(3), "POC_fast": np.zeros(3)}
    RATES_gen = {"O2": np.zeros(3), "POC_fast": np.zeros(3)}
    CROSS_gen = {"O2": [], "POC_fast": []}
    aerobic_respiration_hybrid(c, k, lim, LHS_gen, RHS_gen, RATES_gen, CROSS_gen, mp)

    # Assert exact match
    assert np.allclose(LHS_orig["O2"], LHS_gen["O2"])
    assert np.allclose(RHS_orig["O2"], RHS_gen["O2"])
    assert np.allclose(RATES_orig["O2"], RATES_gen["O2"])
    assert np.allclose(LHS_orig["POC_fast"], LHS_gen["POC_fast"])
    assert np.allclose(RATES_orig["POC_fast"], RATES_gen["POC_fast"])
