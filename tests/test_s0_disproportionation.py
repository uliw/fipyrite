import sys
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parents[1]
experiments_dir = repo_root / "nbk" / "experiments"
if not experiments_dir.exists():
    experiments_dir = repo_root / "experiments"
src_dir = repo_root / "src"
for p in [str(experiments_dir), str(src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fipyrite.diff_lib import data_container
import reactions_new as rn


def test_s0_disproportionation_mass_and_isotope_conservation():
    """Verify stoichiometry, total S mass balance, 32S mass conservation, and delta34S."""
    VCDT = 0.044162589

    # Setup mock concentrations
    S0_tot = np.array([10.0, 5.0, 1.0])
    # Distribute S0 between 32S and 34S at VCDT
    S0_32 = S0_tot / (1.0 + VCDT)

    c = data_container({
        "S0": S0_tot,
        "S0_32": S0_32,
        "SO4": np.array([28.0, 20.0, 10.0]),
        "SO4_32": np.array([28.0, 20.0, 10.0]) / (1.0 + VCDT),
        "TS2": np.array([0.1, 0.5, 1.0]),
        "TS2_32": np.array([0.1, 0.5, 1.0]) / (1.0 + VCDT),
        "O2": np.zeros(3),
    })

    k = data_container({
        "S0_dispro": 1e-4,
    })

    lim = {
        "TS2": np.array([0.9, 0.8, 0.5]),
        "O2_inhibit": np.array([1.0, 1.0, 1.0]),
    }

    mp = data_container({
        "phi": 0.8,
        "current_dt": 86400.0,
        "isotopes": True,
        "dispro_hs_alpha": 0.993,  # -7 permil
        "dispro_SO4_hs_split": 3.0,  # 3:1 split (4 S0 -> 3 H2S + 1 SO4)
    })

    species_list = ["S0", "S0_32", "SO4", "SO4_32", "TS2", "TS2_32"]
    LHS = {s: np.zeros(3) for s in species_list}
    RHS = {s: np.zeros(3) for s in species_list}
    RATES = {s: np.zeros(3) for s in species_list}
    CROSS = {s: [] for s in species_list}

    rn.S0_disproportionation(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    # 1. Check Stoichiometric Split of total S (75% H2S, 25% SO4)
    rate_S0_consumed = -RATES["S0"]
    rate_TS2_produced = RATES["TS2"]
    rate_SO4_produced = RATES["SO4"]

    assert np.all(rate_S0_consumed > 0)
    # Total S conservation:
    np.testing.assert_allclose(rate_S0_consumed, rate_TS2_produced + rate_SO4_produced, rtol=1e-12)
    # 3:1 stoichiometric ratio:
    np.testing.assert_allclose(rate_TS2_produced / rate_S0_consumed, 0.75, rtol=1e-12)
    np.testing.assert_allclose(rate_SO4_produced / rate_S0_consumed, 0.25, rtol=1e-12)

    # 2. Check 32S Mass Conservation
    rate_S0_32_consumed = -RATES["S0_32"]
    rate_TS2_32_produced = RATES["TS2_32"]
    rate_SO4_32_produced = RATES["SO4_32"]

    assert np.all(rate_S0_32_consumed > 0)
    # 32S conservation:
    np.testing.assert_allclose(rate_S0_32_consumed, rate_TS2_32_produced + rate_SO4_32_produced, rtol=1e-12)

    # 3. Check consistency between 32S consumption and total S consumption
    # rate_S0_32_consumed must exactly equal rate_S0_consumed * (S0_32 / S0)
    expected_32_consumed = rate_S0_consumed * (c.S0_32 / c.S0)
    np.testing.assert_allclose(rate_S0_32_consumed, expected_32_consumed, rtol=1e-12)

    # 4. Check 34S conservation and fractionations
    rate_S0_34_consumed = rate_S0_consumed - rate_S0_32_consumed
    rate_TS2_34_produced = rate_TS2_produced - rate_TS2_32_produced
    rate_SO4_34_produced = rate_SO4_produced - rate_SO4_32_produced

    # 34S conservation:
    np.testing.assert_allclose(rate_S0_34_consumed, rate_TS2_34_produced + rate_SO4_34_produced, rtol=1e-12)

    # Calculate instantaneous delta34S relative to reacting S0
    R_S0 = (c.S0 - c.S0_32) / c.S0_32
    R_TS2 = rate_TS2_34_produced / rate_TS2_32_produced
    R_SO4 = rate_SO4_34_produced / rate_SO4_32_produced

    delta_TS2 = 1000.0 * (R_TS2 / R_S0 - 1.0)
    delta_SO4 = 1000.0 * (R_SO4 / R_S0 - 1.0)

    # H2S should be depleted by ~ -7.0 permil
    np.testing.assert_allclose(delta_TS2, -7.0, atol=0.05)
    # SO4 should be enriched by ~ +21.0 permil (compensating 3x fractionation)
    np.testing.assert_allclose(delta_SO4, 21.025, atol=0.05)
    # Weighted delta should be 0.0 permil
    weighted_delta = 0.75 * delta_TS2 + 0.25 * delta_SO4
    np.testing.assert_allclose(weighted_delta, 0.0, atol=0.01)


def test_s0_disproportionation_legacy_split_backward_compatibility():
    """Verify that if legacy split=0.5 is passed, it is safely mapped to 3:1 disproportionation."""
    c = data_container({
        "S0": np.array([5.0]),
        "S0_32": np.array([5.0 / 1.044]),
        "SO4": np.array([10.0]),
        "SO4_32": np.array([10.0 / 1.044]),
        "TS2": np.array([0.5]),
        "TS2_32": np.array([0.5 / 1.044]),
    })
    k = data_container({"S0_dispro": 1e-4})
    lim = {"TS2": np.array([1.0]), "O2_inhibit": np.array([1.0])}
    mp = data_container({
        "phi": 0.8,
        "current_dt": 86400.0,
        "isotopes": False,
        "dispro_SO4_hs_split": 0.5,  # legacy setting
    })

    species_list = ["S0", "SO4", "TS2"]
    LHS = {s: np.zeros(1) for s in species_list}
    RHS = {s: np.zeros(1) for s in species_list}
    RATES = {s: np.zeros(1) for s in species_list}
    CROSS = {s: [] for s in species_list}

    rn.S0_disproportionation(c, k, lim, LHS, RHS, RATES, CROSS, mp)

    # Should safely produce 75% H2S and 25% SO4
    rate_S0 = -RATES["S0"]
    np.testing.assert_allclose(RATES["TS2"] / rate_S0, 0.75, rtol=1e-12)
    np.testing.assert_allclose(RATES["SO4"] / rate_S0, 0.25, rtol=1e-12)
