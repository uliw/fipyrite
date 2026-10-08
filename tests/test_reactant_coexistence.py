"""Tests for post-simulation reactant co-existence diagnostics."""

import numpy as np
import pytest
from fipyrite.diff_lib import check_reactant_coexistence, data_container


def test_reactant_coexistence_well_resolved():
    """Test that a smoothly resolved reaction front passes without warnings."""
    # 100 points over 10 cm, dz = 1 mm
    z = np.linspace(0.0005, 0.0995, 100)

    # Smooth front around z = 3 cm with width ~ 1 cm (spanning ~10 cells)
    z_front = 0.03
    width = 0.008
    o2 = 0.5 * (1.0 - np.tanh((z - z_front) / width))
    ts2 = 0.5 * (1.0 + np.tanh((z - z_front) / width))

    c = {"O2": o2, "TS2": ts2}

    results = check_reactant_coexistence(z, c, verbose=False)
    assert len(results) > 0
    sulfide_ox = next(r for r in results if r["name"] == "Sulfide Oxidation by O2")
    assert sulfide_ox["n_cells"] >= 6
    assert not sulfide_ox["is_warning"]


def test_reactant_coexistence_under_resolved_warning():
    """Test that an unresolved, single-cell collision triggers a co-existence warning."""
    # Coarse grid: dz = 1 cm, 20 cells
    z = np.arange(0.005, 0.205, 0.01)
    assert len(z) == 20

    # Very sharp step collision at cell 5
    o2 = np.zeros_like(z)
    ts2 = np.zeros_like(z)
    o2[:6] = [1.0, 0.8, 0.6, 0.4, 0.2, 0.05]
    # Remaining 15 elements: index 5 to 19 inclusive
    ts2[5:] = [0.05, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    c = data_container({"O2": o2, "TS2": ts2})

    results = check_reactant_coexistence(z, c, verbose=True)
    sulfide_ox = next(r for r in results if r["name"] == "Sulfide Oxidation by O2")
    assert sulfide_ox["is_warning"]
    assert sulfide_ox["n_cells"] <= 2
    assert sulfide_ox["recommended_dz"] < sulfide_ox["dz_peak"]
    assert len(sulfide_ox["reasons"]) > 0


def test_reactant_coexistence_custom_pairs_and_aliases():
    """Test species aliases and custom reaction definitions."""
    z = np.linspace(0.001, 0.1, 50)
    fe2 = np.exp(-z / 0.02)
    h2s = 1.0 - np.exp(-z / 0.02)

    # Aliased names: c_fe2, c_h2s
    c = {"c_fe2": fe2, "c_h2s": h2s}

    custom_pairs = [
        {
            "name": "Custom FeS",
            "sp_A": "fe2",
            "sp_B": "TS2",
            "k": 5e4,
            "D_eff": 5e-10,
        }
    ]

    results = check_reactant_coexistence(z, c, reaction_pairs=custom_pairs, verbose=False)
    assert len(results) == 1
    assert results[0]["name"] == "Custom FeS"
    assert results[0]["da_ii"] is not None
    assert results[0]["delta_phys"] is not None
