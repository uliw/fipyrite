"""Unit tests for generate_equation_system.py (Hoop 1)."""

from pathlib import Path
import pytest
from fipyrite.generate_equation_system import (
    ReactionSystemValidator,
    build_arg_parser,
    load_reactions_from_file,
    main,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SPECIES_FILE = BASE_DIR / "nbk" / "experiments" / "species.py"
CONSTANTS_FILE = BASE_DIR / "nbk" / "experiments" / "reaction_constants.py"
LIMITERS_FILE = BASE_DIR / "nbk" / "experiments" / "limiters.py"


def test_validator_loading():
    """Verify that validator loads definitions from species.py, constants, and limiters."""
    validator = ReactionSystemValidator(
        species_path=SPECIES_FILE,
        constants_path=CONSTANTS_FILE,
        limiters_path=LIMITERS_FILE,
    )
    assert "SO4" in validator.valid_species
    assert "POC_fast" in validator.valid_species
    assert "Fe2_total" in validator.valid_species
    assert "POC_fast" in validator.valid_constants
    assert "FeS_isp" in validator.valid_constants
    assert "O2_implicit" in validator.valid_limiters
    assert "SO4_implicit" in validator.valid_limiters
    assert "Fe3_diss_red_inhib" in validator.valid_limiters


def test_parse_reaction_species():
    """Test parsing of reaction strings including charges and multi-digit coefficients."""
    r, p = ReactionSystemValidator.parse_reaction_species("2 POC_fast + SO4 -> TS2")
    assert r == [(2.0, "POC_fast"), (1.0, "SO4")]
    assert p == [(1.0, "TS2")]

    r2, p2 = ReactionSystemValidator.parse_reaction_species("4 S0 + 4 H2O -> 3 H2S + SO4 + 2 H+")
    assert r2 == [(4.0, "S0"), (4.0, "H2O")]
    assert p2 == [(3.0, "H2S"), (1.0, "SO4"), (2.0, "H+")]


def test_validation_valid_reaction():
    """Verify that a properly specified reaction passes validation."""
    validator = ReactionSystemValidator(
        species_path=SPECIES_FILE,
        constants_path=CONSTANTS_FILE,
        limiters_path=LIMITERS_FILE,
    )
    rxn = {
        "reaction_name": "sulfate_reduction_fast",
        "reaction": "2 POC_fast + SO4 -> TS2",
        "k_value_name": "POC_fast",
        "limiters": {"SO4": ["O2_inhibit", "SO4_implicit", "Fe3_diss_red_inhib"]},
        "isotope_species": {"SO4": "SO4_32", "TS2": "TS2_32"},
    }
    errors = validator.validate_reaction(rxn)
    assert errors == []


def test_validation_catches_unknown_species():
    """Verify that unknown species trigger validation error."""
    validator = ReactionSystemValidator(
        species_path=SPECIES_FILE,
        constants_path=CONSTANTS_FILE,
        limiters_path=LIMITERS_FILE,
    )
    rxn = {
        "reaction_name": "bad_species_rxn",
        "reaction": "Unicorn + SO4 -> TS2",
        "k_value_name": "POC_fast",
    }
    errors = validator.validate_reaction(rxn)
    assert any("Unicorn" in e for e in errors)


def test_validation_catches_unknown_constant():
    """Verify that unknown rate constant triggers validation error."""
    validator = ReactionSystemValidator(
        species_path=SPECIES_FILE,
        constants_path=CONSTANTS_FILE,
        limiters_path=LIMITERS_FILE,
    )
    rxn = {
        "reaction_name": "bad_k_rxn",
        "reaction": "POC_fast + O2 -> CO2",
        "k_value_name": "k_magic_constant",
    }
    errors = validator.validate_reaction(rxn)
    assert any("k_magic_constant" in e for e in errors)


def test_validation_catches_unknown_limiter():
    """Verify that unknown limiter triggers validation error."""
    validator = ReactionSystemValidator(
        species_path=SPECIES_FILE,
        constants_path=CONSTANTS_FILE,
        limiters_path=LIMITERS_FILE,
    )
    rxn = {
        "reaction_name": "bad_limiter_rxn",
        "reaction": "POC_fast + O2 -> CO2",
        "k_value_name": "POC_fast",
        "limiters": {"O2": "nonexistent_limiter"},
    }
    errors = validator.validate_reaction(rxn)
    assert any("nonexistent_limiter" in e for e in errors)


def test_validation_dynamic_variables():
    """Verify that dynamic variables resolve as valid species in reaction equations."""
    validator = ReactionSystemValidator(
        species_path=SPECIES_FILE,
        constants_path=CONSTANTS_FILE,
        limiters_path=LIMITERS_FILE,
    )
    rxn = {
        "reaction_name": "pyrite_formation_fes_ts2",
        "reaction": "1 FeS + 1 HS_custom -> 1 FeS2",
        "k_value_name": "FeS_TS2",
        "dynamic_variables": {"HS_custom": "c.TS2 * mp.hs_frac"},
    }
    errors = validator.validate_reaction(rxn)
    assert errors == []


def test_cli_help_and_defaults():
    """Verify CLI argument parser configuration."""
    parser = build_arg_parser()
    args = parser.parse_args(["--validate-only"])
    assert args.validate_only is True
    assert args.input == Path("nbk/experiments/chemical_equations_new.py")
    assert args.output == Path("nbk/experiments/equations.py")
