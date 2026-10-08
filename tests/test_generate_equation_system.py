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
    assert args.reactions == Path("nbk/experiments/chemical_equations_new.py")
    assert args.equations == Path("nbk/experiments/equations.py")


def test_cli_new_flags():
    """Verify new CLI flags -r/--reactions, -e/--equations, -k/--kinetic-constants, -l/--limiters."""
    parser = build_arg_parser()
    args = parser.parse_args([
        "-r", "custom_rxn.py",
        "-e", "custom_eq.py",
        "-k", "custom_constants.py",
        "-l", "custom_limiters.py",
        "-s", "custom_species.py",
    ])
    assert args.reactions == Path("custom_rxn.py")
    assert args.equations == Path("custom_eq.py")
    assert args.kinetic_constants == Path("custom_constants.py")
    assert args.limiters == Path("custom_limiters.py")
    assert args.species == Path("custom_species.py")
    # Compatibility properties
    assert args.input == Path("custom_rxn.py")
    assert args.output == Path("custom_eq.py")
    assert args.constants == Path("custom_constants.py")


def test_selective_reaction_inclusion(tmp_path):
    """Verify that only the reactions specified in the reactions file are included in DEFAULT_DIAGENETIC_REACTIONS and Jacobian."""
    from fipyrite.generate_equation_system import EquationSystemGenerator

    validator = ReactionSystemValidator(
        species_path=SPECIES_FILE,
        constants_path=CONSTANTS_FILE,
        limiters_path=LIMITERS_FILE,
    )
    # Subset with only aerobic respiration fast and hs_oxidation
    reactions = [
        {
            "reaction_name": "aerobic_respiration_fast",
            "reaction": "POC_fast + O2 -> CO2",
            "k_value_name": "POC_fast",
            "limiters": {"O2": "O2_implicit"},
        },
        {
            "reaction_name": "hs_oxidation_velde",
            "reaction": "HS + 2 O2 -> SO4",
            "k_value_name": "TS2_O2",
            "dynamic_variables": {"HS": "c.TS2 * mp.hs_frac"},
            "limiters": {"O2": "O2_inhibit"},
        },
    ]

    generator = EquationSystemGenerator(reactions=reactions, validator=validator)
    code = generator.generate_code()

    # Verify DEFAULT_DIAGENETIC_REACTIONS contains aerobic_respiration and hs_oxidation_velde
    def_start = code.find("DEFAULT_DIAGENETIC_REACTIONS = [")
    def_end = code.find("DIAGENETIC_REACTIONS = DEFAULT_DIAGENETIC_REACTIONS")
    default_rxns = code[def_start:def_end]
    assert "aerobic_respiration" in default_rxns
    assert "hs_oxidation_velde" in default_rxns
    assert "FeS_precipitation_dissolution_smooth_transition" not in default_rxns
    assert "dissimilatory_iron_reduction" not in default_rxns
    assert "sulfate_reduction" not in default_rxns

    # Verify Jacobian selectively only includes active reactions
    jac_start = code.find("def compute_chemical_jacobian")
    jacobian_code = code[jac_start:]
    assert "1. Aerobic respiration" in jacobian_code
    assert "5. HS oxidation" in jacobian_code
    assert "2. Dissimilatory iron reduction" not in jacobian_code
    assert "3. Sulfate reduction" not in jacobian_code
    assert "4. FeS precipitation / dissolution" not in jacobian_code


def test_resolve_and_prepare_equations_timestamp_check(tmp_path, monkeypatch):
    """Verify resolve_and_prepare_equations regenerates on missing equations or outdated mtime."""
    import time
    from fipyrite.generate_equation_system import resolve_and_prepare_equations

    # Create dummy reaction and support files in tmp_path
    rxn_file = tmp_path / "test_exp_reactions.py"
    eq_file = tmp_path / "test_exp_equations.py"
    
    rxn_file.write_text(
        'reactions = [\n'
        '    {\n'
        '        "reaction_name": "aerobic_respiration_fast",\n'
        '        "reaction": "POC_fast + O2 -> CO2",\n'
        '        "k_value_name": "POC_fast",\n'
        '        "limiters": {"O2": "O2_implicit"},\n'
        '    },\n'
        ']\n',
        encoding="utf-8"
    )

    p_dict = {
        "experiment": "test_exp",
        "reactions": rxn_file,
        "equations": eq_file,
        "kinetic_constants": CONSTANTS_FILE,
        "limiters": LIMITERS_FILE,
        "species": SPECIES_FILE,
    }

    # 1. Equations file doesn't exist -> should generate it
    assert not eq_file.exists()
    eq_mod, r_p, e_p = resolve_and_prepare_equations(p_dict, base_dir=tmp_path)
    assert eq_file.exists()
    assert hasattr(eq_mod, "diagenetic_reactions")
    first_mtime = eq_file.stat().st_mtime

    # 2. Call again without changing reactions file -> should not regenerate (same mtime)
    eq_mod2, _, _ = resolve_and_prepare_equations(p_dict, base_dir=tmp_path)
    assert eq_file.stat().st_mtime == first_mtime

    # 3. Touch reactions file with future timestamp -> should regenerate
    time.sleep(0.05)
    rxn_file.touch()
    new_rxn_mtime = eq_file.stat().st_mtime + 2.0
    import os
    os.utime(rxn_file, (new_rxn_mtime, new_rxn_mtime))

    eq_mod3, _, _ = resolve_and_prepare_equations(p_dict, base_dir=tmp_path)
    assert eq_file.stat().st_mtime > first_mtime


def test_fes2_precipitation_ts2_generation():
    """Verify FeS2_precipitation_TS2 is generated in DEFAULT_DIAGENETIC_REACTIONS and Jacobian."""
    from fipyrite.generate_equation_system import EquationSystemGenerator

    validator = ReactionSystemValidator(
        species_path=SPECIES_FILE,
        constants_path=CONSTANTS_FILE,
        limiters_path=LIMITERS_FILE,
    )
    reactions = [
        {
            "reaction_name": "FeS2_precipitation_TS2",
            "reaction": "1 FeS + 1 HS -> 1 FeS2",
            "k_value_name": "FeS_TS2",
            "dynamic_variables": {"HS": "c.TS2 * mp.hs_frac"},
            "isotope_species": {"TS2": "TS2_32", "FeS": "FeS_32", "HS": "HS_32"},
        },
    ]

    generator = EquationSystemGenerator(reactions=reactions, validator=validator)
    code = generator.generate_code()

    # Function defined
    assert "def FeS2_precipitation_TS2" in code
    # Dispatched in DEFAULT_DIAGENETIC_REACTIONS
    def_start = code.find("DEFAULT_DIAGENETIC_REACTIONS = [")
    def_end = code.find("DIAGENETIC_REACTIONS = DEFAULT_DIAGENETIC_REACTIONS")
    default_rxns = code[def_start:def_end]
    assert "FeS2_precipitation_TS2" in default_rxns

    # Jacobian included
    jac_start = code.find("def compute_chemical_jacobian")
    jacobian_code = code[jac_start:]
    assert "10. FeS2 precipitation via TS2" in jacobian_code


def test_unrecognized_reaction_raises_value_error():
    """Verify that an unrecognized reaction name raises a ValueError instead of logging a warning."""
    from fipyrite.generate_equation_system import EquationSystemGenerator

    validator = ReactionSystemValidator(
        species_path=SPECIES_FILE,
        constants_path=CONSTANTS_FILE,
        limiters_path=LIMITERS_FILE,
    )
    reactions = [
        {
            "reaction_name": "unknown_typo_reaction",
            "reaction": "POC_fast + O2 -> CO2",
            "k_value_name": "POC_fast",
        },
    ]

    generator = EquationSystemGenerator(reactions=reactions, validator=validator)
    with pytest.raises(ValueError, match="Unable to generate reaction 'unknown_typo_reaction'"):
        generator.generate_code()


def test_missing_reactions_file_raises_filenotfound(tmp_path):
    """Verify that a missing reactions file immediately raises FileNotFoundError without silent fallback."""
    from fipyrite.generate_equation_system import resolve_and_prepare_equations

    p_dict = {
        "experiment": "non_existent_experiment",
        "reactions": None,  # Will look for non_existent_experiment_reactions.py
        "equations": None,
        "kinetic_constants": CONSTANTS_FILE,
        "limiters": LIMITERS_FILE,
        "species": SPECIES_FILE,
    }

    with pytest.raises(FileNotFoundError, match="Declarative reactions file not found"):
        resolve_and_prepare_equations(p_dict, base_dir=tmp_path)


