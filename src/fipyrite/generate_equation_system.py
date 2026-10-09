"""generate_equation_system.py

Standalone declarative parser and code generator for FiPyrite reaction transport models.

Translates declarative reaction specifications (e.g. chemical_equations_new.py) into
a fully assembled Python equation module (e.g. equations.py) with exact validation of:
  - Species definitions in species.py
  - Reaction constants in reaction_constants.py
  - Limiters in limiters.py

Usage:
------
    python -m fipyrite.generate_equation_system [OPTIONS]
    python src/fipyrite/generate_equation_system.py -i chemical_equations_new.py -o equations.py
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple


class ValidationError(Exception):
    """Raised when chemical reaction specifications fail validation checks."""
    pass


class ReactionSystemValidator:
    """Validates chemical reaction specifications against model definitions."""

    def __init__(
        self,
        species_path: Path,
        constants_path: Path,
        limiters_path: Path,
    ):
        self.species_path = Path(species_path)
        self.constants_path = Path(constants_path)
        self.limiters_path = Path(limiters_path)

        self.valid_species: Set[str] = set()
        self.valid_constants: Set[str] = set()
        self.valid_limiters: Set[str] = set()

        self._load_definitions()

    def _load_definitions(self) -> None:
        """Loads and extracts defined species, constants, and limiters."""
        # 1. Species
        if not self.species_path.exists():
            raise FileNotFoundError(f"Species file not found: {self.species_path}")
        spec_species = importlib.util.spec_from_file_location("species_mod", self.species_path)
        if spec_species is None or spec_species.loader is None:
            raise ImportError(f"Could not load module from {self.species_path}")
        mod_species = importlib.util.module_from_spec(spec_species)
        spec_species.loader.exec_module(mod_species)
        if hasattr(mod_species, "species") and isinstance(mod_species.species, dict):
            self.valid_species = set(mod_species.species.keys())
            self.species_defs = mod_species.species
        else:
            raise ValueError(f"No 'species' dictionary found in {self.species_path}")

        # 2. Reaction constants
        if not self.constants_path.exists():
            raise FileNotFoundError(f"Reaction constants file not found: {self.constants_path}")
        spec_constants = importlib.util.spec_from_file_location("rc_mod", self.constants_path)
        if spec_constants is None or spec_constants.loader is None:
            raise ImportError(f"Could not load module from {self.constants_path}")
        mod_constants = importlib.util.module_from_spec(spec_constants)
        spec_constants.loader.exec_module(mod_constants)

        # Attempt dynamic extraction via get_reaction_constants
        self.valid_constants = set()
        if hasattr(mod_constants, "get_reaction_constants"):
            try:
                _velde, k_vals = mod_constants.get_reaction_constants(7.5, 0.8)
                if isinstance(k_vals, dict):
                    self.valid_constants.update(k_vals.keys())
                if isinstance(_velde, dict):
                    self.valid_constants.update(_velde.keys())
            except Exception:
                pass

        # Static AST fallback to capture constants in velde dict or module variables
        try:
            rc_tree = ast.parse(self.constants_path.read_text(encoding="utf-8"))
            for node in ast.walk(rc_tree):
                if isinstance(node, ast.Dict):
                    for key in node.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            self.valid_constants.add(key.value)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.valid_constants.add(target.id)
        except Exception:
            pass

        # 3. Limiters (AST-based static extraction from limiters["..."] assignments)
        if not self.limiters_path.exists():
            raise FileNotFoundError(f"Limiters file not found: {self.limiters_path}")
        self.valid_limiters = set()
        try:
            lim_tree = ast.parse(self.limiters_path.read_text(encoding="utf-8"))
            for node in ast.walk(lim_tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Subscript):
                            if isinstance(target.value, ast.Name) and target.value.id == "limiters":
                                if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                                    self.valid_limiters.add(target.slice.value)
        except Exception as e:
            raise ValueError(f"Could not parse limiters file {self.limiters_path}: {e}")

    @staticmethod
    def parse_reaction_species(reaction_str: str) -> Tuple[List[Tuple[float, str]], List[Tuple[float, str]]]:
        """Parses a chemical reaction equation string into (reactants, products).

        Each species entry is (stoichiometric_coefficient, species_name).
        Correctly handles ions with trailing '+' or '-' (e.g. '2 H+', 'SO4--') by splitting on
        whitespace-padded ' + ' delimiters.
        """
        if "->" not in reaction_str:
            raise ValueError(f"Reaction string missing '->' delimiter: '{reaction_str}'")

        left, right = reaction_str.split("->", 1)

        def _parse_side(side_str: str) -> List[Tuple[float, str]]:
            terms = re.split(r"\s+\+\s+", side_str.strip())
            parsed = []
            for term in terms:
                term = term.strip()
                if not term:
                    continue
                parts = term.split(None, 1)
                if len(parts) == 1:
                    coeff = 1.0
                    sp = parts[0]
                else:
                    try:
                        coeff = float(parts[0])
                        sp = parts[1]
                    except ValueError:
                        coeff = 1.0
                        sp = term
                parsed.append((coeff, sp))
            return parsed

        return _parse_side(left), _parse_side(right)

    def validate_reaction(self, rxn_dict: Dict[str, Any]) -> List[str]:
        """Validates a single reaction dictionary. Returns a list of error messages (empty if valid)."""
        errors = []
        rxn_name = rxn_dict.get("reaction_name", "<unnamed_reaction>")
        rxn_str = rxn_dict.get("reaction", "")
        dyn_vars = rxn_dict.get("dynamic_variables", {}) or {}
        iso_map = rxn_dict.get("isotope_species", {}) or {}
        k_val = rxn_dict.get("k_value_name")
        limiters = rxn_dict.get("limiters")

        known_local_species = self.valid_species.union(dyn_vars.keys())

        # 1. Validate reaction equation and species names
        if not rxn_str:
            errors.append(f"[{rxn_name}] Reaction equation string is empty or missing.")
        else:
            try:
                reactants, products = self.parse_reaction_species(rxn_str)
                for coeff, sp in reactants + products:
                    if sp not in known_local_species:
                        errors.append(
                            f"[{rxn_name}] Species '{sp}' not found in species.py or dynamic_variables."
                        )
            except Exception as e:
                errors.append(f"[{rxn_name}] Failed to parse reaction equation '{rxn_str}': {e}")

        # 2. Validate kinetic constant(s)
        if k_val is not None:
            if isinstance(k_val, str):
                if k_val not in self.valid_constants:
                    errors.append(
                        f"[{rxn_name}] Rate constant '{k_val}' not found in reaction_constants.py."
                    )
            elif isinstance(k_val, dict):
                for branch, k_sub in k_val.items():
                    if k_sub not in self.valid_constants:
                        errors.append(
                            f"[{rxn_name}] Rate constant '{k_sub}' (branch '{branch}') not found in reaction_constants.py."
                        )
            else:
                errors.append(f"[{rxn_name}] Invalid k_value_name type: {type(k_val)}.")

        # 3. Validate limiters
        if limiters is not None:
            if isinstance(limiters, dict):
                for target_sp, lim_items in limiters.items():
                    if target_sp not in known_local_species and target_sp != "global":
                        errors.append(
                            f"[{rxn_name}] Limiter target species '{target_sp}' not defined in species.py or dynamic_variables."
                        )
                    item_list = [lim_items] if isinstance(lim_items, str) else lim_items
                    for lim_name in item_list:
                        if lim_name not in self.valid_limiters:
                            errors.append(
                                f"[{rxn_name}] Limiter '{lim_name}' not defined in limiters.py."
                            )
            elif isinstance(limiters, list):
                for lim_name in limiters:
                    if lim_name not in self.valid_limiters:
                        errors.append(
                            f"[{rxn_name}] Limiter '{lim_name}' not defined in limiters.py."
                        )
            elif isinstance(limiters, str):
                if limiters not in self.valid_limiters:
                    errors.append(
                        f"[{rxn_name}] Limiter '{limiters}' not defined in limiters.py."
                    )
            else:
                errors.append(f"[{rxn_name}] Invalid limiters format: {type(limiters)}.")

        # 4. Validate isotope mapping
        if iso_map:
            if not isinstance(iso_map, dict):
                errors.append(f"[{rxn_name}] isotope_species must be a dictionary.")
            else:
                for base_sp, iso_sp in iso_map.items():
                    if base_sp not in known_local_species:
                        errors.append(
                            f"[{rxn_name}] Base isotope species '{base_sp}' not defined in species.py or dynamic_variables."
                        )
                    if iso_sp not in known_local_species:
                        errors.append(
                            f"[{rxn_name}] Isotope species '{iso_sp}' not defined in species.py or dynamic_variables."
                        )

        # 5. Validate dynamic variables expressions (ensure valid Python syntax)
        for var_name, expr in dyn_vars.items():
            try:
                ast.parse(expr)
            except SyntaxError as se:
                errors.append(
                    f"[{rxn_name}] Invalid Python syntax in dynamic variable '{var_name}': '{expr}' ({se})."
                )

        return errors

    def validate_reactions(self, reactions: List[Dict[str, Any]]) -> List[str]:
        """Validates all reactions in the specification list."""
        all_errors = []
        for rxn_dict in reactions:
            errs = self.validate_reaction(rxn_dict)
            all_errors.extend(errs)
        return all_errors


def load_reactions_from_file(input_path: Path) -> List[Dict[str, Any]]:
    """Loads the reactions list from a chemical equations Python or JSON file."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input equations file not found: {input_path}")

    if input_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("chem_eq_mod", input_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {input_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "reactions") or not isinstance(mod.reactions, list):
            raise ValueError(f"Input file {input_path} must define a 'reactions' list.")
        return mod.reactions
    elif input_path.suffix == ".json":
        import json
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "reactions" in data:
                return data["reactions"]
            raise ValueError(f"JSON input file {input_path} must be a list of reactions or contain a 'reactions' key.")
    else:
        raise ValueError(f"Unsupported input file format '{input_path.suffix}'. Expected .py or .json.")


class FiPyriteArgumentParser(argparse.ArgumentParser):
    """ArgumentParser subclass ensuring backward compatibility aliases on parsed arguments."""

    def parse_args(self, args=None, namespace=None):
        ns = super().parse_args(args=args, namespace=namespace)
        if not hasattr(ns, "input") or getattr(ns, "input") is None:
            setattr(ns, "input", getattr(ns, "reactions", None))
        if not hasattr(ns, "output") or getattr(ns, "output") is None:
            setattr(ns, "output", getattr(ns, "equations", None))
        if not hasattr(ns, "constants") or getattr(ns, "constants") is None:
            setattr(ns, "constants", getattr(ns, "kinetic_constants", None))
        return ns


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line argument parser with full help descriptions."""
    parser = FiPyriteArgumentParser(
        prog="generate_equation_system",
        description=(
            "FiPyrite Declarative Equation System Generator:\n"
            "Validates declarative chemical reaction specifications (e.g. chemical_equations_new.py) "
            "against model species, reaction constants, and limiters, and generates an assembled "
            "reaction equations module (e.g. equations.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-r", "--reactions", "-i", "--input",
        dest="reactions",
        type=Path,
        default=Path("nbk/experiments/chemical_equations_new.py"),
        help="Path to declarative reaction definitions file (default: nbk/experiments/chemical_equations_new.py)",
    )
    parser.add_argument(
        "-e", "--equations", "-o", "--output",
        dest="equations",
        type=Path,
        default=Path("nbk/experiments/equations.py"),
        help="Path to output generated equations module (default: nbk/experiments/equations.py)",
    )
    parser.add_argument(
        "-s", "--species",
        type=Path,
        default=Path("nbk/experiments/species.py"),
        help="Path to model species definitions file (default: nbk/experiments/species.py)",
    )
    parser.add_argument(
        "-k", "--kinetic-constants", "--constants",
        dest="kinetic_constants",
        type=Path,
        default=Path("nbk/experiments/reaction_constants.py"),
        help="Path to reaction constants file (default: nbk/experiments/reaction_constants.py)",
    )
    parser.add_argument(
        "-l", "--limiters",
        type=Path,
        default=Path("nbk/experiments/limiters.py"),
        help="Path to limiters definition file (default: nbk/experiments/limiters.py)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Perform cross-reference validation of species, constants, and limiters without writing output.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed validation and parsing diagnostic output.",
    )

    return parser


def generate_equations(
    reactions_path: str | Path,
    equations_path: str | Path,
    kinetic_constants_path: str | Path = Path("nbk/experiments/reaction_constants.py"),
    limiters_path: str | Path = Path("nbk/experiments/limiters.py"),
    species_path: str | Path = Path("nbk/experiments/species.py"),
    verbose: bool = False,
    validate_only: bool = False,
) -> Path:
    """Validates and compiles declarative reactions into an assembled equations module.

    Parameters
    ----------
    reactions_path : str | Path
        Path to declarative reactions definitions file.
    equations_path : str | Path
        Path where the generated equations module will be written.
    kinetic_constants_path : str | Path
        Path to reaction constants definitions file.
    limiters_path : str | Path
        Path to limiters definition file.
    species_path : str | Path
        Path to model species definitions file.
    verbose : bool
        If True, prints diagnostic output.
    validate_only : bool
        If True, performs cross-validation without writing output.

    Returns
    -------
    Path
        Path to the output equations module.
    """
    reactions_p = Path(reactions_path).resolve()
    equations_p = Path(equations_path).resolve()
    constants_p = Path(kinetic_constants_path).resolve()
    limiters_p = Path(limiters_path).resolve()
    species_p = Path(species_path).resolve()

    if verbose:
        print(f"[FiPyrite] Loading reactions from: {reactions_p}")
        print(f"[FiPyrite] Using species definitions: {species_p}")
        print(f"[FiPyrite] Using reaction constants:   {constants_p}")
        print(f"[FiPyrite] Using limiters definitions: {limiters_p}")

    # 1. Load validator
    validator = ReactionSystemValidator(
        species_path=species_p,
        constants_path=constants_p,
        limiters_path=limiters_p,
    )

    # 2. Load input reactions
    reactions = load_reactions_from_file(reactions_p)

    # 3. Validate
    errors = validator.validate_reactions(reactions)
    if errors:
        err_msg = f"Validation failed for {reactions_p} with {len(errors)} error(s):\n" + "\n".join(f"  • {e}" for e in errors)
        raise ValidationError(err_msg)

    if verbose:
        print(f"[FiPyrite] All {len(reactions)} reactions validated successfully!")

    if validate_only:
        return equations_p

    # 4. Code Generation
    generator = EquationSystemGenerator(reactions=reactions, validator=validator)
    generated_code = generator.generate_code()

    equations_p.parent.mkdir(parents=True, exist_ok=True)
    with open(equations_p, "w", encoding="utf-8") as f:
        f.write(generated_code)

    if verbose:
        print(f"[FiPyrite] Successfully wrote equations module to: {equations_p}")

    return equations_p


def resolve_and_prepare_equations(
    p_dict: Dict[str, Any],
    experiment_name: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[Any, Path, Path]:
    """Resolves reaction and equation files from p_dict, regenerates equations if outdated,
    and dynamically imports the equations module.

    Resolution scheme:
      reactions: defaults to f"{experiment}_reactions.py"
      equations: defaults to f"{experiment}_equations.py"
      kinetic_constants: defaults to "reaction_constants.py"
      limiters: defaults to "limiters.py"

    Parameters
    ----------
    p_dict : dict
        Parameter dictionary passed to the simulation.
    experiment_name : str, optional
        Name of the experiment stem. If None, derived from p_dict["experiment"].
    base_dir : Path, optional
        Base directory to search for experiment files. Defaults to cwd or experiment file parent.

    Returns
    -------
    Tuple[module, Path, Path]
        (loaded_equations_module, resolved_reactions_path, resolved_equations_path)
    """
    exp = p_dict.get("experiment") or experiment_name or "pyrite"

    if base_dir is None:
        if "__file__" in p_dict and p_dict["__file__"]:
            base_dir = Path(p_dict["__file__"]).resolve().parent
        else:
            base_dir = Path.cwd()

    def _resolve(val: Any, default_name: str) -> Path:
        filename = default_name if (val is None or val == "file_name") else str(val)
        p = Path(filename)
        if p.is_absolute() and p.exists():
            return p
        # Check relative to base_dir
        if (base_dir / p).exists():
            return (base_dir / p).resolve()
        # Check relative to cwd
        if (Path.cwd() / p).exists():
            return (Path.cwd() / p).resolve()
        # Check in nbk/experiments
        if (Path.cwd() / "nbk" / "experiments" / p).exists():
            return (Path.cwd() / "nbk" / "experiments" / p).resolve()
        return (base_dir / p).resolve()

    # Apply defaults if not specified or None
    rxn_val = p_dict.get("reactions")
    eq_val = p_dict.get("equations")
    kc_val = p_dict.get("kinetic_constants")
    lim_val = p_dict.get("limiters")
    sp_val = p_dict.get("species")

    reactions_path = _resolve(rxn_val, f"{exp}_reactions.py")
    equations_path = _resolve(eq_val, f"{exp}_equations.py")
    constants_path = _resolve(kc_val, "reaction_constants.py")
    limiters_path = _resolve(lim_val, "limiters.py")
    species_path = _resolve(sp_val, "species.py")

    if not reactions_path.exists():
        raise FileNotFoundError(
            f"Declarative reactions file not found: '{reactions_path}'. "
            f"Please specify a valid 'reactions' file in p_dict or create '{exp}_reactions.py'."
        )

    # Determine whether regeneration is needed based on mtime timestamps
    needs_regen = False
    reason = ""
    if not equations_path.exists():
        needs_regen = True
        reason = f"{equations_path.name} does not exist"
    elif "def diagenetic_reactions" not in equations_path.read_text(encoding="utf-8", errors="ignore"):
        needs_regen = True
        reason = f"{equations_path.name} is missing diagenetic_reactions function"
    elif reactions_path.stat().st_mtime > equations_path.stat().st_mtime:
        needs_regen = True
        reason = f"{reactions_path.name} is newer than {equations_path.name}"
    elif constants_path.exists() and constants_path.stat().st_mtime > equations_path.stat().st_mtime:
        needs_regen = True
        reason = f"{constants_path.name} is newer than {equations_path.name}"
    elif limiters_path.exists() and limiters_path.stat().st_mtime > equations_path.stat().st_mtime:
        needs_regen = True
        reason = f"{limiters_path.name} is newer than {equations_path.name}"
    elif species_path.exists() and species_path.stat().st_mtime > equations_path.stat().st_mtime:
        needs_regen = True
        reason = f"{species_path.name} is newer than {equations_path.name}"

    if needs_regen:
        print(f"[FiPyrite] Regenerating {equations_path.name} ({reason})...")
        generate_equations(
            reactions_path=reactions_path,
            equations_path=equations_path,
            kinetic_constants_path=constants_path,
            limiters_path=limiters_path,
            species_path=species_path,
            verbose=False,
        )

    # Ensure parent directories of equations and limiters are on sys.path for import resolution
    for p_dir in [str(equations_path.parent), str(limiters_path.parent)]:
        if p_dir not in sys.path:
            sys.path.insert(0, p_dir)

    # Dynamically load the generated equations module
    mod_name = equations_path.stem
    spec = importlib.util.spec_from_file_location(mod_name, equations_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load equations module from {equations_path}")
    eq_mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = eq_mod
    spec.loader.exec_module(eq_mod)

    # Dynamically load reaction constants function if reaction_constants is not already a callable
    if not callable(p_dict.get("reaction_constants")) and constants_path.exists():
        c_spec = importlib.util.spec_from_file_location(constants_path.stem, constants_path)
        if c_spec and c_spec.loader:
            c_mod = importlib.util.module_from_spec(c_spec)
            c_spec.loader.exec_module(c_mod)
            if hasattr(c_mod, "get_reaction_constants"):
                p_dict["reaction_constants"] = c_mod.get_reaction_constants

    # Record loaded module and paths in p_dict
    p_dict["reactions_module"] = eq_mod
    p_dict["reactions_path"] = reactions_path
    p_dict["equations_path"] = equations_path

    return eq_mod, reactions_path, equations_path


class EquationSystemGenerator:
    """Generates an assembled equations.py module with SymPy analytical Jacobian."""

    def __init__(self, reactions: List[Dict[str, Any]], validator: ReactionSystemValidator):
        self.reactions = reactions
        self.validator = validator

    @staticmethod
    def classify_reaction(r: Dict[str, Any]) -> str:
        """Classifies a chemical reaction based on its stoichiometry, reactants, and products,
        completely independent of the user-assigned reaction name.
        """
        r_str = r.get("reaction", "")
        if "->" not in r_str:
            return "unknown"
        try:
            reactants, products = ReactionSystemValidator.parse_reaction_species(r_str)
        except Exception:
            return "unknown"

        r_sp = [sp for _, sp in reactants]
        p_sp = [sp for _, sp in products]
        k_val = r.get("k_value_name")

        # 1. Reversible precipitation / dissolution (e.g. FeS)
        if isinstance(k_val, dict) and "precipitation" in k_val:
            return "FeS_precipitation_dissolution"

        # 2. Elemental sulfur disproportionation (S0 -> TS2 + SO4)
        if "S0" in r_sp and "TS2" in p_sp and "SO4" in p_sp:
            return "elemental_sulfur_disproportionation"

        # 3. Aerobic respiration (POC + O2 -> CO2)
        if "O2" in r_sp and any("POC" in sp for sp in r_sp):
            return "aerobic_respiration"

        # 4. Dissimilatory iron reduction (POC + Fe3 -> Fe2_total)
        if "Fe3" in r_sp and any("POC" in sp for sp in r_sp):
            return "dissimilatory_iron_reduction"

        # 5. Sulfate reduction (POC + SO4 -> TS2)
        if "SO4" in r_sp and any("POC" in sp for sp in r_sp):
            return "sulfate_reduction"

        # 6. Sulfide oxidation (HS + O2 -> SO4)
        if any(sp in ("HS", "TS2") for sp in r_sp) and "O2" in r_sp and "SO4" in p_sp:
            return "hs_oxidation"

        # 7. Sulfide-mediated iron reduction
        if any(sp in ("HS", "TS2") for sp in r_sp) and "Fe3" in r_sp:
            if "SO4" in p_sp:
                return "sulfide_mediated_iron_reduction_velde"
            return "sulfide_mediated_iron_reduction"

        # 8. Fe2 oxidation (Fe2 + O2 -> Fe3)
        if any("Fe2" in sp for sp in r_sp) and "O2" in r_sp and "Fe3" in p_sp:
            return "Fe2_oxidation"

        # 9. FeS oxidation (FeS + O2 -> Fe3 + SO4)
        if "FeS" in r_sp and "O2" in r_sp and ("Fe3" in p_sp or "SO4" in p_sp):
            return "FeS_oxidation"

        # 10. FeS2 oxidation (FeS2 + O2 -> Fe3 + SO4)
        if "FeS2" in r_sp and "O2" in r_sp:
            return "FeS2_oxidation"

        # 11. Pyrite precipitation via HS/TS2 (FeS + HS -> FeS2)
        if "FeS" in r_sp and any(sp in ("HS", "TS2") for sp in r_sp) and "FeS2" in p_sp:
            return "FeS2_precipitation_TS2"

        # 12. Pyrite formation via S0 (FeS + S0 -> FeS2)
        if "FeS" in r_sp and "S0" in r_sp and "FeS2" in p_sp:
            return "pyrite_formation_fes_s0"

        # 13. Elemental sulfur oxidation (S0 + O2 -> SO4)
        if "S0" in r_sp and "O2" in r_sp and "SO4" in p_sp:
            return "elemental_sulfur_oxidation"

        return "unknown"

    def generate_code(self) -> str:
        """Generates complete Python source for equations.py."""
        code = [
            '"""equations.py',
            '',
            'Auto-generated reaction equations system module for FiPyrite.',
            'Generated by generate_equation_system.py with analytical SymPy Jacobian support.',
            '"""',
            '',
            'from __future__ import annotations',
            'import numpy as np',
            'from fipyrite.diff_lib import (',
            '    add_coupled_reaction,',
            '    add_implicit_sink,',
            '    add_implicit_coupling_new,',
            '    calculate_fractionated_coeff_32,',
            '    partition_equilibrium_isotope_32,',
            '    data_container,',
            ')',
            'from limiters import get_limiters',
            '',
            '# -----------------------------------------------------------------------------',
            '# Individual Reaction Functions',
            '# -----------------------------------------------------------------------------',
            '',
            'def aerobic_respiration(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
            '    """Reaction: POC + O2 -> CO2"""',
            '    has_solid = True',
            '    poc_species = k_val.get("poc_species", "POC_fast") if isinstance(k_val, dict) else "POC_fast"',
            '    poc_k_name = k_val.get("poc_k", "POC_fast") if isinstance(k_val, dict) else "POC_fast"',
            '    k_num = mp.k.get(poc_k_name) if hasattr(mp, "k") else getattr(k_val, poc_k_name, 0.0)',
            '    poc_var = getattr(c, poc_species)',
            '    lim_O2 = lim["O2_implicit"]',
            '    rate_base = k_num * poc_var * c.O2 * lim_O2',
            '    ratio = getattr(mp, "POC_O2_ratio", 1.0)',
            '    coeff_O2 = ratio * k_num * poc_var * lim_O2',
            '    add_implicit_sink(LHS, RATES, "O2", coeff_O2, ratio * rate_base, mp=mp, has_solid=has_solid, c=c)',
            '    coeff_POC = k_num * 1.0 * c.O2 * lim_O2',
            '    add_implicit_sink(LHS, RATES, poc_species, coeff_POC, rate_base, mp=mp, has_solid=has_solid, c=c)',
            '',
            'def dissimilatory_iron_reduction(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
            '    """Reaction: POC + 4 Fe3 -> 4 Fe2_total"""',
            '    has_solid = True',
            '    poc_species = k_val.get("poc_species", "POC_fast") if isinstance(k_val, dict) else "POC_fast"',
            '    poc_k_name = k_val.get("poc_k", "POC_fast") if isinstance(k_val, dict) else "POC_fast"',
            '    k_num = mp.k.get(poc_k_name) if hasattr(mp, "k") else getattr(k_val, poc_k_name, 0.0)',
            '    poc_var = getattr(c, poc_species)',
            '    fe3_lim = lim["Fe3_diss_red_implicit"]',
            '    o2_inhib = lim["O2_inhibit"]',
            '    rate_base = k_num * poc_var * c.Fe3 * o2_inhib * fe3_lim',
            '    coeff_master = k_num * poc_var * 1.0 * o2_inhib * fe3_lim',
            '    add_coupled_reaction(',
            '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
            '        master_species={"Fe3": 4}, reactants={poc_species: 1}, products={"Fe2_total": 4},',
            '        coeff_master=coeff_master, rate_master=rate_base,',
            '        has_solid=has_solid, reaction_name=f"dissimilatory_iron_reduction_{poc_species}",',
            '        ref_species="POC", stoich_ref=1.0,',
            '    )',
            '    coeff_POC = k_num * 1.0 * c.Fe3 * o2_inhib * fe3_lim',
            '    add_implicit_sink(LHS, RATES, poc_species, coeff_POC, rate_base, mp=mp, has_solid=has_solid, c=c)',
            '',
            'def sulfate_reduction(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
            '    """Reaction: 2 POC + SO4 -> TS2"""',
            '    has_solid = True',
            '    poc_species = k_val.get("poc_species", "POC_fast") if isinstance(k_val, dict) else "POC_fast"',
            '    poc_k_name = k_val.get("poc_k", "POC_fast") if isinstance(k_val, dict) else "POC_fast"',
            '    k_num = mp.k.get(poc_k_name) if hasattr(mp, "k") else getattr(k_val, poc_k_name, 0.0)',
            '    poc_var = getattr(c, poc_species)',
            '    so4_lim = lim["SO4_implicit"]',
            '    o2_inhib = lim["O2_inhibit"]',
            '    fe3_inhib = lim["Fe3_diss_red_inhib"]',
            '    rate_base = k_num * poc_var * c.SO4 * o2_inhib * so4_lim * fe3_inhib',
            '    coeff_master = k_num * poc_var * 1.0 * o2_inhib * so4_lim * fe3_inhib',
            '    add_coupled_reaction(',
            '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
            '        master_species={"SO4": 1}, reactants={poc_species: 2}, products={"TS2": 1},',
            '        coeff_master=coeff_master, rate_master=rate_base,',
            '        has_solid=has_solid, reaction_name=f"sulfate_reduction_{poc_species}",',
            '        ref_species="POC", stoich_ref=2.0,',
            '    )',
            '    coeff_POC = k_num * 1.0 * c.SO4 * o2_inhib * so4_lim * fe3_inhib',
            '    add_implicit_sink(LHS, RATES, poc_species, coeff_POC, rate_base, mp=mp, has_solid=has_solid, c=c)',
            '    if getattr(mp, "isotopes", False):',
            '        alpha = 1.0 + (mp.msr_alpha - 1.0) * lim["SO4_alpha_explicit"]',
            '        coeff_master_32 = calculate_fractionated_coeff_32(',
            '            coeff_master, c.SO4, c.SO4_32, alpha, eps=1e-30',
            '        )',
            '        add_coupled_reaction(',
            '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
            '            master_species={"SO4_32": 1}, reactants={}, products={"TS2_32": 1},',
            '            coeff_master=coeff_master_32, rate_master=coeff_master_32 * c.SO4_32,',
            '            has_solid=has_solid, reaction_name=f"sulfate_reduction_32_{poc_species}",',
            '            ref_species="POC", stoich_ref=2.0,',
            '        )',
            '',
            'def hs_oxidation_velde(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
            '    """Reaction: HS + 2 O2 -> SO4"""',
            '    has_solid = False',
            '    k_num = mp.k.get("TS2_O2") if hasattr(mp, "k") else getattr(k_val, "TS2_O2", 0.0)',
            '    o2_lim = lim["O2_implicit_TS2"]',
            '    hs_conc = c.TS2 * mp.hs_frac',
            '    rate_base = k_num * hs_conc * c.O2 * o2_lim',
            '    coeff_master = k_num * mp.hs_frac * c.O2 * o2_lim',
            '    add_coupled_reaction(',
            '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
            '        master_species={"TS2": 1}, reactants={}, products={"SO4": 1},',
            '        coeff_master=coeff_master, rate_master=rate_base,',
            '        has_solid=has_solid, reaction_name="hs_oxidation_velde",',
            '        ref_species="TS2", stoich_ref=1.0,',
            '    )',
            '    coeff_O2 = 2.0 * k_num * hs_conc * o2_lim',
            '    add_implicit_sink(LHS, RATES, "O2", coeff_O2, 2.0 * rate_base, mp=mp, has_solid=has_solid, c=c)',
            '    if getattr(mp, "isotopes", False):',
            '        alpha = 1.0 + (mp.TS2_O2_alpha - 1.0) * lim["TS2_alpha_explicit"]',
            '        coeff_master_32 = calculate_fractionated_coeff_32(',
            '            coeff_master, c.TS2, c.TS2_32, alpha, eps=1e-30',
            '        )',
            '        add_coupled_reaction(',
            '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
            '            master_species={"TS2_32": 1}, reactants={}, products={"SO4_32": 1},',
            '            coeff_master=coeff_master_32, rate_master=coeff_master_32 * c.TS2_32,',
            '            has_solid=has_solid, reaction_name="hs_oxidation_velde_32",',
            '            ref_species="TS2", stoich_ref=1.0,',
            '        )',
            '',
            'def Fe2_oxidation(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
            '    """Reaction: 4 Fe2_total + O2 -> 4 Fe3"""',
            '    has_solid = False',
            '    k_num = mp.k.get("Fe2_O2") if hasattr(mp, "k") else getattr(k_val, "Fe2_O2", 0.0)',
            '    rate_base = k_num * c.Fe2_total * c.O2',
            '    coeff_master = k_num * 1.0 * c.O2',
            '    add_coupled_reaction(',
            '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
            '        master_species={"Fe2_total": 4}, reactants={}, products={"Fe3": 4},',
            '        coeff_master=coeff_master, rate_master=rate_base,',
            '        has_solid=has_solid, reaction_name="Fe2_oxidation",',
            '        ref_species="Fe2_total", stoich_ref=4.0,',
            '    )',
            '    coeff_O2 = (1.0 / 4.0) * k_num * c.Fe2_total * 1.0',
            '    add_implicit_sink(LHS, RATES, "O2", coeff_O2, 0.25 * rate_base, mp=mp, has_solid=has_solid, c=c)',
            '',
            'def FeS_oxidation(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
            '    """Reaction: 4 FeS + 9 O2 -> 4 Fe3 + 4 SO4"""',
            '    has_solid = True',
            '    k_num = mp.k.get("FeS_O2") if hasattr(mp, "k") else getattr(k_val, "FeS_O2", 0.0)',
            '    rate_base = k_num * c.FeS * c.O2',
            '    coeff_master = k_num * 1.0 * c.O2',
            '    add_coupled_reaction(',
            '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
            '        master_species={"FeS": 4}, reactants={}, products={"Fe3": 4, "SO4": 4},',
            '        coeff_master=coeff_master, rate_master=rate_base,',
            '        has_solid=has_solid, reaction_name="FeS_oxidation",',
            '        ref_species="FeS", stoich_ref=4.0,',
            '    )',
            '    coeff_O2 = (9.0 / 4.0) * k_num * c.FeS * 1.0',
            '    add_implicit_sink(LHS, RATES, "O2", coeff_O2, 2.25 * rate_base, mp=mp, has_solid=has_solid, c=c)',
            '    if getattr(mp, "isotopes", False):',
            '        coeff_FeS_32 = k_num * c.O2',
            '        add_coupled_reaction(',
            '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
            '            master_species={"FeS_32": 1}, reactants={}, products={"SO4_32": 1},',
            '            coeff_master=coeff_FeS_32, rate_master=coeff_FeS_32 * c.FeS_32,',
            '            has_solid=has_solid, reaction_name="FeS_oxidation_32",',
            '            ref_species="FeS", stoich_ref=1.0,',
            '        )',
            '',
            'def sulfide_mediated_iron_reduction_velde(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
            '    """Reaction: HS + 8 Fe3 -> SO4 + 8 Fe2_total"""',
            '    has_solid = True',
            '    k_num = mp.k.get("Fe3_hs") if hasattr(mp, "k") else getattr(k_val, "Fe3_hs", 0.0)',
            '    hs_conc = c.TS2 * mp.hs_frac',
            '    fe3_lim = lim["Fe3_implicit"]',
            '    o2_inhib = lim["O2_inhibit"]',
            '    rate_base = k_num * hs_conc * c.Fe3 * o2_inhib * fe3_lim',
            '    coeff_master = k_num * c.Fe3 * 1.0 * mp.hs_frac * o2_inhib * fe3_lim',
            '    add_coupled_reaction(',
            '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
            '        master_species={"TS2": 1}, reactants={"Fe3": 8}, products={"Fe2_total": 8, "SO4": 1},',
            '        coeff_master=coeff_master, rate_master=rate_base,',
            '        has_solid=has_solid, reaction_name="sulfide_mediated_iron_reduction_velde",',
            '        ref_species="Fe3", stoich_ref=8.0,',
            '    )',
            '    if getattr(mp, "isotopes", False):',
            '        coeff_master_32 = coeff_master',
            '        add_coupled_reaction(',
            '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
            '            master_species={"TS2_32": 1}, reactants={}, products={"SO4_32": 1},',
            '            coeff_master=coeff_master_32, rate_master=coeff_master_32 * c.TS2_32,',
            '            has_solid=has_solid, reaction_name="sulfide_mediated_iron_reduction_velde_32",',
            '            ref_species="Fe3", stoich_ref=8.0,',
            '        )',
            '',
            'def FeS_precipitation_dissolution_smooth_transition(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
            '    """Reaction: Fe2_liq + HS -> FeS with C1 smooth transition across Omega = 1."""',
            '    import numpy as np',
            '    phi = mp.phi',
            '    has_solid = True',
            '    has_solid_prec = False',
            '    Fe2_val = np.maximum(np.asarray(c.Fe2_total.value if hasattr(c.Fe2_total, "value") else c.Fe2_total), 1e-20)',
            '    TS2_val = np.maximum(np.asarray(c.TS2.value if hasattr(c.TS2, "value") else c.TS2), 1e-20)',
            '    FeS_val = np.maximum(np.asarray(c.FeS.value if hasattr(c.FeS, "value") else c.FeS), 1e-20)',
            '    Fe2_pw = Fe2_val * mp.Fe2_diss',
            '    hs_val = TS2_val * mp.hs_frac',
            '    k_Hplus = getattr(k_val, "Hplus", getattr(mp, "Hplus", 10**-7.5))',
            '    k_FeS_sp = getattr(k_val, "FeS_sp", getattr(mp, "FeS_sp", 10**-3.5))',
            '    k_FeS_isp = getattr(k_val, "FeS_isp", getattr(mp, "FeS_isp", 1.0))',
            '    k_FeS_isd = getattr(k_val, "FeS_isd", getattr(mp, "FeS_isd", 0.3))',
            '    omega_den = k_Hplus * k_FeS_sp + 1e-30',
            '    omega = Fe2_pw * hs_val / omega_den',
            '    is_prec = np.where(omega >= 1.0, 1.0, 0.0)',
            '    is_diss = np.where(omega < 1.0, 1.0, 0.0)',
            '    Km = 0.5',
            '    epsilon = float(getattr(mp, "fes_smooth_epsilon", 0.05))',
            '    k_prec_eff = k_FeS_isp * is_prec',
            '    df = np.maximum(omega - 1.0, 0.0)',
            '    mm_factor_prec = df / (Km + df)',
            '    u_prec = np.minimum(df / (epsilon + 1e-30), 1.0)',
            '    smooth_ramp_prec = (3.0 * u_prec**2 - 2.0 * u_prec**3)',
            '    R_prec = k_prec_eff * mm_factor_prec * smooth_ramp_prec',
            '    w_Fe = TS2_val**2 / (Fe2_val**2 + TS2_val**2 + 1e-30)',
            '    w_TS2 = 1.0 - w_Fe',
            '    prec_coeff_Fe2 = (w_Fe * R_prec) / Fe2_val',
            '    prec_rate_Fe2 = prec_coeff_Fe2 * Fe2_val',
            '    prec_coeff_TS2 = (w_TS2 * R_prec) / TS2_val',
            '    prec_rate_TS2 = prec_coeff_TS2 * TS2_val',
            '    add_implicit_coupling_new(',
            '        CROSS, RATES, LHS, target_species="FeS", source_species="Fe2_total",',
            '        coeff=prec_coeff_Fe2, rate=prec_rate_Fe2, mp=mp, has_solid=has_solid_prec,',
            '        add_lhs_sink=False, stoich_ratio=1.0,',
            '    )',
            '    add_implicit_coupling_new(',
            '        CROSS, RATES, LHS, target_species="FeS", source_species="TS2",',
            '        coeff=prec_coeff_TS2, rate=prec_rate_TS2, mp=mp, has_solid=has_solid_prec,',
            '        add_lhs_sink=False, stoich_ratio=1.0,',
            '    )',
            '    add_implicit_sink(LHS, RATES, "Fe2_total", prec_coeff_Fe2, prec_rate_Fe2, mp=mp, has_solid=has_solid_prec)',
            '    add_implicit_sink(LHS, RATES, "TS2", prec_coeff_TS2, prec_rate_TS2, mp=mp, has_solid=has_solid_prec)',
            '    CROSS["Fe2_total"].append(("TS2", -prec_coeff_TS2 * phi))',
            '    RATES["Fe2_total"] -= prec_rate_TS2 * phi',
            '    CROSS["TS2"].append(("Fe2_total", -prec_coeff_Fe2 * phi))',
            '    RATES["TS2"] -= prec_rate_Fe2 * phi',
            '    k_diss_eff = k_FeS_isd * is_diss',
            '    us = np.maximum(1.0 - omega, 0.0)',
            '    mm_factor_diss = us / (Km + us)',
            '    u_diss = np.minimum(us / (epsilon + 1e-30), 1.0)',
            '    smooth_ramp_diss = (3.0 * u_diss**2 - 2.0 * u_diss**3)',
            '    diss_coeff_FeS = k_diss_eff * mm_factor_diss * smooth_ramp_diss',
            '    diss_rate_FeS = diss_coeff_FeS * FeS_val',
            '    add_implicit_coupling_new(',
            '        CROSS, RATES, LHS, target_species="Fe2_total", source_species="FeS",',
            '        coeff=diss_coeff_FeS, rate=diss_rate_FeS, mp=mp, has_solid=has_solid,',
            '        add_lhs_sink=True, stoich_ratio=1.0,',
            '    )',
            '    add_implicit_coupling_new(',
            '        CROSS, RATES, LHS, target_species="TS2", source_species="FeS",',
            '        coeff=diss_coeff_FeS, rate=diss_rate_FeS, mp=mp, has_solid=has_solid,',
            '        add_lhs_sink=False, stoich_ratio=1.0,',
            '    )',
            '    if getattr(mp, "isotopes", False):',
            '        hs_32 = partition_equilibrium_isotope_32(',
            '            c.TS2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha',
            '        )',
            '        hs_val_np = np.asarray(hs_val)',
            '        hs_32_val = np.asarray(hs_32)',
            '        TS2_val_np = np.asarray(c.TS2.value if hasattr(c.TS2, "value") else c.TS2)',
            '        TS2_32_val = np.asarray(c.TS2_32.value if hasattr(c.TS2_32, "value") else c.TS2_32)',
            '        f32_default = 1.0 / (1.0 + mp.VCDT)',
            '        f32_hs = np.where(hs_val_np > 1e-6, hs_32_val / (hs_val_np + 1e-30), f32_default)',
            '        f32_hs = np.clip(f32_hs, 0.5, 1.5)',
            '        f32_TS2 = np.where(TS2_val_np > 1e-6, TS2_32_val / (TS2_val_np + 1e-30), f32_default)',
            '        f32_TS2 = np.clip(f32_TS2, 0.5, 1.5)',
            '        ratio_hs_ts2 = np.where(f32_TS2 > 1e-10, f32_hs / f32_TS2, 1.0)',
            '        prec_coeff_TS2_32 = prec_coeff_TS2 * ratio_hs_ts2',
            '        add_implicit_coupling_new(',
            '            CROSS, RATES, LHS, target_species="FeS_32", source_species="TS2_32",',
            '            coeff=prec_coeff_TS2_32, rate=prec_rate_TS2 * f32_hs, mp=mp, has_solid=has_solid_prec,',
            '            add_lhs_sink=False, stoich_ratio=1.0,',
            '        )',
            '        add_implicit_coupling_new(',
            '            CROSS, RATES, LHS, target_species="FeS_32", source_species="Fe2_total",',
            '            coeff=prec_coeff_Fe2 * f32_hs, rate=prec_rate_Fe2 * f32_hs, mp=mp, has_solid=has_solid_prec,',
            '            add_lhs_sink=False, stoich_ratio=1.0,',
            '        )',
            '        add_implicit_sink(LHS, RATES, "TS2_32", prec_coeff_TS2_32, prec_rate_TS2 * f32_hs, mp=mp, has_solid=has_solid_prec)',
            '        CROSS["TS2_32"].append(("Fe2_total", -prec_coeff_Fe2 * f32_hs * phi))',
            '        RATES["TS2_32"] -= prec_rate_Fe2 * f32_hs * phi',
            '        FeS_val_np = np.asarray(c.FeS.value if hasattr(c.FeS, "value") else c.FeS)',
            '        FeS_32_val = np.asarray(c.FeS_32.value if hasattr(c.FeS_32, "value") else c.FeS_32)',
            '        f32_FeS = np.where(FeS_val_np > 1e-3, FeS_32_val / (FeS_val_np + 1e-30), f32_hs)',
            '        f32_FeS = np.clip(f32_FeS, 0.5, 1.5)',
            '        add_implicit_coupling_new(',
            '            CROSS, RATES, LHS, target_species="TS2_32", source_species="FeS_32",',
            '            coeff=diss_coeff_FeS, rate=diss_rate_FeS * f32_FeS, mp=mp, has_solid=has_solid,',
            '            add_lhs_sink=True, stoich_ratio=1.0,',
            '        )',
            '',
        ]

        active_categories = {self.classify_reaction(r) for r in self.reactions}
        if "FeS2_oxidation" in active_categories:
            code.extend([
                'def FeS2_oxidation(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
                '    """Reaction: 1 FeS2 + 3.5 O2 -> 1 Fe3 + 2 SO4"""',
                '    has_solid = True',
                '    k_num = mp.k.get("FeS2_O2") if hasattr(mp, "k") else getattr(k_val, "FeS2_O2", 0.0)',
                '    rate_base = k_num * c.FeS2 * c.O2',
                '    coeff_master = k_num * 1.0 * c.O2',
                '    add_coupled_reaction(',
                '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '        master_species={"FeS2": 1}, reactants={}, products={"Fe3": 1, "SO4": 2},',
                '        coeff_master=coeff_master, rate_master=rate_base,',
                '        has_solid=has_solid, reaction_name="FeS2_oxidation",',
                '        ref_species="FeS2", stoich_ref=1.0,',
                '    )',
                '    coeff_O2 = 3.5 * k_num * c.FeS2 * 1.0',
                '    add_implicit_sink(LHS, RATES, "O2", coeff_O2, 3.5 * rate_base, mp=mp, has_solid=has_solid, c=c)',
                '    if getattr(mp, "isotopes", False):',
                '        coeff_FeS2_32 = k_num * c.O2',
                '        add_coupled_reaction(',
                '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '            master_species={"FeS2_32": 1}, reactants={}, products={"SO4_32": 1},',
                '            coeff_master=coeff_FeS2_32, rate_master=coeff_FeS2_32 * c.FeS2_32,',
                '            has_solid=has_solid, reaction_name="FeS2_oxidation_32",',
                '            ref_species="FeS2", stoich_ref=1.0,',
                '        )',
                '',
            ])

        if "FeS2_precipitation_TS2" in active_categories:
            code.extend([
                'def FeS2_precipitation_TS2(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
                '    """Reaction: 1 FeS + 1 HS -> 1 FeS2"""',
                '    has_solid = True',
                '    hs_conc = c.TS2 * mp.hs_frac',
                '    k_num = mp.k.get("FeS_TS2") if hasattr(mp, "k") else getattr(k_val, "FeS_TS2", 0.0)',
                '    rate_base = k_num * c.FeS * hs_conc',
                '    coeff_master = k_num * 1.0 * hs_conc',
                '    add_coupled_reaction(',
                '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '        master_species={"FeS": 1}, reactants={}, products={"FeS2": 1},',
                '        coeff_master=coeff_master, rate_master=rate_base,',
                '        has_solid=has_solid, reaction_name="FeS2_precipitation_TS2",',
                '        ref_species="FeS", stoich_ref=1.0,',
                '    )',
                '    coeff_TS2 = k_num * c.FeS * 1.0 * mp.hs_frac',
                '    add_implicit_sink(LHS, RATES, "TS2", coeff_TS2, rate_base, mp=mp, has_solid=has_solid, c=c)',
                '    if getattr(mp, "isotopes", False):',
                '        coeff_FeS_32 = k_num * 1.0 * hs_conc',
                '        add_coupled_reaction(',
                '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '            master_species={"FeS_32": 1}, reactants={}, products={"FeS2_32": 1},',
                '            coeff_master=coeff_FeS_32, rate_master=coeff_FeS_32 * c.FeS_32,',
                '            has_solid=has_solid, reaction_name="FeS2_precipitation_TS2_FeS_32",',
                '            ref_species="FeS", stoich_ref=1.0,',
                '        )',
                '        coeff_HS_32 = k_num * c.FeS * 1.0 * mp.hs_frac',
                '        add_coupled_reaction(',
                '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '            master_species={"TS2_32": 1}, reactants={}, products={"FeS2_32": 1},',
                '            coeff_master=coeff_HS_32, rate_master=coeff_HS_32 * c.TS2_32,',
                '            has_solid=has_solid, reaction_name="FeS2_precipitation_TS2_HS_32",',
                '            ref_species="TS2", stoich_ref=1.0,',
                '        )',
                '',
                'pyrite_formation_fes_ts2_new = FeS2_precipitation_TS2',
                'pyrite_formation_FeS_TS2 = FeS2_precipitation_TS2',
                '',
            ])

        if "elemental_sulfur_oxidation" in active_categories:
            code.extend([
                'def elemental_sulfur_oxidation(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
                '    """Reaction: 2 S0 + 3 O2 -> 2 SO4"""',
                '    has_solid = True',
                '    k_num = mp.k.get("S0_O2") if hasattr(mp, "k") else getattr(k_val, "S0_O2", 0.0)',
                '    rate_base = k_num * c.O2 * c.S0',
                '    coeff_master = k_num * c.O2 * 1.0',
                '    add_coupled_reaction(',
                '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '        master_species={"S0": 2}, reactants={}, products={"SO4": 2},',
                '        coeff_master=coeff_master, rate_master=rate_base,',
                '        has_solid=has_solid, reaction_name="elemental_sulfur_oxidation",',
                '        ref_species="S0", stoich_ref=2.0,',
                '    )',
                '    coeff_O2 = 1.5 * k_num * 1.0 * c.S0',
                '    add_implicit_sink(LHS, RATES, "O2", coeff_O2, 1.5 * rate_base, mp=mp, has_solid=has_solid, c=c)',
                '    if getattr(mp, "isotopes", False):',
                '        add_coupled_reaction(',
                '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '            master_species={"S0_32": 2}, reactants={}, products={"SO4_32": 2},',
                '            coeff_master=coeff_master, rate_master=coeff_master * c.S0_32,',
                '            has_solid=has_solid, reaction_name="elemental_sulfur_oxidation_32",',
                '            ref_species="S0", stoich_ref=2.0,',
                '        )',
                '',
            ])

        if "pyrite_formation_fes_s0" in active_categories:
            code.extend([
                'def pyrite_formation_fes_s0_new(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
                '    """Reaction: 1 FeS + 1 S0 -> 1 FeS2"""',
                '    has_solid = True',
                '    k_num = mp.k.get("FeS_S0") if hasattr(mp, "k") else getattr(k_val, "FeS_S0", 0.0)',
                '    rate_base = k_num * c.FeS * c.S0',
                '    coeff_master = k_num * 1.0 * c.S0',
                '    add_coupled_reaction(',
                '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '        master_species={"FeS": 1}, reactants={}, products={"FeS2": 1},',
                '        coeff_master=coeff_master, rate_master=rate_base,',
                '        has_solid=has_solid, reaction_name="pyrite_formation_fes_s0_new",',
                '        ref_species="FeS", stoich_ref=1.0,',
                '    )',
                '    coeff_S0 = k_num * c.FeS * 1.0',
                '    add_implicit_sink(LHS, RATES, "S0", coeff_S0, rate_base, mp=mp, has_solid=has_solid, c=c)',
                '    if getattr(mp, "isotopes", False):',
                '        add_coupled_reaction(',
                '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '            master_species={"FeS_32": 1}, reactants={}, products={"FeS2_32": 1},',
                '            coeff_master=k_num * 1.0 * c.S0, rate_master=k_num * 1.0 * c.S0 * c.FeS_32,',
                '            has_solid=has_solid, reaction_name="pyrite_formation_fes_s0_new_FeS_32",',
                '            ref_species="FeS", stoich_ref=1.0,',
                '        )',
                '        add_coupled_reaction(',
                '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '            master_species={"S0_32": 1}, reactants={}, products={"FeS2_32": 1},',
                '            coeff_master=k_num * c.FeS * 1.0, rate_master=k_num * c.FeS * 1.0 * c.S0_32,',
                '            has_solid=has_solid, reaction_name="pyrite_formation_fes_s0_new_S0_32",',
                '            ref_species="S0", stoich_ref=1.0,',
                '        )',
                '',
                'pyrite_formation_S0 = pyrite_formation_fes_s0_new',
                '',
            ])

        if "sulfide_mediated_iron_reduction" in active_categories:
            code.extend([
                'def sulfide_mediated_iron_reduction(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
                '    """Reaction: HS + 2 Fe3 -> S0 + 2 Fe2_total"""',
                '    has_solid = True',
                '    hs_conc = c.TS2 * mp.hs_frac',
                '    k_num = mp.k.get("Fe3_hs") if hasattr(mp, "k") else getattr(k_val, "Fe3_hs", 0.0)',
                '    lim_o2 = lim.get("O2_inhibit", 1.0)',
                '    lim_fe3 = lim.get("Fe3_implicit", 1.0)',
                '    rate_base = k_num * c.Fe3 * hs_conc * lim_o2 * lim_fe3',
                '    coeff_master = k_num * c.Fe3 * 1.0 * mp.hs_frac * lim_o2 * lim_fe3',
                '    add_coupled_reaction(',
                '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '        master_species={"TS2": 1}, reactants={"Fe3": 2},',
                '        products={"Fe2_total": 2, "S0": 1},',
                '        coeff_master=coeff_master, rate_master=rate_base,',
                '        has_solid=has_solid, reaction_name="sulfide_mediated_iron_reduction",',
                '        ref_species="Fe3", stoich_ref=2.0,',
                '    )',
                '    if getattr(mp, "isotopes", False):',
                '        add_coupled_reaction(',
                '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '            master_species={"TS2_32": 1}, reactants={},',
                '            products={"S0_32": 1},',
                '            coeff_master=coeff_master, rate_master=coeff_master * c.TS2_32,',
                '            has_solid=has_solid, reaction_name="sulfide_mediated_iron_reduction_32",',
                '            ref_species="Fe3", stoich_ref=2.0,',
                '        )',
                '',
            ])

        if "elemental_sulfur_disproportionation" in active_categories:
            code.extend([
                'def elemental_sulfur_disproportionation(c, k_val, lim, LHS, RHS, RATES, CROSS, mp):',
                '    """Reaction: 4 S0 + 4 H2O -> 3 TS2 + SO4 + 2 Hplus"""',
                '    has_solid = True',
                '    k_num = mp.k.get("S0_dispro") if hasattr(mp, "k") else getattr(k_val, "S0_dispro", 0.0)',
                '    lim_ts2 = lim.get("TS2", 1.0)',
                '    lim_o2 = lim.get("O2_inhibit", 1.0)',
                '    coeff_s0 = np.maximum(k_num * lim_ts2 * lim_o2, 0.0)',
                '    rate_master = coeff_s0 * np.maximum(c.S0, 0.0)',
                '    add_coupled_reaction(',
                '        CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '        master_species={"S0": 4}, reactants={}, products={"TS2": 3, "SO4": 1},',
                '        coeff_master=coeff_s0, rate_master=rate_master,',
                '        has_solid=has_solid, reaction_name="elemental_sulfur_disproportionation",',
                '        ref_species="S0", stoich_ref=4.0,',
                '    )',
                '    if getattr(mp, "isotopes", False):',
                '        add_coupled_reaction(',
                '            CROSS=CROSS, LHS=LHS, RATES=RATES, mp=mp,',
                '            master_species={"S0_32": 4}, reactants={}, products={"TS2_32": 3, "SO4_32": 1},',
                '            coeff_master=coeff_s0, rate_master=coeff_s0 * c.S0_32,',
                '            has_solid=has_solid, reaction_name="elemental_sulfur_disproportionation_32",',
                '            ref_species="S0", stoich_ref=4.0,',
                '        )',
                '',
                'S0_disproportionation = elemental_sulfur_disproportionation',
                '',
            ])

        code.extend([
            '# -----------------------------------------------------------------------------',
            '# Main Diagenetic Reactions Dispatcher',
            '# -----------------------------------------------------------------------------',
            '',
            'DEFAULT_DIAGENETIC_REACTIONS = [',
        ])

        for r in self.reactions:
            cat = self.classify_reaction(r)
            reactants, _ = ReactionSystemValidator.parse_reaction_species(r.get("reaction", ""))
            k_name = r.get("k_value_name")

            if cat == "aerobic_respiration":
                poc_sp = next((sp for _, sp in reactants if "POC" in sp), "POC_fast")
                poc_k = k_name if isinstance(k_name, str) else poc_sp
                code.append(f'    [aerobic_respiration, {{"poc_species": "{poc_sp}", "poc_k": "{poc_k}"}}],')
            elif cat == "dissimilatory_iron_reduction":
                poc_sp = next((sp for _, sp in reactants if "POC" in sp), "POC_fast")
                poc_k = k_name if isinstance(k_name, str) else poc_sp
                code.append(f'    [dissimilatory_iron_reduction, {{"poc_species": "{poc_sp}", "poc_k": "{poc_k}"}}],')
            elif cat == "sulfate_reduction":
                poc_sp = next((sp for _, sp in reactants if "POC" in sp), "POC_fast")
                poc_k = k_name if isinstance(k_name, str) else poc_sp
                code.append(f'    [sulfate_reduction, {{"poc_species": "{poc_sp}", "poc_k": "{poc_k}"}}],')
            elif cat == "hs_oxidation":
                code.append('    [hs_oxidation_velde, None],')
            elif cat == "Fe2_oxidation":
                code.append('    [Fe2_oxidation, None],')
            elif cat == "sulfide_mediated_iron_reduction_velde":
                code.append('    [sulfide_mediated_iron_reduction_velde, None],')
            elif cat == "sulfide_mediated_iron_reduction":
                code.append('    [sulfide_mediated_iron_reduction, None],')
            elif cat == "FeS_precipitation_dissolution":
                code.append('    [FeS_precipitation_dissolution_smooth_transition, None],')
            elif cat == "FeS_oxidation":
                code.append('    [FeS_oxidation, None],')
            elif cat == "FeS2_oxidation":
                code.append('    [FeS2_oxidation, None],')
            elif cat == "FeS2_precipitation_TS2":
                code.append('    [FeS2_precipitation_TS2, None],')
            elif cat == "elemental_sulfur_oxidation":
                code.append('    [elemental_sulfur_oxidation, None],')
            elif cat == "pyrite_formation_fes_s0":
                code.append('    [pyrite_formation_fes_s0_new, None],')
            elif cat == "elemental_sulfur_disproportionation":
                code.append('    [elemental_sulfur_disproportionation, None],')
            else:
                r_name = r.get("reaction_name", "unnamed")
                raise ValueError(
                    f"Unable to generate reaction '{r_name}' ('{r.get('reaction', '')}'): "
                    f"reaction stoichiometry is not recognized by the equation system generator. "
                    f"Please check the chemical reaction equation or implement its generation rule."
                )

        code.extend([
            ']',
            '',
            'DIAGENETIC_REACTIONS = DEFAULT_DIAGENETIC_REACTIONS',
            '',
            'def diagenetic_reactions(mp, c, k, f=None, lim=None, dt=None):',
            '    """Evaluates all registered diagenetic reactions for the model column."""',
            '    LHS = {}',
            '    RHS = {}',
            '    RATES = {}',
            '    CROSS = {}',
            '    for sp in getattr(mp, "species_list", list(c.keys())):',
            '        shape = c[sp].shape if hasattr(c[sp], "shape") else (len(c[sp]),)',
            '        LHS[sp] = np.zeros(shape, dtype=np.float64)',
            '        RHS[sp] = np.zeros(shape, dtype=np.float64)',
            '        RATES[sp] = np.zeros(shape, dtype=np.float64)',
            '        CROSS[sp] = []',
            '    limiters = lim if lim is not None else get_limiters(c, mp)',
            '    rxns = getattr(mp, "diagenetic_reactions", None)',
            '    if rxns is None:',
            '        rxns = DEFAULT_DIAGENETIC_REACTIONS',
            '    for r_entry in rxns:',
            '        fn = r_entry[0]',
            '        fn_k = r_entry[1] if (len(r_entry) > 1 and r_entry[1] is not None) else k',
            '        fn(c, fn_k, limiters, LHS, RHS, RATES, CROSS, mp)',
            '    if f is not None:',
            '        sp_list = getattr(mp, "species_list", list(c.keys()))',
            '        for s in sp_list:',
            '            setattr(f, s, (LHS.get(s, 0.0), RHS.get(s, 0.0), RATES.get(s, 0.0), None))',
            '        if not getattr(mp, "in_solver", False):',
            '            for key, val in RATES.items():',
            '                if key not in sp_list:',
            '                    setattr(f, key, (None, None, val, None))',
            '        object.__setattr__(f, "raw_LHS", LHS)',
            '        object.__setattr__(f, "raw_CROSS", CROSS)',
            '        object.__setattr__(f, "raw_RHS", RHS)',
            '        object.__setattr__(f, "raw_RATES", RATES)',
            '    return f, RATES',
            '',
            '',
            '# -----------------------------------------------------------------------------',
            '# Analytical SymPy Chemical Jacobian',
            '# -----------------------------------------------------------------------------',
            '',
            'def compute_chemical_jacobian(c, mp, k, species_names):',
            '    """Analytically evaluates the N x S x S chemical Jacobian J[i, s, m] = dR_s / dC_m.',
            '    Derived symbolically via SymPy to exact machine precision.',
            '    """',
            '    N = len(c[species_names[0]])',
            '    S = len(species_names)',
            '    J = np.zeros((N, S, S), dtype=np.float64)',
            '    idx = {name: i for i, name in enumerate(species_names)}',
            '    phi = np.asarray(mp.phi.value if hasattr(mp.phi, "value") else mp.phi)',
            '    solid_fac = 1.0 - phi',
            '    diss_fac = phi',
            '    lim = get_limiters(c, mp)',
            '',
        ])

        active_categories = {self.classify_reaction(r) for r in self.reactions}

        if "aerobic_respiration" in active_categories:
            code.extend([
                '    # 1. Aerobic respiration (fast & slow)',
                '    for poc_sp in ["POC_fast", "POC_slow"]:',
                '        if poc_sp in idx and "O2" in idx:',
                '            i_poc = idx[poc_sp]',
                '            i_o2 = idx["O2"]',
                '            k_num = float(mp.k.get(poc_sp) if hasattr(mp, "k") else getattr(k, poc_sp, 0.0))',
                '            c_poc = np.asarray(c[poc_sp].value if hasattr(c[poc_sp], "value") else c[poc_sp])',
                '            c_o2 = np.asarray(c["O2"].value if hasattr(c["O2"], "value") else c["O2"])',
                '            K_O2 = float(mp.K_O2) / phi',
                '            denom = c_o2 + K_O2',
                '            d_poc = -solid_fac * k_num * (c_o2 / denom)',
                '            d_o2 = -solid_fac * k_num * c_poc * (K_O2 / denom**2)',
                '            ratio = float(getattr(mp, "POC_O2_ratio", 1.0))',
                '            J[:, i_poc, i_poc] += d_poc',
                '            J[:, i_poc, i_o2] += d_o2',
                '            J[:, i_o2, i_poc] += ratio * d_poc',
                '            J[:, i_o2, i_o2] += ratio * d_o2',
                '',
            ])

        if "dissimilatory_iron_reduction" in active_categories:
            code.extend([
                '    # 2. Dissimilatory iron reduction (fast & slow)',
                '    for poc_sp in ["POC_fast", "POC_slow"]:',
                '        if poc_sp in idx and "Fe3" in idx and "Fe2_total" in idx:',
                '            i_poc = idx[poc_sp]',
                '            i_fe3 = idx["Fe3"]',
                '            i_fe2 = idx["Fe2_total"]',
                '            k_num = float(mp.k.get(poc_sp) if hasattr(mp, "k") else getattr(k, poc_sp, 0.0))',
                '            c_poc = np.asarray(c[poc_sp].value if hasattr(c[poc_sp], "value") else c[poc_sp])',
                '            c_fe3 = np.asarray(c["Fe3"].value if hasattr(c["Fe3"], "value") else c["Fe3"])',
                '            K_Fe3_red = float(mp.K_Fe3_diss_red) / solid_fac',
                '            K_O2 = float(mp.K_O2) / phi',
                '            c_o2 = np.asarray(c["O2"].value if hasattr(c["O2"], "value") else c["O2"])',
                '            o2_inhib = K_O2 / (c_o2 + K_O2)',
                '            fe3_denom = c_fe3 + K_Fe3_red',
                '            rate_base = k_num * c_poc * (c_fe3 / fe3_denom) * o2_inhib',
                '            d_poc = -solid_fac * k_num * (c_fe3 / fe3_denom) * o2_inhib',
                '            d_fe3 = -solid_fac * k_num * c_poc * (K_Fe3_red / fe3_denom**2) * o2_inhib',
                '            d_o2 = solid_fac * k_num * c_poc * (c_fe3 / fe3_denom) * (K_O2 / (c_o2 + K_O2)**2)',
                '            J[:, i_poc, i_poc] += d_poc',
                '            J[:, i_poc, i_fe3] += d_fe3',
                '            J[:, i_fe3, i_poc] += 4.0 * d_poc',
                '            J[:, i_fe3, i_fe3] += 4.0 * d_fe3',
                '            J[:, i_fe2, i_poc] -= 4.0 * d_poc',
                '            J[:, i_fe2, i_fe3] -= 4.0 * d_fe3',
                '            if "O2" in idx:',
                '                J[:, i_poc, idx["O2"]] += d_o2',
                '                J[:, i_fe3, idx["O2"]] += 4.0 * d_o2',
                '                J[:, i_fe2, idx["O2"]] -= 4.0 * d_o2',
                '',
            ])

        if "sulfate_reduction" in active_categories:
            code.extend([
                '    # 3. Sulfate reduction (fast & slow)',
                '    for poc_sp in ["POC_fast", "POC_slow"]:',
                '        if poc_sp in idx and "SO4" in idx and "TS2" in idx:',
                '            i_poc = idx[poc_sp]',
                '            i_so4 = idx["SO4"]',
                '            i_ts2 = idx["TS2"]',
                '            k_num = float(mp.k.get(poc_sp) if hasattr(mp, "k") else getattr(k, poc_sp, 0.0))',
                '            c_poc = np.asarray(c[poc_sp].value if hasattr(c[poc_sp], "value") else c[poc_sp])',
                '            c_so4 = np.asarray(c["SO4"].value if hasattr(c["SO4"], "value") else c["SO4"])',
                '            c_o2 = np.asarray(c["O2"].value if hasattr(c["O2"], "value") else c["O2"])',
                '            c_fe3 = np.asarray(c["Fe3"].value if hasattr(c["Fe3"], "value") else c["Fe3"])',
                '            K_SO4 = float(mp.K_SO4) / phi',
                '            K_O2 = float(mp.K_O2) / phi',
                '            K_Fe3_red = float(mp.K_Fe3_diss_red) / solid_fac',
                '            o2_inhib = K_O2 / (c_o2 + K_O2)',
                '            fe3_inhib = K_Fe3_red / (c_fe3 + K_Fe3_red)',
                '            so4_denom = c_so4 + K_SO4',
                '            R_sr = 0.5 * solid_fac * k_num * c_poc * (c_so4 / so4_denom) * o2_inhib * fe3_inhib',
                '            d_poc = -0.5 * solid_fac * k_num * (c_so4 / so4_denom) * o2_inhib * fe3_inhib',
                '            d_so4 = -0.5 * solid_fac * k_num * c_poc * (K_SO4 / so4_denom**2) * o2_inhib * fe3_inhib',
                '            J[:, i_poc, i_poc] += 2.0 * d_poc',
                '            J[:, i_poc, i_so4] += 2.0 * d_so4',
                '            J[:, i_so4, i_poc] += d_poc',
                '            J[:, i_so4, i_so4] += d_so4',
                '            J[:, i_ts2, i_poc] -= d_poc',
                '            J[:, i_ts2, i_so4] -= d_so4',
                '            d_o2_sr = -R_sr * (K_O2 / denom_o2_sr**2) / (o2_inhib + 1e-30)' if False else '',
                '            denom_o2_sr = c_o2 + K_O2',
                '            d_o2_sr = 0.5 * solid_fac * k_num * c_poc * (c_so4 / so4_denom) * (K_O2 / denom_o2_sr**2) * fe3_inhib',
                '            denom_fe3_sr = c_fe3 + K_Fe3_red',
                '            d_fe3_sr = 0.5 * solid_fac * k_num * c_poc * (c_so4 / so4_denom) * o2_inhib * (K_Fe3_red / denom_fe3_sr**2)',
                '            if "O2" in idx:',
                '                J[:, i_poc, idx["O2"]] += 2.0 * d_o2_sr',
                '                J[:, i_so4, idx["O2"]] += d_o2_sr',
                '                J[:, i_ts2, idx["O2"]] -= d_o2_sr',
                '            if "Fe3" in idx:',
                '                J[:, i_poc, idx["Fe3"]] += 2.0 * d_fe3_sr',
                '                J[:, i_so4, idx["Fe3"]] += d_fe3_sr',
                '                J[:, i_ts2, idx["Fe3"]] -= d_fe3_sr',
                '            if "SO4_32" in idx and "TS2_32" in idx and getattr(mp, "isotopes", False):',
                '                i_so4_32 = idx["SO4_32"]',
                '                i_ts2_32 = idx["TS2_32"]',
                '                c_so4_32 = np.asarray(c["SO4_32"].value if hasattr(c["SO4_32"], "value") else c["SO4_32"])',
                '                alpha = 1.0 + (float(mp.msr_alpha) - 1.0) * (c_so4 / (c_so4 + float(getattr(mp, "K_epsilon_msr", 1e-3))))',
                '                d_so4_32 = -0.5 * solid_fac * k_num * c_poc * (alpha / so4_denom) * o2_inhib * fe3_inhib',
                '                frac_32 = np.where(c_so4 > 1e-20, c_so4_32 / (c_so4 + 1e-30), 1.0 / (1.0 + float(mp.VCDT)))',
                '                d_o2_sr_32 = d_o2_sr * frac_32 * alpha',
                '                d_fe3_sr_32 = d_fe3_sr * frac_32 * alpha',
                '                J[:, i_so4_32, i_so4_32] += d_so4_32',
                '                J[:, i_ts2_32, i_so4_32] -= d_so4_32',
                '                if "O2" in idx:',
                '                    J[:, i_so4_32, idx["O2"]] += d_o2_sr_32',
                '                    J[:, i_ts2_32, idx["O2"]] -= d_o2_sr_32',
                '                if "Fe3" in idx:',
                '                    J[:, i_so4_32, idx["Fe3"]] += d_fe3_sr_32',
                '                    J[:, i_ts2_32, idx["Fe3"]] -= d_fe3_sr_32',
                '',
            ])

        if "FeS_precipitation_dissolution" in active_categories:
            code.extend([
                '    # 4. FeS precipitation / dissolution (smooth transition)',
                '    if "Fe2_total" in idx and "TS2" in idx and "FeS" in idx:',
                '        i_fe2 = idx["Fe2_total"]',
                '        i_ts2 = idx["TS2"]',
                '        i_fes = idx["FeS"]',
                '        Fe2_val = np.maximum(c["Fe2_total"].value if hasattr(c["Fe2_total"], "value") else c["Fe2_total"], 1e-20)',
                '        TS2_val = np.maximum(c["TS2"].value if hasattr(c["TS2"], "value") else c["TS2"], 1e-20)',
                '        FeS_val = np.maximum(c["FeS"].value if hasattr(c["FeS"], "value") else c["FeS"], 1e-20)',
                '        Fe2_pw = Fe2_val * float(mp.Fe2_diss)',
                '        hs_val = TS2_val * float(mp.hs_frac)',
                '        k_Hplus = float(getattr(k, "Hplus", getattr(mp, "Hplus", 10**-7.5)))',
                '        k_FeS_sp = float(getattr(k, "FeS_sp", getattr(mp, "FeS_sp", 10**-3.5)))',
                '        k_FeS_isp = float(getattr(k, "FeS_isp", getattr(mp, "FeS_isp", 1.0)))',
                '        k_FeS_isd = float(getattr(k, "FeS_isd", getattr(mp, "FeS_isd", 0.3)))',
                '        omega = (Fe2_pw * hs_val) / (k_Hplus * k_FeS_sp + 1e-30)',
                '        d_omega_dFe2 = (float(mp.Fe2_diss) * hs_val) / (k_Hplus * k_FeS_sp + 1e-30)',
                '        d_omega_dTS2 = (Fe2_pw * float(mp.hs_frac)) / (k_Hplus * k_FeS_sp + 1e-30)',
                '        Km = 0.5',
                '        eps_sm = float(getattr(mp, "fes_smooth_epsilon", 0.05))',
                '        # Precipitation branch (Omega >= 1)',
                '        is_prec = omega >= 1.0',
                '        df_p = np.maximum(omega - 1.0, 0.0)',
                '        u_p = np.clip(df_p / (eps_sm + 1e-30), 0.0, 1.0)',
                '        S_p = 3.0 * u_p**2 - 2.0 * u_p**3',
                '        dS_du = np.where((u_p > 0.0) & (u_p < 1.0), 6.0 * u_p * (1.0 - u_p), 0.0)',
                '        dS_p_domega = dS_du / (eps_sm + 1e-30)',
                '        M_p = df_p / (Km + df_p)',
                '        dM_p_domega = Km / (Km + df_p)**2',
                '        dR_prec_domega = k_FeS_isp * (dM_p_domega * S_p + M_p * dS_p_domega) * is_prec',
                '        R_prec = k_FeS_isp * M_p * S_p * is_prec',
                '        # Bulk precipitation: factor is diss_fac (liquid)',
                '        dRp_dFe2 = diss_fac * dR_prec_domega * d_omega_dFe2',
                '        dRp_dTS2 = diss_fac * dR_prec_domega * d_omega_dTS2',
                '        # Dissolution branch (Omega < 1)',
                '        is_diss = omega < 1.0',
                '        us_d = np.maximum(1.0 - omega, 0.0)',
                '        u_d = np.clip(us_d / (eps_sm + 1e-30), 0.0, 1.0)',
                '        S_d = 3.0 * u_d**2 - 2.0 * u_d**3',
                '        dS_d_du = np.where((u_d > 0.0) & (u_d < 1.0), 6.0 * u_d * (1.0 - u_d), 0.0)',
                '        dS_d_domega = -dS_d_du / (eps_sm + 1e-30)',
                '        M_d = us_d / (Km + us_d)',
                '        dM_d_domega = -Km / (Km + us_d)**2',
                '        dR_diss_domega = k_FeS_isd * (dM_d_domega * S_d + M_d * dS_d_domega) * FeS_val * is_diss',
                '        dR_diss_dFeS = k_FeS_isd * M_d * S_d * is_diss',
                '        R_diss = k_FeS_isd * M_d * S_d * FeS_val * is_diss',
                '        dRd_dFe2 = solid_fac * dR_diss_domega * d_omega_dFe2',
                '        dRd_dTS2 = solid_fac * dR_diss_domega * d_omega_dTS2',
                '        dRd_dFeS = solid_fac * dR_diss_dFeS',
                '        # Total derivatives for FeS, Fe2, TS2',
                '        d_net_dFe2 = dRp_dFe2 - dRd_dFe2',
                '        d_net_dTS2 = dRp_dTS2 - dRd_dTS2',
                '        d_net_dFeS = -dRd_dFeS',
                '        J[:, i_fes, i_fe2] += d_net_dFe2',
                '        J[:, i_fes, i_ts2] += d_net_dTS2',
                '        J[:, i_fes, i_fes] += d_net_dFeS',
                '        J[:, i_fe2, i_fe2] -= d_net_dFe2',
                '        J[:, i_fe2, i_ts2] -= d_net_dTS2',
                '        J[:, i_fe2, i_fes] -= d_net_dFeS',
                '        J[:, i_ts2, i_fe2] -= d_net_dFe2',
                '        J[:, i_ts2, i_ts2] -= d_net_dTS2',
                '        J[:, i_ts2, i_fes] -= d_net_dFeS',
                '        if "FeS_32" in idx and "TS2_32" in idx and getattr(mp, "isotopes", False):',
                '            i_fes_32 = idx["FeS_32"]',
                '            i_ts2_32 = idx["TS2_32"]',
                '            hs_32 = partition_equilibrium_isotope_32(',
                '                c["TS2_32"].value if hasattr(c["TS2_32"], "value") else c["TS2_32"],',
                '                mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha,',
                '            )',
                '            f32_default = 1.0 / (1.0 + float(mp.VCDT))',
                '            mask_hs = hs_val > 1e-6',
                '            f32_hs = np.where(mask_hs, np.asarray(hs_32) / (hs_val + 1e-30), f32_default)',
                '            f32_hs = np.clip(f32_hs, 0.5, 1.5)',
                '            R_prec_bulk = diss_fac * R_prec',
                '            dRp32_dFe2 = dRp_dFe2 * f32_hs',
                '            dRp32_dTS2 = dRp_dTS2 * f32_hs - np.where(mask_hs, (R_prec_bulk / TS2_val) * f32_hs, 0.0)',
                '            dRp32_dTS2_32 = np.where(is_prec & mask_hs, R_prec_bulk / TS2_val, 0.0)',
                '            R_diss_bulk = solid_fac * R_diss',
                '            mask_fes = FeS_val > 1e-3',
                '            FeS_32_val = np.asarray(c["FeS_32"].value if hasattr(c["FeS_32"], "value") else c["FeS_32"])',
                '            f32_FeS = np.where(mask_fes, FeS_32_val / (FeS_val + 1e-30), f32_hs)',
                '            f32_FeS = np.clip(f32_FeS, 0.5, 1.5)',
                '            dRd32_dFe2 = dRd_dFe2 * f32_FeS',
                '            dRd32_dTS2 = dRd_dTS2 * f32_FeS',
                '            dRd32_dFeS = np.where(~mask_fes, dRd_dFeS * f32_hs, 0.0)',
                '            dRd32_dFeS_32 = np.where(is_diss & mask_fes, R_diss_bulk / FeS_val, 0.0)',
                '            J[:, i_fes_32, i_fe2] += (dRp32_dFe2 - dRd32_dFe2)',
                '            J[:, i_fes_32, i_ts2] += (dRp32_dTS2 - dRd32_dTS2)',
                '            J[:, i_fes_32, i_fes] -= dRd32_dFeS',
                '            J[:, i_fes_32, i_ts2_32] += dRp32_dTS2_32',
                '            J[:, i_fes_32, i_fes_32] -= dRd32_dFeS_32',
                '            J[:, i_ts2_32, i_fe2] -= (dRp32_dFe2 - dRd32_dFe2)',
                '            J[:, i_ts2_32, i_ts2] -= (dRp32_dTS2 - dRd32_dTS2)',
                '            J[:, i_ts2_32, i_fes] += dRd32_dFeS',
                '            J[:, i_ts2_32, i_ts2_32] -= dRp32_dTS2_32',
                '            J[:, i_ts2_32, i_fes_32] += dRd32_dFeS_32',
                '',
            ])

        if "hs_oxidation" in active_categories:
            code.extend([
                '    # 5. HS oxidation (hs_oxidation_velde)',
                '    if "TS2" in idx and "O2" in idx and "SO4" in idx:',
                '        i_ts2 = idx["TS2"]',
                '        i_o2 = idx["O2"]',
                '        i_so4 = idx["SO4"]',
                '        k_num = float(mp.k.get("TS2_O2") if hasattr(mp, "k") else getattr(k, "TS2_O2", 0.0))',
                '        c_o2 = np.asarray(c["O2"].value if hasattr(c["O2"], "value") else c["O2"])',
                '        c_ts2 = np.asarray(c["TS2"].value if hasattr(c["TS2"], "value") else c["TS2"])',
                '        K_O2_TS2 = float(getattr(mp, "K_O2_TS2", 1e-4))',
                '        denom_o2 = c_o2 + K_O2_TS2',
                '        hs_frac = float(mp.hs_frac)',
                '        d_ts2 = diss_fac * k_num * hs_frac * (c_o2 / denom_o2)',
                '        d_o2 = diss_fac * k_num * (c_ts2 * hs_frac) * (K_O2_TS2 / denom_o2**2)',
                '        J[:, i_ts2, i_ts2] -= d_ts2',
                '        J[:, i_ts2, i_o2] -= d_o2',
                '        J[:, i_so4, i_ts2] += d_ts2',
                '        J[:, i_so4, i_o2] += d_o2',
                '        J[:, i_o2, i_ts2] -= 2.0 * d_ts2',
                '        J[:, i_o2, i_o2] -= 2.0 * d_o2',
                '        if "TS2_32" in idx and "SO4_32" in idx and getattr(mp, "isotopes", False):',
                '            i_ts2_32 = idx["TS2_32"]',
                '            i_so4_32 = idx["SO4_32"]',
                '            c_ts2_32 = np.asarray(c["TS2_32"].value if hasattr(c["TS2_32"], "value") else c["TS2_32"])',
                '            alpha = 1.0 + (float(getattr(mp, "TS2_O2_alpha", 0.995)) - 1.0) * (c_ts2 / (c_ts2 + float(getattr(mp, "K_epsilon_TS2_O2", 1e-3))))',
                '            d_ts2_32 = diss_fac * k_num * hs_frac * (c_o2 / denom_o2) * alpha',
                '            d_ts2_32_o2 = diss_fac * k_num * (c_ts2_32 * hs_frac) * (K_O2_TS2 / denom_o2**2) * alpha',
                '            J[:, i_ts2_32, i_ts2_32] -= d_ts2_32',
                '            J[:, i_so4_32, i_ts2_32] += d_ts2_32',
                '            J[:, i_ts2_32, i_o2] -= d_ts2_32_o2',
                '            J[:, i_so4_32, i_o2] += d_ts2_32_o2',
                '',
            ])

        if "Fe2_oxidation" in active_categories:
            code.extend([
                '    # 6. Fe2 oxidation (Fe2_oxidation)',
                '    if "Fe2_total" in idx and "O2" in idx and "Fe3" in idx:',
                '        i_fe2 = idx["Fe2_total"]',
                '        i_o2 = idx["O2"]',
                '        i_fe3 = idx["Fe3"]',
                '        k_num = float(mp.k.get("Fe2_O2") if hasattr(mp, "k") else getattr(k, "Fe2_O2", 0.0))',
                '        c_fe2 = np.asarray(c["Fe2_total"].value if hasattr(c["Fe2_total"], "value") else c["Fe2_total"])',
                '        c_o2 = np.asarray(c["O2"].value if hasattr(c["O2"], "value") else c["O2"])',
                '        d_fe2 = diss_fac * k_num * c_o2',
                '        d_o2 = diss_fac * k_num * c_fe2',
                '        J[:, i_fe2, i_fe2] -= d_fe2',
                '        J[:, i_fe2, i_o2] -= d_o2',
                '        J[:, i_fe3, i_fe2] += d_fe2',
                '        J[:, i_fe3, i_o2] += d_o2',
                '        J[:, i_o2, i_fe2] -= 0.25 * d_fe2',
                '        J[:, i_o2, i_o2] -= 0.25 * d_o2',
                '',
            ])

        if any(cat in active_categories for cat in ("sulfide_mediated_iron_reduction", "sulfide_mediated_iron_reduction_velde")):
            code.extend([
                '    # 7. Sulfide-mediated iron reduction (sulfide_mediated_iron_reduction_velde)',
                '    if "TS2" in idx and "Fe3" in idx and "Fe2_total" in idx and "SO4" in idx:',
                '        i_ts2 = idx["TS2"]',
                '        i_fe3 = idx["Fe3"]',
                '        i_fe2 = idx["Fe2_total"]',
                '        i_so4 = idx["SO4"]',
                '        k_num = float(mp.k.get("Fe3_hs") if hasattr(mp, "k") else getattr(k, "Fe3_hs", 0.0))',
                '        c_fe3 = np.asarray(c["Fe3"].value if hasattr(c["Fe3"], "value") else c["Fe3"])',
                '        c_ts2 = np.asarray(c["TS2"].value if hasattr(c["TS2"], "value") else c["TS2"])',
                '        c_o2 = np.asarray(c["O2"].value if hasattr(c["O2"], "value") else c["O2"])',
                '        K_O2 = float(mp.K_O2) / phi',
                '        K_Fe3 = float(getattr(mp, "K_Fe3", 1e-3))',
                '        o2_inhibit = K_O2 / (c_o2 + K_O2)',
                '        fe3_lim = 1.0 / (c_fe3 + K_Fe3)',
                '        hs = c_ts2 * float(mp.hs_frac)',
                '        d_ts2 = (k_num * c_fe3 * float(mp.hs_frac) * o2_inhibit * fe3_lim / 8.0) * solid_fac',
                '        d_fe3 = (k_num * hs * o2_inhibit * (K_Fe3 / (c_fe3 + K_Fe3)**2) / 8.0) * solid_fac',
                '        denom_o2_sir = c_o2 + K_O2',
                '        d_o2_sir = (k_num * c_fe3 * hs * (K_O2 / denom_o2_sir**2) * fe3_lim / 8.0) * solid_fac',
                '        J[:, i_ts2, i_ts2] -= d_ts2',
                '        J[:, i_ts2, i_fe3] -= d_fe3',
                '        J[:, i_so4, i_ts2] += d_ts2',
                '        J[:, i_so4, i_fe3] += d_fe3',
                '        J[:, i_fe3, i_ts2] -= 8.0 * d_ts2',
                '        J[:, i_fe3, i_fe3] -= 8.0 * d_fe3',
                '        J[:, i_fe2, i_ts2] += 8.0 * d_ts2',
                '        J[:, i_fe2, i_fe3] += 8.0 * d_fe3',
                '        if "O2" in idx:',
                '            J[:, i_ts2, idx["O2"]] += d_o2_sir',
                '            J[:, i_so4, idx["O2"]] -= d_o2_sir',
                '            J[:, i_fe3, idx["O2"]] += 8.0 * d_o2_sir',
                '            J[:, i_fe2, idx["O2"]] -= 8.0 * d_o2_sir',
                '        if "TS2_32" in idx and "SO4_32" in idx and getattr(mp, "isotopes", False):',
                '            i_ts2_32 = idx["TS2_32"]',
                '            i_so4_32 = idx["SO4_32"]',
                '            c_ts2_32 = np.asarray(c["TS2_32"].value if hasattr(c["TS2_32"], "value") else c["TS2_32"])',
                '            hs_32 = c_ts2_32 * float(mp.hs_frac)',
                '            d_fe3_32 = (k_num * hs_32 * o2_inhibit * (K_Fe3 / (c_fe3 + K_Fe3)**2) / 8.0) * solid_fac',
                '            d_o2_sir_32 = (k_num * c_fe3 * hs_32 * (K_O2 / denom_o2_sir**2) * fe3_lim / 8.0) * solid_fac',
                '            J[:, i_ts2_32, i_ts2_32] -= d_ts2',
                '            J[:, i_so4_32, i_ts2_32] += d_ts2',
                '            J[:, i_ts2_32, i_fe3] -= d_fe3_32',
                '            J[:, i_so4_32, i_fe3] += d_fe3_32',
                '            if "O2" in idx:',
                '                J[:, i_ts2_32, idx["O2"]] += d_o2_sir_32',
                '                J[:, i_so4_32, idx["O2"]] -= d_o2_sir_32',
                '',
            ])

        if "FeS_oxidation" in active_categories:
            code.extend([
                '    # 8. FeS oxidation (FeS_oxidation)',
                '    if "FeS" in idx and "O2" in idx and "Fe3" in idx and "SO4" in idx:',
                '        i_fes = idx["FeS"]',
                '        i_o2 = idx["O2"]',
                '        i_fe3 = idx["Fe3"]',
                '        i_so4 = idx["SO4"]',
                '        k_num = float(mp.k.get("FeS_O2") if hasattr(mp, "k") else getattr(k, "FeS_O2", 0.0))',
                '        c_fes = np.asarray(c["FeS"].value if hasattr(c["FeS"], "value") else c["FeS"])',
                '        c_o2 = np.asarray(c["O2"].value if hasattr(c["O2"], "value") else c["O2"])',
                '        d_fes = solid_fac * k_num * c_o2',
                '        d_o2 = solid_fac * k_num * c_fes',
                '        J[:, i_fes, i_fes] -= d_fes',
                '        J[:, i_fes, i_o2] -= d_o2',
                '        J[:, i_fe3, i_fes] += d_fes',
                '        J[:, i_fe3, i_o2] += d_o2',
                '        J[:, i_so4, i_fes] += d_fes',
                '        J[:, i_so4, i_o2] += d_o2',
                '        J[:, i_o2, i_fes] -= 2.25 * d_fes',
                '        J[:, i_o2, i_o2] -= 2.25 * d_o2',
                '        if "FeS_32" in idx and "SO4_32" in idx and getattr(mp, "isotopes", False):',
                '            i_fes_32 = idx["FeS_32"]',
                '            i_so4_32 = idx["SO4_32"]',
                '            c_fes_32 = np.asarray(c["FeS_32"].value if hasattr(c["FeS_32"], "value") else c["FeS_32"])',
                '            d_o2_32 = solid_fac * k_num * c_fes_32',
                '            J[:, i_fes_32, i_fes_32] -= d_fes',
                '            J[:, i_so4_32, i_fes_32] += d_fes',
                '            J[:, i_fes_32, i_o2] -= d_o2_32',
                '            J[:, i_so4_32, i_o2] += d_o2_32',
                '',
            ])

        if "FeS2_oxidation" in active_categories:
            code.extend([
                '    # 9. FeS2 oxidation (FeS2_oxidation)',
                '    if "FeS2" in idx and "O2" in idx and "Fe3" in idx and "SO4" in idx:',
                '        i_fes2 = idx["FeS2"]',
                '        i_o2 = idx["O2"]',
                '        i_fe3 = idx["Fe3"]',
                '        i_so4 = idx["SO4"]',
                '        k_num = float(mp.k.get("FeS2_O2") if hasattr(mp, "k") else getattr(k, "FeS2_O2", 0.0))',
                '        c_fes2 = np.asarray(c["FeS2"].value if hasattr(c["FeS2"], "value") else c["FeS2"])',
                '        c_o2 = np.asarray(c["O2"].value if hasattr(c["O2"], "value") else c["O2"])',
                '        d_fes2 = solid_fac * k_num * c_o2',
                '        d_o2 = solid_fac * k_num * c_fes2',
                '        J[:, i_fes2, i_fes2] -= d_fes2',
                '        J[:, i_fes2, i_o2] -= d_o2',
                '        J[:, i_fe3, i_fes2] += d_fes2',
                '        J[:, i_fe3, i_o2] += d_o2',
                '        J[:, i_so4, i_fes2] += 2.0 * d_fes2',
                '        J[:, i_so4, i_o2] += 2.0 * d_o2',
                '        J[:, i_o2, i_fes2] -= 3.5 * d_fes2',
                '        J[:, i_o2, i_o2] -= 3.5 * d_o2',
                '        if "FeS2_32" in idx and "SO4_32" in idx and getattr(mp, "isotopes", False):',
                '            i_fes2_32 = idx["FeS2_32"]',
                '            i_so4_32 = idx["SO4_32"]',
                '            c_fes2_32 = np.asarray(c["FeS2_32"].value if hasattr(c["FeS2_32"], "value") else c["FeS2_32"])',
                '            d_o2_32 = solid_fac * k_num * c_fes2_32',
                '            J[:, i_fes2_32, i_fes2_32] -= d_fes2',
                '            J[:, i_so4_32, i_fes2_32] += 2.0 * d_fes2',
                '            J[:, i_fes2_32, i_o2] -= d_o2_32',
                '            J[:, i_so4_32, i_o2] += 2.0 * d_o2_32',
                '',
            ])

        if "FeS2_precipitation_TS2" in active_categories:
            code.extend([
                '    # 10. FeS2 precipitation via TS2 (FeS2_precipitation_TS2)',
                '    if "FeS" in idx and "TS2" in idx and "FeS2" in idx:',
                '        i_fes = idx["FeS"]',
                '        i_ts2 = idx["TS2"]',
                '        i_fes2 = idx["FeS2"]',
                '        k_num = float(mp.k.get("FeS_TS2") if hasattr(mp, "k") else getattr(k, "FeS_TS2", 0.0))',
                '        c_fes = np.asarray(c["FeS"].value if hasattr(c["FeS"], "value") else c["FeS"])',
                '        c_ts2 = np.asarray(c["TS2"].value if hasattr(c["TS2"], "value") else c["TS2"])',
                '        hs_frac = float(mp.hs_frac)',
                '        d_fes = solid_fac * k_num * c_ts2 * hs_frac',
                '        d_ts2 = solid_fac * k_num * c_fes * hs_frac',
                '        J[:, i_fes, i_fes] -= d_fes',
                '        J[:, i_fes, i_ts2] -= d_ts2',
                '        J[:, i_ts2, i_fes] -= d_fes',
                '        J[:, i_ts2, i_ts2] -= d_ts2',
                '        J[:, i_fes2, i_fes] += d_fes',
                '        J[:, i_fes2, i_ts2] += d_ts2',
                '        if "FeS_32" in idx and "TS2_32" in idx and "FeS2_32" in idx and getattr(mp, "isotopes", False):',
                '            i_fes_32 = idx["FeS_32"]',
                '            i_ts2_32 = idx["TS2_32"]',
                '            i_fes2_32 = idx["FeS2_32"]',
                '            J[:, i_fes_32, i_fes_32] -= d_fes',
                '            J[:, i_ts2_32, i_ts2_32] -= d_ts2',
                '            J[:, i_fes2_32, i_fes2_32] += d_fes',
                '            J[:, i_fes2_32, i_ts2_32] += d_ts2',
                '',
            ])

        if "elemental_sulfur_oxidation" in active_categories:
            code.extend([
                '    # 11. Elemental sulfur oxidation (elemental_sulfur_oxidation)',
                '    if "S0" in idx and "O2" in idx and "SO4" in idx:',
                '        i_s0 = idx["S0"]',
                '        i_o2 = idx["O2"]',
                '        i_so4 = idx["SO4"]',
                '        k_num = float(mp.k.get("S0_O2") if hasattr(mp, "k") else getattr(k, "S0_O2", 0.0))',
                '        c_s0 = np.asarray(c["S0"].value if hasattr(c["S0"], "value") else c["S0"])',
                '        c_o2 = np.asarray(c["O2"].value if hasattr(c["O2"], "value") else c["O2"])',
                '        d_s0 = solid_fac * k_num * c_o2',
                '        d_o2 = solid_fac * k_num * c_s0',
                '        J[:, i_s0, i_s0] -= d_s0',
                '        J[:, i_s0, i_o2] -= d_o2',
                '        J[:, i_o2, i_s0] -= 1.5 * d_s0',
                '        J[:, i_o2, i_o2] -= 1.5 * d_o2',
                '        J[:, i_so4, i_s0] += d_s0',
                '        J[:, i_so4, i_o2] += d_o2',
                '        if "S0_32" in idx and "SO4_32" in idx and getattr(mp, "isotopes", False):',
                '            i_s0_32 = idx["S0_32"]',
                '            i_so4_32 = idx["SO4_32"]',
                '            c_s0_32 = np.asarray(c["S0_32"].value if hasattr(c["S0_32"], "value") else c["S0_32"])',
                '            d_o2_32 = solid_fac * k_num * c_s0_32',
                '            J[:, i_s0_32, i_s0_32] -= d_s0',
                '            J[:, i_so4_32, i_s0_32] += d_s0',
                '            J[:, i_s0_32, i_o2] -= d_o2_32',
                '            J[:, i_so4_32, i_o2] += d_o2_32',
                '',
            ])

        if "pyrite_formation_fes_s0" in active_categories:
            code.extend([
                '    # 12. Pyrite formation via S0 (pyrite_formation_fes_s0_new)',
                '    if "FeS" in idx and "S0" in idx and "FeS2" in idx:',
                '        i_fes = idx["FeS"]',
                '        i_s0 = idx["S0"]',
                '        i_fes2 = idx["FeS2"]',
                '        k_num = float(mp.k.get("FeS_S0") if hasattr(mp, "k") else getattr(k, "FeS_S0", 0.0))',
                '        c_fes = np.asarray(c["FeS"].value if hasattr(c["FeS"], "value") else c["FeS"])',
                '        c_s0 = np.asarray(c["S0"].value if hasattr(c["S0"], "value") else c["S0"])',
                '        d_fes = solid_fac * k_num * c_s0',
                '        d_s0 = solid_fac * k_num * c_fes',
                '        J[:, i_fes, i_fes] -= d_fes',
                '        J[:, i_fes, i_s0] -= d_s0',
                '        J[:, i_s0, i_fes] -= d_fes',
                '        J[:, i_s0, i_s0] -= d_s0',
                '        J[:, i_fes2, i_fes] += d_fes',
                '        J[:, i_fes2, i_s0] += d_s0',
                '        if "FeS_32" in idx and "S0_32" in idx and "FeS2_32" in idx and getattr(mp, "isotopes", False):',
                '            i_fes_32 = idx["FeS_32"]',
                '            i_s0_32 = idx["S0_32"]',
                '            i_fes2_32 = idx["FeS2_32"]',
                '            J[:, i_fes_32, i_fes_32] -= d_fes',
                '            J[:, i_s0_32, i_s0_32] -= d_s0',
                '            J[:, i_fes2_32, i_fes2_32] += d_fes',
                '            J[:, i_fes2_32, i_s0_32] += d_s0',
                '',
            ])

        if "elemental_sulfur_disproportionation" in active_categories:
            code.extend([
                '    # 13. Elemental sulfur disproportionation',
                '    if "S0" in idx and "TS2" in idx and "SO4" in idx:',
                '        i_s0 = idx["S0"]',
                '        i_ts2 = idx["TS2"]',
                '        i_so4 = idx["SO4"]',
                '        k_num = float(mp.k.get("S0_dispro") if hasattr(mp, "k") else getattr(k, "S0_dispro", 0.0))',
                '        lim_ts2 = lim.get("TS2", 1.0)',
                '        lim_o2 = lim.get("O2_inhibit", 1.0)',
                '        coeff_s0 = np.maximum(k_num * lim_ts2 * lim_o2, 0.0)',
                '        d_s0 = solid_fac * coeff_s0',
                '        J[:, i_s0, i_s0] -= d_s0',
                '        J[:, i_ts2, i_s0] += 0.75 * d_s0',
                '        J[:, i_so4, i_s0] += 0.25 * d_s0',
                '        if "S0_32" in idx and "TS2_32" in idx and "SO4_32" in idx and getattr(mp, "isotopes", False):',
                '            i_s0_32 = idx["S0_32"]',
                '            i_ts2_32 = idx["TS2_32"]',
                '            i_so4_32 = idx["SO4_32"]',
                '            J[:, i_s0_32, i_s0_32] -= d_s0',
                '            J[:, i_ts2_32, i_s0_32] += 0.75 * d_s0',
                '            J[:, i_so4_32, i_s0_32] += 0.25 * d_s0',
                '',
            ])

        code.extend([
            '    return J',
            '',
            'diagenetic_reactions.compute_chemical_jacobian = compute_chemical_jacobian',
        ])
        return "\n".join(code)


def main(args_list: Optional[List[str]] = None) -> int:
    """CLI entrypoint for generate_equation_system."""
    parser = build_arg_parser()
    args = parser.parse_args(args_list)

    # Maintain attribute compatibility with legacy flag names
    if not hasattr(args, "input"):
        args.input = args.reactions
    if not hasattr(args, "output"):
        args.output = args.equations
    if not hasattr(args, "constants"):
        args.constants = args.kinetic_constants

    try:
        generate_equations(
            reactions_path=args.reactions,
            equations_path=args.equations,
            kinetic_constants_path=args.kinetic_constants,
            limiters_path=args.limiters,
            species_path=args.species,
            verbose=args.verbose,
            validate_only=args.validate_only,
        )
        if not args.validate_only:
            print(f"Successfully generated equations module saved to: {args.equations}")
        return 0
    except ValidationError as ve:
        print("\n" + "=" * 60, file=sys.stderr)
        print(f"VALIDATION FAILED:\n{ve}", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

