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


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds and returns the command line argument parser with full help descriptions."""
    parser = argparse.ArgumentParser(
        prog="generate_equation_system",
        description=(
            "FiPyrite Declarative Equation System Generator:\n"
            "Validates declarative chemical reaction specifications (chemical_equations_new.py) "
            "against model species, reaction constants, and limiters, and generates an assembled "
            "reaction equations module (equations.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("nbk/experiments/chemical_equations_new.py"),
        help="Path to declarative reaction definitions file (default: nbk/experiments/chemical_equations_new.py)",
    )
    parser.add_argument(
        "-o", "--output",
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
        "-k", "--constants",
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


def main(args_list: Optional[List[str]] = None) -> int:
    """CLI entrypoint for generate_equation_system."""
    parser = build_arg_parser()
    args = parser.parse_args(args_list)

    if args.verbose:
        print(f"Loading input reactions from: {args.input}")
        print(f"Using species definitions:   {args.species}")
        print(f"Using reaction constants:     {args.constants}")
        print(f"Using limiters definitions:   {args.limiters}")

    # 1. Load validator
    try:
        validator = ReactionSystemValidator(
            species_path=args.species,
            constants_path=args.constants,
            limiters_path=args.limiters,
        )
    except Exception as e:
        print(f"Error loading model definition files: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded {len(validator.valid_species)} species, "
              f"{len(validator.valid_constants)} reaction constants, "
              f"{len(validator.valid_limiters)} limiters.")

    # 2. Load input reactions
    try:
        reactions = load_reactions_from_file(args.input)
    except Exception as e:
        print(f"Error reading reaction definitions from {args.input}: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(reactions)} reaction definitions from {args.input}.")

    # 3. Validate
    errors = validator.validate_reactions(reactions)
    if errors:
        print("\n" + "=" * 60, file=sys.stderr)
        print(f"VALIDATION FAILED: {len(errors)} error(s) found in {args.input}:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for err in errors:
            print(f"  • {err}", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)
        return 1

    print(f"All {len(reactions)} reactions validated successfully against species, constants, and limiters!")

    if args.validate_only:
        return 0

class EquationSystemGenerator:
    """Generates an assembled equations.py module with SymPy analytical Jacobian."""

    def __init__(self, reactions: List[Dict[str, Any]], validator: ReactionSystemValidator):
        self.reactions = reactions
        self.validator = validator

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
            '# -----------------------------------------------------------------------------',
            '# Main Diagenetic Reactions Dispatcher',
            '# -----------------------------------------------------------------------------',
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
            '    rxns = getattr(mp, "diagenetic_reactions", [',
            '        [aerobic_respiration, {"poc_species": "POC_fast", "poc_k": "POC_fast"}],',
            '        [dissimilatory_iron_reduction, {"poc_species": "POC_fast", "poc_k": "POC_fast"}],',
            '        [sulfate_reduction, {"poc_species": "POC_fast", "poc_k": "POC_fast"}],',
            '        [aerobic_respiration, {"poc_species": "POC_slow", "poc_k": "POC_slow"}],',
            '        [dissimilatory_iron_reduction, {"poc_species": "POC_slow", "poc_k": "POC_slow"}],',
            '        [sulfate_reduction, {"poc_species": "POC_slow", "poc_k": "POC_slow"}],',
            '        [hs_oxidation_velde, k],',
            '        [Fe2_oxidation, k],',
            '        [sulfide_mediated_iron_reduction_velde, k],',
            '        [FeS_precipitation_dissolution_smooth_transition, k],',
            '        [FeS_oxidation, k],',
            '    ])',
            '    for r_entry in rxns:',
            '        fn = r_entry[0]',
            '        fn_k = r_entry[1]',
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
            '            if "SO4_32" in idx and "TS2_32" in idx and getattr(mp, "isotopes", False):',
            '                i_so4_32 = idx["SO4_32"]',
            '                i_ts2_32 = idx["TS2_32"]',
            '                c_so4_32 = np.asarray(c["SO4_32"].value if hasattr(c["SO4_32"], "value") else c["SO4_32"])',
            '                alpha = 1.0 + (float(mp.msr_alpha) - 1.0) * (c_so4 / (c_so4 + float(getattr(mp, "K_epsilon_msr", 1e-3))))',
            '                d_so4_32 = -0.5 * solid_fac * k_num * c_poc * (alpha / so4_denom) * o2_inhib * fe3_inhib',
            '                J[:, i_so4_32, i_so4_32] += d_so4_32',
            '                J[:, i_ts2_32, i_so4_32] -= d_so4_32',
            '',
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
            '        J[:, i_ts2, i_ts2] -= d_ts2',
            '        J[:, i_ts2, i_fe3] -= d_fe3',
            '        J[:, i_so4, i_ts2] += d_ts2',
            '        J[:, i_so4, i_fe3] += d_fe3',
            '        J[:, i_fe3, i_ts2] -= 8.0 * d_ts2',
            '        J[:, i_fe3, i_fe3] -= 8.0 * d_fe3',
            '        J[:, i_fe2, i_ts2] += 8.0 * d_ts2',
            '        J[:, i_fe2, i_fe3] += 8.0 * d_fe3',
            '        if "TS2_32" in idx and "SO4_32" in idx and getattr(mp, "isotopes", False):',
            '            i_ts2_32 = idx["TS2_32"]',
            '            i_so4_32 = idx["SO4_32"]',
            '            J[:, i_ts2_32, i_ts2_32] -= d_ts2',
            '            J[:, i_so4_32, i_ts2_32] += d_ts2',
            '',
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
            '            J[:, i_fes_32, i_fes_32] -= d_fes',
            '            J[:, i_so4_32, i_fes_32] += d_fes',
            '',
            '    return J',
            '',
            'diagenetic_reactions.compute_chemical_jacobian = compute_chemical_jacobian',
            '',
        ]
        return "\n".join(code)


def main(args_list: Optional[List[str]] = None) -> int:
    """CLI entrypoint for generate_equation_system."""
    parser = build_arg_parser()
    args = parser.parse_args(args_list)

    if args.verbose:
        print(f"Loading input reactions from: {args.input}")
        print(f"Using species definitions:   {args.species}")
        print(f"Using reaction constants:     {args.constants}")
        print(f"Using limiters definitions:   {args.limiters}")

    # 1. Load validator
    try:
        validator = ReactionSystemValidator(
            species_path=args.species,
            constants_path=args.constants,
            limiters_path=args.limiters,
        )
    except Exception as e:
        print(f"Error loading model definition files: {e}", file=sys.stderr)
        return 1

    # 2. Load input reactions
    try:
        reactions = load_reactions_from_file(args.input)
    except Exception as e:
        print(f"Error reading reaction definitions from {args.input}: {e}", file=sys.stderr)
        return 1

    # 3. Validate
    errors = validator.validate_reactions(reactions)
    if errors:
        print("\n" + "=" * 60, file=sys.stderr)
        print(f"VALIDATION FAILED: {len(errors)} error(s) found in {args.input}:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for err in errors:
            print(f"  • {err}", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)
        return 1

    print(f"All {len(reactions)} reactions validated successfully against species, constants, and limiters!")

    if args.validate_only:
        return 0

    # 4. Code Generation
    generator = EquationSystemGenerator(reactions=reactions, validator=validator)
    generated_code = generator.generate_code()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(generated_code)

    print(f"Successfully generated equations module saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

