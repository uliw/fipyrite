"""
# Generate Python reaction functions from chemical equations config

### Usage
```bash
generate-equations -i chemical_equations.py -o generated_equations.py
```

The input file (e.g., `chemical_equations.py`) must be a Python file defining a list of 
dictionaries named `reactions`. Each dictionary represents a reaction configuration.

### Supported keys in each reaction dictionary
* **`reaction`** (*str*): The chemical equation, e.g., `"2 POC + SO4 -> TS2"`. Reactants and products must be separated by `'->'` and species by `'+'`. Coefficients must be integers (defaults to 1 if omitted).
* **`reaction_name`** (*str*): The name of the generated Python function, e.g., `"sulfate_reduction"`.
* **`k_value_name`** (*str*): The kinetic rate constant key (e.g., `"poc_k"`). If `"poc_k"` is used, the generated code dynamically looks up `'poc_fast'` or `'poc_slow'`.
* **`master_species`** (*str*, optional): The primary reactant solved implicitly. If not provided, it is automatically derived from the `k_value_name` or reactants.
* **`fractionation`** (*list of lists*, optional): Isotope fractionation definitions of the form:
    `[[source_species, target_species, alpha_param_name, limiter_param_name]]`.
    E.g., `[["SO4", "TS2", "msr_alpha", "SO4_alpha_explicit"]]`.
* **`isotope_suffix`** (*str*, optional): Suffix for isotope species (default: `"32"`).
* **`Monod_kinetics`** (*float*, optional): Half-saturation constant ($K_m$) for the master species. If greater than 0, applies a Monod limitation term: `c.master / (c.master + K_m)`.
* **`dynamic_variables`** (*dict*, optional): Map of reactants to custom expressions, e.g., `{"HS": "c.TS2 * mp.hs_frac"}`.
* **`limiters`** (*list of str*, optional): List of limiter keys from the `lim` dict, e.g., `["O2_inhibit", "SO4_implicit"]`.
* **`verify_stoichiometry`** (*bool*, optional): If True, validates stoichiometry using ChemPy.

### Example `chemical_equations.py`
```python
reactions = [
    {
        "reaction": "2 POC + SO4 -> TS2",
        "reaction_name": "sulfate_reduction",
        "k_value_name": "poc_k",
        "limiters": ["O2_inhibit", "SO4_implicit"],
        "fractionation": [["SO4", "TS2", "msr_alpha", "SO4_alpha_explicit"]],
    }
]
```
"""

import argparse
import json
import sys
import importlib.util
from pathlib import Path
from chempy import Reaction

# Load species.py from experiments relative to package or sys.path
try:
    from species import species
except ImportError:
    # If not in path, add experiments folder
    experiments_dir = Path(__file__).resolve().parents[2] / "experiments"
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    try:
        from species import species
    except ImportError:
        species = {}

KNOWN_MONOD_LIMITERS = {
    "O2_implicit": {"species": "O2", "K_expr": "mp.K_O2 / mp.phi"},
    "O2_implicit_TS2": {"species": "O2", "K_expr": "mp.K_O2_TS2"},
    "SO4_implicit": {"species": "SO4", "K_expr": "mp.K_SO4 / mp.phi"},
    "Fe3_implicit": {"species": "Fe3", "K_expr": "mp.K_Fe3 / (1.0 - mp.phi)"},
    "Fe3_diss_red_implicit": {"species": "Fe3", "K_expr": "mp.K_Fe3_diss_red / (1.0 - mp.phi)"},
}

def verify_stoichiometry(reaction: str) -> bool:
    """Verify stoichiometry of the reaction string. Currently a stub."""
    # TODO: Implement full chemical element balancing checks using chempy
    return True

def create_reaction(
    reaction: str = "POC + 4 Fe3 -> HCO3 + 4Fe2",
    reaction_name: str = "dissimilatory_iron_reduction",
    k_value_name: str = "poc_k",
    fractionation: list = None,
    isotope_suffix: str = "32",
    Monod_kinetics: float = 0,
    dynamic_variables: dict = None,
    limiters: list = None,
    master_species: str = None,
    ref_species: str = None,
    monod_scheme: str = "picard",
    monod_limiter: dict = None,
):
    """
    Generate the Python function code for a reaction using ChemPy parser.

    Parameters:
    -----------
    reaction : str
        The chemical equation string, e.g. "2 POC + SO4 -> TS2". 
        Reactants and products must be separated by '->' and species by '+'.
        Any species with 'include: True' in species.py is tracked by the solver.
    reaction_name : str
        The name of the generated Python function, e.g. "sulfate_reduction".
    k_value_name : str
        The name of the kinetic rate constant. If "poc_k", the generator
        will dynamically look up either 'poc_fast' or 'poc_slow' based on 
        the scenario configuration. Otherwise, it looks up the specific key in mp.k.
    fractionation : list of lists, optional
        Isotope fractionation definitions of the form:
        [[source_species, target_species, alpha_param_name, limiter_param_name]]
        E.g. [["SO4", "TS2", "msr_alpha", "SO4_alpha_explicit"]].
    isotope_suffix : str, default "32"
        The suffix appended to isotope species, e.g. "32" for SO4_32, TS2_32.
    Monod_kinetics : float, default 0
        Half-saturation constant (K_m) for the master species. If non-zero,
        applies a Monod limitation multiplier: c.master / (c.master + Monod_kinetics).
    dynamic_variables : dict, optional
        A mapping of generic equation terms to their actual variable/expression.
        E.g. {"HS": "c.TS2 * mp.hs_frac"} maps the reactant HS to its bisulfide speciation.
    verify_stoichiometry : bool, default False
        If True, validates the element balancing of the reaction using ChemPy.
    limiters : list of str, optional
        List of limiter names from the 'lim' dictionary to multiply into the rate.
        E.g. ["O2_inhibit", "SO4_implicit"].
    master_species : str, optional
        The primary reactant solved implicitly. If not specified, derived automatically
        from k_value_name or the first non-POC reactant.

    Returns:
    --------
    code : str
        The complete, auto-generated Python function code.
    """
    # Parse reaction equation using chempy
    rxn = Reaction.from_string(reaction)
    reactants = rxn.reac
    products = rxn.prod

    # Verify species names against species.py
    all_parsed_species = list(reactants.keys()) + list(products.keys())
    for s in all_parsed_species:
        if s not in species and s != "POC" and (not dynamic_variables or s not in dynamic_variables):
            raise ValueError(f"Species '{s}' in reaction '{reaction}' is not in species.py and is not mapped.")

    # Helper to get coefficient expression
    def get_coeff_expr(s, is_product=False):
        dict_to_use = products if is_product else reactants
        if s in dict_to_use:
            return str(dict_to_use[s])
        if not is_product and s == "TS2" and "HS" in dict_to_use:
            return str(dict_to_use["HS"])
        return str(dict_to_use.get(s, 1.0))

    def is_same_species(s1, s2):
        if s1 == s2:
            return True
        if (s1 == "TS2" and s2 == "HS") or (s1 == "HS" and s2 == "TS2"):
            return True
        return False

    # Derive ref_species from k_value_name if not specified
    if not ref_species:
        parts = k_value_name.split("_")
        first = parts[0]
        if first == "poc":
            ref_species = "POC"
        elif first == "Fe2":
            ref_species = "Fe2_total"
        else:
            for s in species:
                if s.lower() == first.lower():
                    ref_species = s
                    break
        if not ref_species:
            ref_species = "POC" if "POC" in reactants else master_species

    # Derive master_species automatically if not specified
    if not master_species:
        # Determine candidate ref_species
        ref_candidate = ref_species
        # If ref_candidate is liquid, it is the master species
        if ref_candidate and ref_candidate != "POC" and not species[ref_candidate].get("solid", False):
            master_species = ref_candidate
        else:
            # If ref_candidate is solid, check if we need to couple to a liquid sulfur species
            if "TS2" in reactants or "HS" in reactants:
                master_species = "TS2"
            elif ref_candidate and ref_candidate != "POC":
                master_species = ref_candidate
            else:
                # Fallback to first non-POC reactant
                for r in reactants:
                    if r != "POC":
                        master_species = r
                        break
                else:
                    master_species = list(reactants.keys())[0]

    # Determine has_solid based on reactants only
    has_solid = False
    for r in reactants:
        if r == "POC":
            has_solid = True
        elif r in species and species[r].get("solid", False):
            has_solid = True
            break

    # Start building code
    code = f"def {reaction_name}(c, k, lim, LHS, RHS, RATES, CROSS, mp):\n"
    code += f"    has_solid = {has_solid}\n"

    # Add dynamic variables mapping
    if dynamic_variables:
        for var_name, expr in dynamic_variables.items():
            code += f"    {var_name} = {expr}\n"

    # Resolve k_val
    if "POC" in reactants or "POC" in products or k_value_name == "poc_k":
        code += "    poc_species = k.get('poc_species', 'POC_fast')\n"
        code += "    poc_k_name = k.get('poc_k', 'POC_fast')\n"
        code += "    k_val = mp.k.get(poc_k_name) if hasattr(mp, 'k') else k.get(poc_k_name, 0.0)\n"
    else:
        code += f"    k_val = mp.k.get('{k_value_name}') if hasattr(mp, 'k') else k.get('{k_value_name}', 0.0)\n"

    # Build concentration terms for the rate equation
    conc_terms_map = {}
    for r in reactants:
        if r == "POC":
            conc_terms_map[r] = "getattr(c, poc_species)"
        elif dynamic_variables and r in dynamic_variables:
            conc_terms_map[r] = r
        elif r in species and species[r]["include"]:
            conc_terms_map[r] = f"c.{r}"

    # Helper to generate analytical coefficient expression (avoiding division by c.r + 1e-30)
    def make_coeff_expr(r, multiplier=None):
        lookup_key = "HS" if r == "TS2" and "HS" in conc_terms_map else r
        terms = ["k_val"]
        for k_sp, v_expr in conc_terms_map.items():
            if k_sp == lookup_key:
                if r == "POC":
                    terms.append("1.0")
                elif r == "TS2":
                    if v_expr == "HS" and dynamic_variables and "HS" in dynamic_variables:
                        expr = dynamic_variables["HS"]
                        terms.append(expr.replace("c.TS2", "1.0"))
                    else:
                        terms.append(v_expr.replace("c.TS2", "1.0"))
                else:
                    terms.append(v_expr.replace(f"c.{r}", "1.0"))
            else:
                terms.append(v_expr)
        if limiters:
            for lim_name in limiters:
                terms.append(f"lim['{lim_name}']")
        if Monod_kinetics > 0:
            if r == master_species:
                terms.append(f"(1.0 / (c.{master_species} + {Monod_kinetics}))")
            else:
                terms.append(f"(c.{master_species} / (c.{master_species} + {Monod_kinetics}))")
        expr = " * ".join(terms)
        if multiplier:
            expr = f"({multiplier}) * {expr}"
        return expr

    def find_monod_limiter(r):
        if monod_limiter and monod_limiter.get("species") == r:
            return monod_limiter.get("limiter_name"), monod_limiter.get("K_expr")
        if limiters:
            for lim_name in limiters:
                if lim_name in KNOWN_MONOD_LIMITERS and KNOWN_MONOD_LIMITERS[lim_name]["species"] == r:
                    return lim_name, KNOWN_MONOD_LIMITERS[lim_name]["K_expr"]
        return None, None

    def make_rmax_expr(r, skip_limiter=None, multiplier=None):
        lookup_key = "HS" if r == "TS2" and "HS" in conc_terms_map else r
        terms = ["k_val"]
        for k_sp, v_expr in conc_terms_map.items():
            if k_sp == lookup_key:
                if r == "POC":
                    terms.append("1.0")
                elif r == "TS2":
                    if v_expr == "HS" and dynamic_variables and "HS" in dynamic_variables:
                        expr = dynamic_variables["HS"]
                        terms.append(expr.replace("c.TS2", "1.0"))
                    else:
                        terms.append(v_expr.replace("c.TS2", "1.0"))
                else:
                    terms.append(v_expr.replace(f"c.{r}", "1.0"))
            else:
                terms.append(v_expr)
        if limiters:
            for lim_name in limiters:
                if lim_name != skip_limiter:
                    terms.append(f"lim['{lim_name}']")
        expr = " * ".join(terms)
        if multiplier:
            expr = f"({multiplier}) * {expr}"
        return expr

    # Build base rate expression
    rate_base_terms = ["k_val"] + list(conc_terms_map.values())
    if limiters:
        for lim_name in limiters:
            rate_base_terms.append(f"lim['{lim_name}']")
    if Monod_kinetics > 0:
        rate_base_terms.append(f"(c.{master_species} / (c.{master_species} + {Monod_kinetics}))")
    code += "    rate_base = " + " * ".join(rate_base_terms) + "\n"

    ref_stoich_key = ref_species if ref_species else master_species
    if ref_stoich_key == "TS2" and "HS" in reactants:
        ref_stoich_key = "HS"
    ref_stoich = float(reactants.get("POC", reactants.get(ref_stoich_key, 1.0)))

    # Separate solid/other reactants that we handle via implicit sinks directly
    # and those that are coupled via add_coupled_reaction
    tracked_products = [p for p in products if p in species and species[p]["include"]]

    if tracked_products:
        # We have products to couple
        code += "    rate_master = rate_base\n"
        code += f"    coeff_master = {make_coeff_expr(master_species)}\n"

        # Format reactants and products dictionaries for the helper function
        reac_dict_parts = []
        for r, coeff in reactants.items():
            if not is_same_species(r, master_species):
                if r == "POC" or (r in species and species[r].get("solid", False)):
                    coeff_expr = get_coeff_expr(r)
                    if r == "POC":
                        reac_dict_parts.append(f"poc_species: {coeff_expr}")
                    else:
                        reac_dict_parts.append(f"'{r}': {coeff_expr}")
        reac_dict_str = "{" + ", ".join(reac_dict_parts) + "}"

        prod_dict_parts = []
        for p, coeff in products.items():
            if p in species and species[p]["include"]:
                coeff_expr = get_coeff_expr(p, is_product=True)
                prod_dict_parts.append(f"'{p}': {coeff_expr}")
        prod_dict_str = "{" + ", ".join(prod_dict_parts) + "}"

        master_coeff_expr = get_coeff_expr(master_species)
        code += "    add_coupled_reaction(\n"
        code += "        CROSS=CROSS,\n"
        code += "        LHS=LHS,\n"
        code += "        RATES=RATES,\n"
        code += "        mp=mp,\n"
        code += f"        master_species={{'{master_species}': {master_coeff_expr}}},\n"
        code += f"        reactants={reac_dict_str},\n"
        code += f"        products={prod_dict_str},\n"
        code += "        coeff_master=coeff_master,\n"
        code += "        rate_master=rate_master,\n"
        code += "        has_solid=has_solid,\n"
        code += f"        reaction_name='{reaction_name}',\n"
        if ref_species:
            code += f"        ref_species='{ref_species}',\n"
        elif "POC" in reactants:
            code += "        ref_species=poc_species,\n"
        else:
            code += f"        ref_species='{master_species}',\n"
        code += f"        stoich_ref={ref_stoich},\n"
        code += "    )\n"

        # Generate implicit sinks for other reactants
        for r, coeff in reactants.items():
            if not is_same_species(r, master_species):
                # Generate if POC or if NOT solid
                if r == "POC" or not (r in species and species[r].get("solid", False)):
                    coeff_expr = get_coeff_expr(r)
                    if r == "POC":
                        code += f"    coeff_POC = {make_coeff_expr('POC')}\n"
                        code += "    add_implicit_sink(LHS, RATES, poc_species, coeff_POC, rate_base, mp=mp, has_solid=has_solid, c=c)\n"
                    elif r in species and species[r]["include"]:
                        multiplier = f"({coeff_expr}) / {ref_stoich}"
                        monod_lim_name, k_m_expr = find_monod_limiter(r)
                        if monod_scheme != "picard" and monod_lim_name:
                            code += f"    R_max_{r} = {make_rmax_expr(r, skip_limiter=monod_lim_name, multiplier=multiplier)}\n"
                            code += f"    add_monod_sink(\n"
                            code += f"        LHS=LHS, RHS=RHS, RATES=RATES,\n"
                            code += f"        species='{r}',\n"
                            code += f"        conc=c.{r},\n"
                            code += f"        K_m={k_m_expr},\n"
                            code += f"        R_max=R_max_{r},\n"
                            code += f"        mp=mp,\n"
                            code += f"        has_solid=has_solid,\n"
                            code += f"        scheme=getattr(mp, 'monod_scheme', '{monod_scheme}'),\n"
                            code += f"        c=c,\n"
                            code += f"    )\n"
                        else:
                            code += f"    coeff_{r} = {make_coeff_expr(r, multiplier=multiplier)}\n"
                            code += f"    add_implicit_sink(LHS, RATES, '{r}', coeff_{r}, ({multiplier}) * rate_base, mp=mp, has_solid=has_solid, c=c)\n"
    else:
        # No tracked products, just add implicit sinks for all tracked reactants
        for r, coeff in reactants.items():
            coeff_expr = get_coeff_expr(r)
            if r == "POC":
                code += f"    coeff_POC = {make_coeff_expr('POC')}\n"
                code += "    add_implicit_sink(LHS, RATES, poc_species, coeff_POC, rate_base, mp=mp, has_solid=has_solid, c=c)\n"
            elif r in species and species[r]["include"]:
                multiplier = f"({coeff_expr}) / {ref_stoich}"
                monod_lim_name, k_m_expr = find_monod_limiter(r)
                if monod_scheme != "picard" and monod_lim_name:
                    code += f"    R_max_{r} = {make_rmax_expr(r, skip_limiter=monod_lim_name, multiplier=multiplier)}\n"
                    code += f"    add_monod_sink(\n"
                    code += f"        LHS=LHS, RHS=RHS, RATES=RATES,\n"
                    code += f"        species='{r}',\n"
                    code += f"        conc=c.{r},\n"
                    code += f"        K_m={k_m_expr},\n"
                    code += f"        R_max=R_max_{r},\n"
                    code += f"        mp=mp,\n"
                    code += f"        has_solid=has_solid,\n"
                    code += f"        scheme=getattr(mp, 'monod_scheme', '{monod_scheme}'),\n"
                    code += f"        c=c,\n"
                    code += f"    )\n"
                else:
                    code += f"    coeff_{r} = {make_coeff_expr(r, multiplier=multiplier)}\n"
                    code += f"    add_implicit_sink(LHS, RATES, '{r}', coeff_{r}, ({multiplier}) * rate_base, mp=mp, has_solid=has_solid, c=c)\n"

    # Isotope generation
    has_sulfur = any(s in species and "S" in species[s]["formula"] for s in all_parsed_species)
    if has_sulfur and isotope_suffix:
        code += "    if mp.isotopes:\n"
        if fractionation:
            for frac in fractionation:
                source, target, alpha_name, limiter_name = frac
                code += f"        alpha = 1.0 + (mp.{alpha_name} - 1.0) * lim['{limiter_name}']\n"
                if dynamic_variables and source in dynamic_variables:
                    if source == "HS":
                        code += f"        hs_{isotope_suffix} = partition_equilibrium_isotope_32(\n"
                        code += f"            c.TS2_{isotope_suffix}, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha\n"
                        code += f"        )\n"
                        code += f"        coeff_master_{isotope_suffix} = calculate_fractionated_coeff_32(\n"
                        code += f"            coeff_master, c.TS2 * mp.hs_frac, hs_{isotope_suffix}, alpha, eps=1e-30\n"
                        code += f"        )\n"
                else:
                    code += f"        coeff_master_{isotope_suffix} = calculate_fractionated_coeff_32(\n"
                    code += f"            coeff_master, c.{source}, c.{source}_{isotope_suffix}, alpha, eps=1e-30\n"
                    code += f"        )\n"
        else:
            code += f"        coeff_master_{isotope_suffix} = coeff_master\n"

        iso_reac_parts = []
        for r, coeff in reactants.items():
            if not is_same_species(r, master_species):
                if r + f"_{isotope_suffix}" in species:
                    coeff_expr = get_coeff_expr(r)
                    iso_reac_parts.append(f"'{r}_{isotope_suffix}': {coeff_expr}")
        iso_reac_str = "{" + ", ".join(iso_reac_parts) + "}"

        iso_prod_parts = []
        for p, coeff in products.items():
            if p in species and species[p]["include"]:
                coeff_expr = get_coeff_expr(p, is_product=True)
                if p + f"_{isotope_suffix}" in species:
                    iso_prod_parts.append(f"'{p}_{isotope_suffix}': {coeff_expr}")
        iso_prod_str = "{" + ", ".join(iso_prod_parts) + "}"

        code += "        add_coupled_reaction(\n"
        code += "            CROSS=CROSS,\n"
        code += "            LHS=LHS,\n"
        code += "            RATES=RATES,\n"
        code += "            mp=mp,\n"
        code += f"            master_species={{'{master_species}_{isotope_suffix}': {master_coeff_expr}}},\n"
        code += f"            reactants={iso_reac_str},\n"
        code += f"            products={iso_prod_str},\n"
        code += f"            coeff_master=coeff_master_{isotope_suffix},\n"
        code += f"            rate_master=coeff_master_{isotope_suffix} * c.{master_species}_{isotope_suffix},\n"
        code += "            has_solid=has_solid,\n"
        code += f"            reaction_name='{reaction_name}_{isotope_suffix}',\n"
        if ref_species:
            code += f"            ref_species='{ref_species}',\n"
        elif "POC" in reactants:
            code += "            ref_species=poc_species,\n"
        else:
            code += f"            ref_species='{master_species}',\n"
        code += f"            stoich_ref={ref_stoich},\n"
        code += "        )\n"

    return code

def load_reactions_from_py(filepath: Path):
    spec = importlib.util.spec_from_file_location("chem_eqs", filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules["chem_eqs"] = module
    spec.loader.exec_module(module)
    return module.reactions

class RichArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            from rich.table import Table

            console = Console(file=file)
            if self.description:
                console.print(Markdown(self.description.strip()))
                console.print()

            usage_str = self.format_usage()
            console.print(f"[bold]Usage:[/bold] {usage_str.strip().replace('usage: ', '')}")
            console.print()

            table = Table(box=None, padding=(0, 2), show_header=False)
            table.add_column("Option", style="cyan", no_wrap=True)
            table.add_column("Description")

            for action in self._actions:
                if action.option_strings:
                    opts = ", ".join(action.option_strings)
                    metavar = f" {action.metavar}" if action.metavar and action.option_strings else ""
                    help_text = action.help or ""
                    if action.default and action.default != argparse.SUPPRESS:
                        help_text += f" (default: {action.default})"
                    table.add_row(f"{opts}{metavar}", help_text)

            console.print("[bold]Options:[/bold]")
            console.print(table)
        except Exception:
            super().print_help(file)

def main():
    parser = RichArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="chemical_equations.py",
        help="Path to Python (or JSON) file containing reaction configurations. Defaults to chemical_equations.py in the current working directory."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="generated_equations.py",
        help="Path to save the generated Python reactions code. Defaults to generated_equations.py in the current working directory."
    )
    parser.add_argument(
        "--monod-scheme",
        type=str,
        choices=["picard", "hybrid", "newton"],
        default="picard",
        help="Linearization scheme for Monod terms: 'picard' (legacy secant), 'hybrid' (safeguarded Newton), or 'newton' (full Newton-Raphson). Defaults to 'picard'."
    )
    
    # Print help text and exit if called without any arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file {args.input} not found.")
        sys.exit(1)

    if input_path.suffix == ".py":
        configs = load_reactions_from_py(input_path)
    elif input_path.suffix == ".json":
        with open(input_path, "r") as f:
            configs = json.load(f)
    else:
        print(f"Unsupported input file type: {input_path.suffix}. Please provide a .py or .json file.")
        sys.exit(1)

    generated_code = "# Auto-generated reaction functions\n\n"
    if args.monod_scheme == "picard":
        generated_code += "from fipyrite.diff_lib import add_coupled_reaction, add_implicit_sink, calculate_fractionated_coeff_32, partition_equilibrium_isotope_32\n\n"
    else:
        generated_code += "from fipyrite.diff_lib import add_coupled_reaction, add_implicit_sink, add_monod_sink, calculate_fractionated_coeff_32, partition_equilibrium_isotope_32\n\n"

    for cfg in configs:
        cfg_copy = dict(cfg)
        if "monod_scheme" not in cfg_copy:
            cfg_copy["monod_scheme"] = args.monod_scheme
        code = create_reaction(**cfg_copy)
        generated_code += code + "\n\n"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(generated_code)

    print(f"Successfully generated reactions code saved to {output_path}")

if __name__ == "__main__":
    main()
