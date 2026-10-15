"""species.py

Declarative definition of all chemical species in the diagenetic system.

Template Format & Structure:
----------------------------
Each key in the `species` dictionary represents a chemical species name:
- phase: 'liquid', 'solid', or 'mixed'.
    * 'liquid': Dissolved species in porewater (scaled by porosity phi).
    * 'solid': Mineral or particulate species (scaled by 1 - phi).
    * 'mixed': Lumped pool (e.g. Fe2_total) partitioned into porewater and sorbed phase.
- include: bool.
    * True: Active transport PDE variable solved by FiPy / Newton solver.
    * False: Equilibrium, diagnostic, or fixed external species.
- formula: Chemical formula string (e.g., 'FeS2', 'SO4', 'CH2O6').
- equilibrium_of: (Optional) Parent species for fast dynamic speciation (e.g., 'TS2').
- fraction: (Optional) Analytical expression or constant defining the fraction of parent.
- partition: (Optional for 'mixed') Fractions partitioned into dissolved and sorbed phases.
- diffusion: (Optional) Recipe for molecular diffusion coefficient calculation:
    * dict with {'type': 'diff_coeff', 'm0': float, 'm1': float} (Boudreau 1996)
    * dict with {'type': 'boudreau_o2'} (temperature & Archie porosity corrected O2 diffusion)
    * str pointing to another species name to inherit its diffusion coefficient.
    * None for solid species (which only transport via sediment burial and bioturbation).
- bc: (Optional) Default boundary condition dict:
    * {'top': float or pint-string, 'type': 'dissolved' or 'particulate'}
    * For Hplus: {'pH': 7.5} sets porewater pH as the single source of truth.
- isotope: (Optional) Name of corresponding isotopic companion species (e.g., 'SO4_32').
- isotope_of: (Optional) Base species name if this entry is an isotope companion.
- delta_threshold: (Optional) Minimum concentration threshold for calculating delta values (mmol/L).
"""

import numpy as np


# -----------------------------------------------------------------------------
# Isotope Reference Standards
# -----------------------------------------------------------------------------
isotope_standards: dict = {
    "S": {
        "standard": "VCDT",
        "ratio": 0.044162589,  # 34S / 32S
        "symbol": "d34S",
        "unit": "permil",
    },
}


"""
Most species are covered by the diff_coeff function with the m0, m1 parameter listed in
Boudreau 1996. If no parameters are availble, please provide any custom functions here.
Note that diffusions coefficients are only needed for species that are par of the
reaction-transport martrix (i.e., where include = True).
"""
def boudreau_o2(T_profile, phi):
    """Temperature porosity corrected O2 molecular diffusion."""
    phi_val = phi.value if hasattr(phi, "value") else phi
    return (
        (0.2604 + 0.006363 * (T_profile + 273.15))
        * 1e-9
        / (1.0 - np.log(phi_val**2))
    )


species = {
    # -------------------------------------------------------------------------
    # Dissolved Liquid Species (Porewater Transport)
    # -------------------------------------------------------------------------
    "SO4": {
        "phase": "liquid",
        "include": True,
        "formula": "SO4",
        "diff_coeff_function": "diff_coeff",
        "diffusion": {"m0": 4.88, "m1": 0.232},
        "bc": {"top": 28.2, "type": "dissolved", "delta": 21.0},
        "isotope": "SO4_32",
    },
    "SO4_32": {
        "phase": "liquid",
        "include": True,
        "formula": "SO4",
        "isotope_of": "SO4",
        "diffusion": "SO4",
    },
    "TS2": {
        "phase": "liquid",
        "include": True,
        "formula": "H2S",
        "diff_coeff_function": "diff_coeff",
        "diffusion": {"m0": 43.3, "m1": 0.85},
        "bc": {"top": 0.0, "type": "dissolved", "delta": 0.0},
        "isotope": "TS2_32",
    },
    "TS2_32": {
        "phase": "liquid",
        "include": True,
        "formula": "H2S",
        "isotope_of": "TS2",
        "diffusion": "TS2",
    },
    "O2": {
        "phase": "liquid",
        "include": True,
        "formula": "O2",
        "diff_coeff_function": "boudreau_o2",
        "diffusion": None,
        "bc": {"top": 0.28, "type": "dissolved"},
    },

    # -------------------------------------------------------------------------
    # Dynamic Speciation & Acid-Base Equilibrium
    # -------------------------------------------------------------------------
    "Hplus": {
        "phase": "liquid",
        "include": False,
        "formula": "H",
        "bc": {"pH": 7.5},  # Single source of truth for porewater pH
    },
    "HS": {
        "phase": "liquid",
        "include": False,
        "formula": "HS",
        "equilibrium_of": "TS2",
        "fraction": "Ka1 / (Hplus + Ka1)",
        "constants": {"pKa1": 7.0},
    },
    "HS_32": {
        "phase": "liquid",
        "include": False,
        "formula": "HS",
        "isotope_of": "HS",
    },
    "H2S": {
        "phase": "liquid",
        "include": False,
        "formula": "H2S",
        "equilibrium_of": "TS2",
        "fraction": "Hplus / (Hplus + Ka1)",
        "constants": {"pKa1": 7.0},
    },

    # -------------------------------------------------------------------------
    # Mixed / Lumped Species (Porewater + Sorbed Solid)
    # -------------------------------------------------------------------------
    "Fe2_total": {
        "phase": "mixed",
        "include": True,
        "formula": "Fe2",
        "partition": {
            "dissolved": "1.0 / (1.0 + k.Fe2_p_eq)",
            "sorbed": "k.Fe2_p_eq / (1.0 + k.Fe2_p_eq)",
        },
        "diff_coeff_function": "diff_coeff",
        "diffusion": {"m0": 27.7, "m1": 1.0},
        "bc": {"top": 0.0, "type": "dissolved"},
    },
    "Fe2_liq": {
        "phase": "liquid",
        "include": False,
        "formula": "Fe",
        "alias_of": "Fe2_total.dissolved",
    },
    "Fe2_sorbed": {
        "phase": "solid",
        "include": False,
        "formula": "Fe",
        "alias_of": "Fe2_total.sorbed",
    },

    # -------------------------------------------------------------------------
    # Solid Mineral & Particulate Species (Burial + Bioturbation)
    # -------------------------------------------------------------------------
    "POC_fast": {
        "phase": "solid",
        "include": True,
        "formula": "CH2O6",
        "diffusion": None,
        "bc": {"top": "365 umol/(cm^2 * yr)", "type": "particulate"},
    },
    "POC_slow": {
        "phase": "solid",
        "include": True,
        "formula": "CH2O6",
        "diffusion": None,
        "bc": {"top": "183 umol/(cm^2 * yr)", "type": "particulate"},
    },
    "Fe3": {
        "phase": "solid",
        "include": True,
        "formula": "FeOOH",
        "diffusion": None,
        "bc": {"top": "12 umol/(cm^2 * yr)", "type": "particulate"},
    },
    "FeS": {
        "phase": "solid",
        "include": True,
        "formula": "FeS",
        "diffusion": None,
        "bc": {"top": 0.0, "type": "particulate"},
        "isotope": "FeS_32",
    },
    "FeS_32": {
        "phase": "solid",
        "include": True,
        "formula": "FeS",
        "isotope_of": "FeS",
        "diffusion": None,
    },
    "FeS2": {
        "phase": "solid",
        "include": True,
        "formula": "FeS2",
        "diffusion": None,
        "bc": {"top": 0.0, "type": "particulate"},
        "isotope": "FeS2_32",
        "stoichiometry": 2.0,
        "isotope_standard": "S",
    },
    "FeS2_32": {
        "phase": "solid",
        "include": True,
        "formula": "FeS2",
        "isotope_of": "FeS2",
        "diffusion": None,
    },
    "S0": {
        "phase": "solid",
        "include": True,
        "formula": "S",
        "diffusion": None,
        "bc": {"top": 0.0, "type": "particulate", "delta": 8.0},
        "isotope": "S0_32",
    },
    "S0_32": {
        "phase": "solid",
        "include": True,
        "formula": "S",
        "isotope_of": "S0",
        "diffusion": None,
    },

    # External / Reaction Balance Species (Excluded from transport)
    "CO2": {"phase": "liquid", "include": False, "formula": "CO2"},
    "H2O": {"phase": "liquid", "include": False, "formula": "H2O"},
}
