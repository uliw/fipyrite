# chemical_equations.py
# This file contains only the declarative definitions for the chemical reactions.
# It does not execute any code.

reactions = [
    # --- Class 1 & 2 POC Reactions ---
    {
        "reaction": "POC + O2 -> CO2",
        "reaction_name": "aerobic_respiration",
        "k_value_name": "poc_k",
        "limiters": ["O2_implicit"],
    },
    {
        "reaction": "POC + 4 Fe3 -> 4 Fe2_total",
        "reaction_name": "dissimilatory_iron_reduction",
        "k_value_name": "poc_k",
        "limiters": ["O2_inhibit", "Fe3_diss_red_implicit"],
    },
    {
        "reaction": "2 POC + SO4 -> TS2",
        "reaction_name": "sulfate_reduction",
        "k_value_name": "poc_k",
        "limiters": ["O2_inhibit", "SO4_implicit", "Fe3_diss_red_inhib"],
        # FIXME: does fractionation need to be a list of lists?
        "fractionation": [["SO4", "TS2", "msr_alpha", "SO4_alpha_explicit"]],
    },
    # --- Class 1 & 2 Secondary Redox Reactions ---
    {
        "reaction": "2 HS + O2 -> 2 S0",
        "reaction_name": "hs_oxidation",
        "k_value_name": "TS2_O2",
        "dynamic_variables": {"HS": "c.TS2 * mp.hs_frac"},
        # FIXME: does fractionation need to be a list of lists?
        "fractionation": [["HS", "S0", "TS2_O2_alpha", "TS2_alpha_explicit"]],
    },
    {
        "reaction": "HS + 2 O2 -> SO4",
        "reaction_name": "hs_oxidation_velde",
        "k_value_name": "TS2_O2",
        "dynamic_variables": {"HS": "c.TS2 * mp.hs_frac"},
        # FIXME: does fractionation need to be a list of lists?
        "fractionation": [["HS", "SO4", "TS2_O2_alpha", "TS2_alpha_explicit"]],
        "limiters": ["O2_implicit_TS2"],
    },
    {
        "reaction": "2 S0 + 3 O2 -> 2 SO4",
        "reaction_name": "elemental_sulfur_oxidation",
        "k_value_name": "S0_O2",
    },
    {
        "reaction": "HS + 2 Fe3 -> S0 + 2 Fe2_total",
        "reaction_name": "sulfide_mediated_iron_reduction",
        "k_value_name": "Fe3_hs",
        "dynamic_variables": {"HS": "c.TS2 * mp.hs_frac"},
        "limiters": ["O2_inhibit", "Fe3_implicit"],
    },
    {
        "reaction": "HS + 8 Fe3 -> SO4 + 8 Fe2_total",
        "reaction_name": "sulfide_mediated_iron_reduction_velde",
        "k_value_name": "Fe3_hs",
        "dynamic_variables": {"HS": "c.TS2 * mp.hs_frac"},
        "limiters": ["O2_inhibit", "Fe3_implicit"],
    },
    {
        "reaction": "4 Fe2_total + O2 -> 4 Fe3",
        "reaction_name": "Fe2_oxidation",
        "k_value_name": "Fe2_O2",
    },
    {
        "reaction_name": "FeS_oxidation",
        "reaction": "4 FeS + 9 O2 -> 4 Fe3 + 4 SO4",
        "k_value_name": "FeS_O2",
    },
    # Currently missing, can these be used with the above scheme?
    {
        "reaction_name": "pyrite_oxidation_new",
        "reaction": "2 FeS2 + 7 O2 -> 2 Fe3 + 4 SO4",
        "k_value_name": "FeS2_O2",
    },
    {
        # this may need special attention wrt isotope handling since
        # we have two sulfur sources
        "reaction_name": "pyrite_formation_fes_s0_new",
        "reaction": "1 FeS + 1 S0 -> 1 FeS2",
        "k_value_name": "FeS_S0",
    },
    {
        # this may need special attention wrt isotope handling since
        # we have two sulfur sources
        "reaction_name": "pyrite_formation_fes_ts2_new",
        "reaction": "1 FeS + 1 HS -> 1 FeS2",
        "k_value_name": "FeS_TS2",
    },
]
