"""fes_reactions.py

Declarative definition of diagenetic chemical reactions and kinetic limiters.

Template Structure & Design:
----------------------------
1. `limiters`:
   Dictionary mapping unique limiter names to their mathematical expressions.
   Moving limiters here keeps all kinetic terms and thresholds adjacent to the
   reactions they regulate, eliminating detached limiter scripts.

2. `reactions`:
   List of chemical reaction specifications. Each entry contains:
   - reaction: Stoichiometric equation string (e.g., '1 FeS + 1 HS -> 1 FeS2').
     Reactants and products are parsed and verified against species.py.
     Dynamic equilibrium species (like HS and Fe2_liq) are recognized automatically.
   - reaction_name: Unique identifier for reporting and logging.
   - k_value_name: Rate constant identifier defined in kinetic_constants_slow.py.
     Can be a string (e.g. 'POC_fast') or a dict for multi-branch reactions:
     {'precipitation': 'FeS_isp', 'dissolution': 'FeS_isd', 'equilibrium': 'FeS_sp'}.
   - limiters: Dict or list of limiters applied to this reaction.
   - isotope_species: (Optional) Explicit mapping between base and 32S companion species.
   - isotope_fractionation: (Optional) Isotope fractionation factor and limiter.
"""

limiters = {
    # Base implicit limiters: L_imp = 1.0 / (C + K)
    # The parser automatically derives explicit (C * L_imp) and inhibition (K * L_imp = 1 - C * L_imp)
    "O2": "1.0 / (c.O2 + K_O2)",
    "O2_TS2": "1.0 / (c.O2 + mp.K_O2_TS2)",
    "TS2": "1.0 / (c.TS2 + K_TS2)",
    "SO4": "1.0 / (c.SO4 + K_SO4)",
    "Fe3": "1.0 / (np.maximum(c.Fe3, 0.0) + K_Fe3)",
    "Fe3_diss_red": "1.0 / (np.maximum(c.Fe3, 0.0) + K_Fe3_diss_red)",

    # Custom isotope explicit fractionation limiters 
    "SO4_alpha_explicit": "c.SO4 / (c.SO4 + 1e-3)",
    "TS2_alpha_explicit": "c.TS2 / (c.TS2 + 1e-3)",
}

reactions = [
    # -------------------------------------------------------------------------
    # Organic Carbon Primary Respiration
    # -------------------------------------------------------------------------
    {
        "reaction_name": "aerobic_respiration_fast",
        "reaction": "POC_fast + O2 -> CO2",
        "k_value_name": "POC_fast",
        "limiters": ["O2"],
    },
    {
        "reaction_name": "aerobic_respiration_slow",
        "reaction": "POC_slow + O2 -> CO2",
        "k_value_name": "POC_slow",
        "limiters": ["O2"],
    },
    {
        "reaction_name": "dissimilatory_iron_reduction_fast",
        "reaction": "POC_fast + 4 Fe3 -> 4 Fe2_total",
        "k_value_name": "POC_fast",
        "limiters": ["Fe3_diss_red"],
        "inhibitors": ["O2"],
    },
    {
        "reaction_name": "dissimilatory_iron_reduction_slow",
        "reaction": "POC_slow + 4 Fe3 -> 4 Fe2_total",
        "k_value_name": "POC_slow",
        "limiters": ["Fe3_diss_red"],
        "inhibitors": ["O2"],
    },
    {
        "reaction_name": "sulfate_reduction_fast",
        "reaction": "2 POC_fast + SO4 -> TS2",
        "k_value_name": "POC_fast",
        "limiters": ["SO4"],
        "inhibitors": ["O2", "Fe3_diss_red"],
        "isotope_species": {"SO4": "SO4_32", "TS2": "TS2_32"},
        "isotope_fractionation": {
            "SO4_32 to TS2_32": ["msr_alpha", "SO4_alpha_explicit"]
        },
    },
    {
        "reaction_name": "sulfate_reduction_slow",
        "reaction": "2 POC_slow + SO4 -> TS2",
        "k_value_name": "POC_slow",
        "limiters": ["SO4"],
        "inhibitors": ["O2", "Fe3_diss_red"],
        "isotope_species": {"SO4": "SO4_32", "TS2": "TS2_32"},
        "isotope_fractionation": {
            "SO4_32 to TS2_32": ["msr_alpha", "SO4_alpha_explicit"]
        },
    },

    # -------------------------------------------------------------------------
    # Secondary Redox & Sulfide Oxidation
    # -------------------------------------------------------------------------
    {
        "reaction_name": "hs_oxidation",
        "reaction": "1 HS + 2 O2 -> 1 SO4",
        "k_value_name": "TS2_O2",
        "limiters": ["O2_TS2"],
        "isotope_species": {"SO4": "SO4_32", "TS2": "TS2_32", "HS": "HS_32"},
        "isotope_fractionation": {
            "HS_32 to S0_32": ["TS2_O2_alpha", "TS2_alpha_explicit"]
        },
    },
    {
        "reaction_name": "sulfide_mediated_iron_reduction_velde",
        "reaction": "HS + 8 Fe3 -> SO4 + 8 Fe2_total",
        "k_value_name": "Fe3_hs",
        "limiters": ["Fe3"],
        "inhibitors": ["O2"],
        "isotope_species": {"SO4": "SO4_32", "TS2": "TS2_32", "HS": "HS_32"},
    },
    {
        "reaction_name": "Fe2_oxidation",
        "reaction": "4 Fe2_total + O2 -> 4 Fe3",
        "k_value_name": "Fe2_O2",
    },
    {
        "reaction_name": "FeS_oxidation",
        "reaction": "4 FeS + 9 O2 -> 4 Fe3 + 4 SO4",
        "k_value_name": "FeS_O2",
    },

    # -------------------------------------------------------------------------
    # Mineral Precipitation & Dissolution (Smooth Transition)
    # -------------------------------------------------------------------------
    {
        "reaction_name": "FeS_precipitation_dissolution",
        "reaction": "Fe2_liq + HS -> FeS",
        "k_value_name": {
            "precipitation": "FeS_isp",
            "dissolution": "FeS_isd",
            "equilibrium": "FeS_sp",
        },
        "isotope_species": {"TS2": "TS2_32", "FeS": "FeS_32", "HS": "HS_32"},
    },
]

