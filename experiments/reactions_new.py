"""Define the reactions.

Note on numerix (nx):
We use `fipy.tools.numerix` instead of standard `numpy` because the reaction functions
are called polymorphically:
1) During setup/plotting, they are called with FiPy `CellVariable`s to build lazy-evaluated equation trees.
2) During time-stepping, they are called with `ArrayProxy` wrapping raw NumPy arrays for high speed.
Using `nx` allows these mathematical operations (e.g. `nx.maximum`, `nx.sqrt`) to build lazy expression
trees in (1) and run at full NumPy speeds in (2). Standard `numpy` would force immediate evaluation in (1).
"""

from __future__ import annotations  # noqa: I001

from fipy.tools import numerix as nx

from fipyrite.diff_lib import (
    add_coupled_reaction,
    add_explicit_source,
    add_implicit_coupling_new,
    add_implicit_sink,
    calculate_fractionated_coeff_32,
    partition_equilibrium_isotope_32,
    smooth_ramp,
)

from ggg import (
    aerobic_respiration,
    dissimilatory_iron_reduction,
    sulfate_reduction,
    hs_oxidation,
    hs_oxidation_velde,
    elemental_sulfur_oxidation,
    sulfide_mediated_iron_reduction,
    sulfide_mediated_iron_reduction_velde,
    Fe2_oxidation,
    FeS_oxidation,
    pyrite_formation_fes_ts2_new,
    pyrite_oxidation_new,
    pyrite_formation_fes_s0_new,
)


def equilibrium_reactions(mp, c, k, f, RATES, dt):
    """Instantenous reactions.

    that are calculated after the
    transport matrix has been solved.
    """
    import inspect
    
    for r in mp.instantenous_reactions:
        sig = inspect.signature(r[0])
        if "f" in sig.parameters:
            r[0](c, r[1], mp, dt, RATES, f=f)
        else:
            r[0](c, r[1], mp, dt, RATES)

    return f, RATES


def diagenetic_reactions(mp, c, k, f):
    """
    Orchestrate diagenetic reactions inside the transport matrix.

    Calculates limiters, initializes matrices, and calls specific process functions.

    Porosity Handling:

    Model units are meter/second, concentrations are given mmol/liter (mol/m^3) and
    solids are expressed as concentration per unit of solid volume (mmol/L_solid).

    This keeps the physics of the "solid phase" independent of how much water is
    currently squeezing around it.  If the sediment compacts (porosity ϕ decreases), the
    amount of organic matter per gram of rock doesn't change, but the amount of organic
    matter per liter of bulk sediment does.  ​

    As such, a reaction between a liquid and a solid needs to be scaled

    f = k * [SO4] * (1 - phi)/phi * [OM]

    This is now handled by the maxtrix helper functions via the has_solids parameter, where
    has_solid indicates weather the reaction inolves solids. E.g.,
    when calculting the consumption of organic matter by sulfate you would write

    # POC Sink - SOLID
    coeff_poc = k.poc_O2 * c.SO4
    add_implicit_sink(
        LHS, RATES, "poc", coeff_poc, rate_base, has_solid=has_solid, mp=mp, c=c
    )

    whereas the consumption of sulfate would be

    coeff_SO4 = k.poc_SO4 * c.poc
    add_implicit_sink(LHS, RATES, "SO4", coeff_SO4, SO4_rate, has_solid=has_solid, mp=mp, c=c)

    or as a coupled reaction:

    add_implicit_coupling_new(
        CROSS,  #  Off-diagonal coupling matrix
        RATES,  #  Rate reporting dictionary
        LHS,  # Diagonal matrix (implicit sinks)
        "TS2",  # species that is produced
        "SO4",  # species that is consumed
        coeff_SO4,  # reaction coefficient
        SO4_rate,  # coeff * concentration
        mp=mp,
        has_solid= has_solid # type
        c=c,  # model parameters
    )

    the porosity correction will then applied automatically depending on the
    has_solid parameter,
    """
    # 1. SETUP & INITIALIZATION
    # -------------------------
    species_list = list(c.keys())

    # Store global reaction constants dictionary in mp
    mp.k = k

    # Accumulators (The State)
    # LHS: Diagonal (Self) Coefficients (Implicit Sinks)
    # LHS = {s: nx.zeros_like(c.SO4.value) for s in species_list}
    LHS = {s: nx.zeros_like(c.SO4) for s in species_list}

    # CROSS: Off-Diagonal / Coupled Terms
    CROSS = {s: [] for s in species_list}

    RHS = {s: nx.zeros_like(c.SO4) for s in species_list}
    RATES = {s: nx.zeros_like(c.SO4) for s in species_list}

    # 2. CALCULATE LIMITERS
    # ---------------------
    limiters = {}

    # Velde et al specify their k-values in bulk concentrations, so
    # we convert to phase specific space first
    phi = mp.phi
    K_O2 = mp.K_O2 / phi
    K_TS2 = mp.K_TS2 / phi
    K_SO4 = mp.K_SO4 / phi
    K_Fe3 = mp.K_Fe3 / (1 - phi)
    K_Fe3_diss_red = mp.K_Fe3_diss_red / (1 - phi)

    # O2 Limitation for low O2 (1.0 -> 0.0)
    limiters["O2_implicit"] = 1 / (c.O2 + K_O2)

    # O2 Limitation for low O2 (1.0 -> 0.0)
    limiters["O2_implicit_TS2"] = 1 / (c.O2 + mp.K_O2_TS2)
    
    # O2 Inhibition for high O2 (1.0 -> 0.0)
    limiters["O2_inhibit"] = K_O2 / (c.O2 + K_O2)

    # TS2 inhibition for high TS2
    limiters["TS2"] = K_TS2 / (c.TS2 + K_TS2)

    # Sulfate Limiter (Implicit 1/[S+K] and Explicit [S]/[S+K])
    limiters["SO4_implicit"] = 1.0 / (c.SO4 + K_SO4)
    limiters["SO4_explicit"] = c.SO4 / (c.SO4 + K_SO4)

    # isotope enrichment during MSR
    limiters["SO4_alpha_explicit"] = c.SO4 / (c.SO4 + mp.K_epsilon_msr)
    # isotope enrichment during HS oxidation
    limiters["TS2_alpha_explicit"] = c.TS2 / (c.TS2 + mp.K_epsilon_TS2_O2)

    # Fe3 limiters
    limiters["Fe3_implicit"] = 1.0 / (c.Fe3 + K_Fe3)
    limiters["Fe3_diss_red_implicit"] = 1.0 / (c.Fe3 + K_Fe3_diss_red)
    limiters["Fe3_diss_red_inhib"] = K_Fe3_diss_red / (c.Fe3 + K_Fe3_diss_red)

    # 3. RUN PROCESSES
    # ----------------
    # Each function updates LHS, RHS, and RATES in place
    # r is list of list where r[0] is the function and r[1] is the k val

    for r in mp.diagenetic_reactions:
        r[0](c, r[1], limiters, LHS, RHS, RATES, CROSS, mp)

    for s in species_list:
        lhs_coeff = LHS.get(s, 0.0)
        cross_list = CROSS.get(s, [])

        cross_term = None

        setattr(f, s, (lhs_coeff, RHS[s], RATES[s], cross_term))

    # Pack process-specific rates dynamically
    for key, val in RATES.items():
        if key not in species_list:
            setattr(f, key, (None, None, val, None))

    # Store raw accumulators for static allocation (Option C)
    object.__setattr__(f, "raw_LHS", LHS)
    object.__setattr__(f, "raw_CROSS", CROSS)
    object.__setattr__(f, "raw_RHS", RHS)

    return f, RATES


# =============================================================================
# PROCESS FUNCTIONS (The Biogeochemistry)
# =============================================================================

# def sulfide_speciation_clip(c, k, mp, dt, RATES, f=None):
#     """Update reporting species (h2s, hs) based on total sulfide (TS2) and pH.

#     This is FYI only and does not affect the reaction rates, which are all based on TS2.
#     """
#     c.h2s.setValue(c.TS2 * mp.h2s_frac)
#     c.hs.setValue(c.TS2 * mp.hs_frac)

#     if mp.isotopes:
#         hs_frac_val = getattr(mp.hs_frac, "value", mp.hs_frac)
#         h2s_frac_val = getattr(mp.h2s_frac, "value", mp.h2s_frac)
#         alpha_val = getattr(mp.h2s_hs_alpha, "value", mp.h2s_hs_alpha)

#         denom = hs_frac_val + alpha_val * h2s_frac_val + 1e-30
#         c.hs_32.setValue(c.TS2_32 * hs_frac_val / denom)
#         c.h2s_32.setValue(c.TS2_32 * alpha_val * h2s_frac_val / denom)


# def Fe2_sorption_clip(c, k, mp, dt, RATES, f=None):
#     """Handle Iron Partitioning algebraically.

#     Instead of calculating rates, we calculate fractions.

#     System State: 'fe_total' is the primary variable.
#     Fe2 (liquid) and Fe2_p (solid) are derived helper views.
#     These are not used in the equations, and FYI only.
#     """
#     c.Fe2.setValue(c.Fe2_total * mp.Fe2_diss)
#     c.Fe2_p.setValue(c.Fe2_total * mp.Fe2_sorb)



# def pyrite_formation_S0(c, k, lim, LHS, RHS, RATES, CROSS, mp):
#     """Reaction: 1 FeS + 1 S0 -> 1 FeS2.

#     This is a bit tricky as we have two different S atoms into the same
#     target (FeS2)
#     """
#     has_solid = True  # True if the reactants contain a solid phase species
#     # S0 Sink - SOLID
#     coeff_S0 = k.FeS_S0 * c.FeS
#     add_implicit_sink(
#         LHS,
#         RATES,
#         "S0",
#         coeff_S0,
#         coeff_S0 * c.S0,
#         mp=mp,
#         has_solid=has_solid,
#         c=c,
#     )

#     # FeS to FeS2 SOLID, Rate = k * FeS * S0.
#     coeff_FeS = k.FeS_S0 * c.S0
#     add_coupled_reaction(
#         CROSS=CROSS,
#         LHS=LHS,
#         RATES=RATES,
#         mp=mp,
#         master_species="FeS",
#         reactants={},
#         products={"FeS2": 1.0},
#         coeff_master=coeff_FeS,
#         rate_master=coeff_FeS * c.FeS,
#         has_solid=has_solid,
#         reaction_name="pyrite_formation_S0_FeS",
#         ref_species="FeS",
#     )

#     if mp.isotopes:
#         # S0 is porewater → must include mp.fac_s to match bulk sink coefficient,
#         # and use "liquid_2_solid" for correct volume conversion to FeS2_32 (solid)
 
#         # 1st S atom: from S0_32 (porewater) to FeS2_32 (solid)
#         add_coupled_reaction(
#             CROSS=CROSS,
#             LHS=LHS,
#             RATES=RATES,
#             mp=mp,
#             master_species="S0_32",
#             reactants={},
#             products={"FeS2_32": 1.0},
#             coeff_master=k.FeS_S0 * c.FeS,
#             rate_master=k.FeS_S0 * c.FeS * c.S0_32,
#             has_solid=has_solid,
#             reaction_name="pyrite_formation_S0_32",
#             ref_species="FeS",
#         )
 
#         # 2nd S atom: from FeS_32 (solid) to FeS2_32 (solid)
#         add_coupled_reaction(
#             CROSS=CROSS,
#             LHS=LHS,
#             RATES=RATES,
#             mp=mp,
#             master_species="FeS_32",
#             reactants={},
#             products={"FeS2_32": 1.0},
#             coeff_master=k.FeS_S0 * c.S0,
#             rate_master=k.FeS_S0 * c.S0 * c.FeS_32,
#             has_solid=has_solid,
#             reaction_name="pyrite_formation_FeS_32",
#             ref_species="FeS",
#         )


def pyrite_formation_FeS_TS2(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Reaction: 1 FeS + 1 HS -> 1 FeS2Fe."""
    has_solid = True  # True if the reactants contain a solid phase species
    # H2S Sink (1.0x) - LIQUID
    coeff_TS2 = k.FeS_TS2 * c.FeS * mp.hs_frac
    add_implicit_sink(
        LHS,
        RATES,
        "TS2",
        coeff_TS2,
        coeff_TS2 * c.TS2,
        mp=mp,
        has_solid=has_solid,
        c=c,
    )

    # FeS2 Source (1.0x) - SOLID
    # Couple to FeS.
    # Rate = k * H2S * FeS.
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="FeS",
        reactants={},
        products={"FeS2": 1.0},
        coeff_master=k.FeS_TS2 * c.TS2 * mp.hs_frac,
        rate_master=k.FeS_TS2 * c.TS2 * c.FeS * mp.hs_frac,
        has_solid=has_solid,
        reaction_name="pyrite_formation_FeS_TS2",
        ref_species="FeS",
    )

    if mp.isotopes:
        # 1st S atom: from FeS_32 (solid) -> FeS2_32 (solid)
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="FeS_32",
            reactants={},
            products={"FeS2_32": 1.0},
            coeff_master=k.FeS_TS2 * c.TS2 * mp.hs_frac,
            rate_master=k.FeS_TS2 * c.TS2 * c.FeS_32 * mp.hs_frac,
            has_solid=has_solid,
            reaction_name="pyrite_formation_FeS_TS2_32_solid",
            ref_species="FeS",
        )

        # 2nd S atom: from H2S_32 (TS2_32) (liquid) -> FeS2_32 (solid)
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="TS2_32",
            reactants={},
            products={"FeS2_32": 1.0},
            coeff_master=coeff_TS2,
            rate_master=coeff_TS2 * c.TS2_32,
            has_solid=has_solid,
            reaction_name="pyrite_formation_FeS_TS2_32_liquid",
            ref_species="FeS",
        )


def pyrite_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    Reaction: 1 FeS2 + 3.5 O2 → 1 Fe3 + 2 SO4.

    Coupling strategy:
      - SO4 is cross-coupled to O2  (liquid_2_liquid, stoich 2/3.5)
      - Fe3 is cross-coupled to FeS2 (solid_2_solid, stoich 1.0)
    """
    has_solid = True  # True if the reactants contain a solid phase species
    # ------------------------------------------------------------------
    # 1. Base coefficients
    # ------------------------------------------------------------------
    # O2 is porewater master: coeff_O2 is implicit sink on O2
    coeff_O2 = k.FeS2_O2 * c.FeS2  # [L_pw basis]
    coeff_FeS2 = k.FeS2_O2 * c.O2
    rate_O2 = coeff_O2 * c.O2  # mol/L_pw/s
    rate_FeS2 = coeff_FeS2 * c.FeS2  # mol/L_pw/s

    # 1. O2 -> SO4 (2/3.5 ratio)
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="O2",
        reactants={"FeS2": 1.0 / 3.5},
        products={"SO4": 2.0 / 3.5},
        coeff_master=coeff_O2,
        rate_master=rate_O2,
        has_solid=has_solid,
        reaction_name="pyrite_oxidation_O2",
        ref_species="FeS2",
    )

    # 2. FeS2 -> Fe3
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="FeS2",
        reactants={},
        products={"Fe3": 1.0},
        coeff_master=coeff_FeS2,
        rate_master=rate_FeS2,
        has_solid=has_solid,
        reaction_name="pyrite_oxidation_FeS2",
        ref_species="FeS2",
    )

    # 3. Isotopes: FeS2_32 -> SO4_32
    if mp.isotopes:
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="FeS2_32",
            reactants={},
            products={"SO4_32": 1.0},
            coeff_master=coeff_FeS2,
            rate_master=coeff_FeS2 * c.FeS2_32,
            has_solid=has_solid,
            reaction_name="pyrite_oxidation_32",
            ref_species="FeS2",
        )


def S0_disproportionation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """Calculate elemental sulfur disproportionation.

    Reaction: 3S0 + 8H2O -> H2S + 2SO4

    Notes
    -----
    - The split between H2S and SO4 depends on mp.dispro_SO4_hs_split,
      which is the ratio between H2S/SO4, typically 1:2 -> 0.5
    - The isotope fractionation between S0 and H2S is given by mp.dispro_hs_alpha (0.993)
    - The isotope fractionation between S0 and SO4 is given by mp.dispro_SO4_alpha (1.02)
    - The reaction constant for the overall reaction is given by k.S0_dispro
    - The reaction rate depends on S0, and H2S & O2 as inhibitors.
    """
    has_solid = True  # True if the reactants contain a solid phase species
    # 1. Base Rate Calculation (Master Species: S0)
    # Disproportionation is anaerobic, so O2 is NOT a reactant.
    # Instead, we use the inhibitor lim["disp_O2_inhibit"] to ensure it only
    # proceeds under low oxygen conditions.
    rate_uncapped = k.S0_dispro * c.S0 * lim["TS2"] * lim["O2_inhibit"]

    # Capping to prevent over-consumption in a single timestep
    max_rate_S0 = 0.7 * c.S0 / (mp.current_dt + 1e-30)
    rate_actual = nx.minimum(rate_uncapped, max_rate_S0)

    # Coefficient based on the consumed species (S0)
    coeff_S0_base = rate_actual / (c.S0 + 1e-30)

    # 2. Calculate the Stoichiometric Split
    # If split = 0.5 (1 H2S : 2 SO4), then for 1.5 moles of S0:
    # 1.0 mole goes to SO4 and 0.5 moles go to H2S
    split = mp.dispro_SO4_hs_split
    SO4_fraction = 1.0 / (1.0 + split)
    h2s_fraction = split / (1.0 + split)

    # S0 Disproportionation coupling (using the wrapper)
    coeff_SO4 = coeff_S0_base * SO4_fraction
    coeff_TS2 = coeff_S0_base * h2s_fraction
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species="S0",
        reactants={},
        products={"SO4": SO4_fraction, "TS2": h2s_fraction},
        coeff_master=coeff_S0_base,
        rate_master=coeff_S0_base * c.S0,
        has_solid=has_solid,
        reaction_name="S0_disproportionation",
        ref_species="S0",
    )

    # O2 Consumption - REMOVED (Disproportionation is anaerobic)
    # If the user intended this to be S0 oxidation, O2 should be a reactant.
    # But for disproportionation, it is purely internal redox.

    # 5. Isotopes
    if mp.isotopes:
        # To maintain isotope mass balance, the 32S leaving S0 must exactly equal
        # the 32S entering H2S and SO4. If the user-provided alphas do not have a
        # weighted average of 1.0, mass is created/destroyed. We normalize them here:
        weighted_alpha = (
            h2s_fraction * mp.dispro_hs_alpha + SO4_fraction * mp.dispro_SO4_alpha
        )
        norm_hs_alpha = mp.dispro_hs_alpha / weighted_alpha
        norm_SO4_alpha = mp.dispro_SO4_alpha / weighted_alpha

        # Fractionation for H2S path
        coeff_hs_32 = calculate_fractionated_coeff_32(
            coeff_TS2, c.S0, c.S0_32, norm_hs_alpha, eps=1e-30
        )

        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="S0_32",
            reactants={},
            products={"TS2_32": 1.0},
            coeff_master=coeff_hs_32,
            rate_master=coeff_hs_32 * c.S0_32,
            has_solid=has_solid,
            reaction_name="S0_disproportionation_32_hs",
            ref_species="S0",
        )

        # Fractionation for SO4 path
        coeff_SO4_32 = calculate_fractionated_coeff_32(
            coeff_SO4, c.S0, c.S0_32, norm_SO4_alpha, eps=1e-30
        )

        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species="S0_32",
            reactants={},
            products={"SO4_32": 1.0},
            coeff_master=coeff_SO4_32,
            rate_master=coeff_SO4_32 * c.S0_32,
            has_solid=has_solid,
            reaction_name="S0_disproportionation_32_SO4",
            ref_species="S0",
        )

# def FeS_precipitation_dissolution_linearized(c, k, lim, LHS, RHS, RATES, CROSS, mp):
#     """
#     FeS precipitation / dissolution with Picard (Quasi-Linear) formulation.

#     Precipitation (Ω ≥ 1):  Fe2⁺ + HS⁻  →  FeS(s)
#         R_prec = k_prec_eff · (Ω - 1) / (Km + Ω - 1)
#                = prec_coeff_TS2 · TS2^{n+1}
#         where prec_coeff_TS2 = k_prec_eff · mm_factor_prec / (TS2 + 1e-30)

#     Dissolution   (Ω < 1):  FeS(s)  →  Fe2⁺ + HS⁻
#         R_diss = k_diss_eff · FeS · (1 - Ω) / (Km + 1 - Ω)
#                = diss_coeff_FeS · FeS^{n+1}
#         where diss_coeff_FeS = k_diss_eff · mm_factor_diss

#     Ω = (Fe2_total · Fe2_diss · TS2 · hs_frac) / (H⁺ · Ksp)

#     Advantages of Picard form:
#         - Zero explicit residual (no catastrophic cancellation in fast kinetics)
#         - Exact diagonal coupling with 1:1 stoichiometry
#         - Exact isotope ratio inheritance with zero off-diagonal stiffness
#     """
#     import numpy as np
    
#     phi = mp.phi
#     has_solid = True  # True for dissolution branch (solid reactant FeS)
#     has_solid_prec = (
#         False  # False for precipitation branch (liquid reactants Fe2_pw, HS-)
#     )

#     # ── Current sweep iterate ─────────────────────────────────────────────
#     Fe2_val = nx.maximum(c.Fe2_total.value, 1e-20)
#     TS2_val = nx.maximum(c.TS2.value, 1e-20)
#     FeS_val = nx.maximum(c.FeS.value, 1e-20)

#     Fe2_pw = Fe2_val * mp.Fe2_diss
#     hs_val = TS2_val * mp.hs_frac
#     omega_den = k.Hplus * k.FeS_sp + 1e-30
#     omega = Fe2_pw * hs_val / omega_den

#     # ── Per-cell branch selector ──────────────────────────────────────────
#     epsilon = 0.05
#     is_prec = nx.minimum(nx.maximum((omega - (1.0 - epsilon)) / (2.0 * epsilon), 0.0), 1.0)
#     is_diss = 1.0 - is_prec

#     Km = 0.5

#     # ══════════════════════════════════════════════════════════════════════
#     # PRECIPITATION BRANCH (Picard Form: implicit in TS2)
#     # ══════════════════════════════════════════════════════════════════════
#     k_prec_eff = k.FeS_isp * is_prec
    
#     df = nx.maximum(omega - 1.0, 1e-20)
#     mm_factor_prec = df / (Km + df)
#     is_prec_active = nx.where(omega >= 1.0, 1.0, 0.0)

#     # Picard pseudo-first-order coefficient in TS2 [1/s]
#     prec_coeff_TS2 = (k_prec_eff * mm_factor_prec / TS2_val) * is_prec_active
#     prec_rate_TS2 = prec_coeff_TS2 * TS2_val

#     # (a) FeS source coupled to TS2
#     add_implicit_coupling_new(
#         CROSS,
#         RATES,
#         LHS,
#         target_species="FeS",
#         source_species="TS2",
#         coeff=prec_coeff_TS2,
#         rate=prec_rate_TS2,
#         mp=mp,
#         has_solid=has_solid_prec,
#         add_lhs_sink=False,
#         stoich_ratio=1.0,
#     )

#     # (b) TS2 diagonal sink
#     add_implicit_sink(
#         LHS,
#         RATES,
#         "TS2",
#         prec_coeff_TS2,
#         prec_rate_TS2,
#         mp=mp,
#         has_solid=has_solid_prec,
#     )

#     # (c) Fe2_total 1:1 stoichiometric sink coupled to TS2
#     CROSS["Fe2_total"].append(("TS2", -prec_coeff_TS2 * phi))
#     RATES["Fe2_total"] -= prec_rate_TS2 * phi

#     # ══════════════════════════════════════════════════════════════════════
#     # DISSOLUTION BRANCH (Picard Form: implicit in FeS)
#     # ══════════════════════════════════════════════════════════════════════
#     k_diss_eff = k.FeS_isd * is_diss
    
#     us = nx.maximum(1.0 - omega, 1e-20)
#     mm_factor_diss = us / (Km + us)
#     is_diss_active = nx.where(omega < 1.0, 1.0, 0.0)

#     # Picard pseudo-first-order coefficient in FeS [1/s]
#     diss_coeff_FeS = k_diss_eff * mm_factor_diss * is_diss_active
#     diss_rate_FeS = diss_coeff_FeS * FeS_val

#     # (a) FeS diagonal sink & Fe2_total source
#     add_implicit_coupling_new(
#         CROSS,
#         RATES,
#         LHS,
#         target_species="Fe2_total",
#         source_species="FeS",
#         coeff=diss_coeff_FeS,
#         rate=diss_rate_FeS,
#         mp=mp,
#         has_solid=has_solid,
#         add_lhs_sink=True,
#         stoich_ratio=1.0,
#     )

#     # (b) TS2 source coupled to FeS
#     add_implicit_coupling_new(
#         CROSS,
#         RATES,
#         LHS,
#         target_species="TS2",
#         source_species="FeS",
#         coeff=diss_coeff_FeS,
#         rate=diss_rate_FeS,
#         mp=mp,
#         has_solid=has_solid,
#         add_lhs_sink=False,
#         stoich_ratio=1.0,
#     )

#     # ══════════════════════════════════════════════════════════════════════
#     # ISOTOPES
#     # ══════════════════════════════════════════════════════════════════════
#     if mp.isotopes:
#         # 1. Precipitation (TS2_32 -> FeS_32)
#         hs_32 = partition_equilibrium_isotope_32(
#             c.TS2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
#         )
#         hs_val_np = np.asarray(hs_val)
#         hs_32_val = np.asarray(hs_32)
#         TS2_val_np = np.asarray(c.TS2.value)
#         TS2_32_val = np.asarray(c.TS2_32.value)
        
#         f32_default = 1.0 / (1.0 + mp.VCDT)
#         f32_hs = np.where(hs_val_np > 1e-6, hs_32_val / (hs_val_np + 1e-30), f32_default)
#         f32_hs = np.clip(f32_hs, 0.5, 1.5)

#         # Bulk ratio of total sulfide for porewater sink (prevents d34S_TS2 distortion)
#         f32_TS2 = np.where(TS2_val_np > 1e-6, TS2_32_val / (TS2_val_np + 1e-30), f32_default)
#         f32_TS2 = np.clip(f32_TS2, 0.5, 1.5)

#         ratio_hs_ts2 = np.where(f32_TS2 > 1e-10, f32_hs / f32_TS2, 1.0)
#         prec_coeff_FeS_32 = prec_coeff_TS2 * ratio_hs_ts2

#         # (a) FeS_32 solid source
#         add_implicit_coupling_new(
#             CROSS,
#             RATES,
#             LHS,
#             target_species="FeS_32",
#             source_species="TS2_32",
#             coeff=prec_coeff_FeS_32,
#             rate=prec_rate_TS2 * f32_hs,
#             mp=mp,
#             has_solid=has_solid_prec,
#             add_lhs_sink=False,
#             stoich_ratio=1.0,
#         )

#         # (b) TS2_32 porewater diagonal sink
#         add_implicit_sink(
#             LHS,
#             RATES,
#             "TS2_32",
#             prec_coeff_TS2,
#             prec_rate_TS2 * f32_TS2,
#             mp=mp,
#             has_solid=has_solid_prec,
#         )

#         # 2. Dissolution (FeS_32 -> TS2_32)
#         FeS_val_np = np.asarray(c.FeS.value)
#         FeS_32_val = np.asarray(c.FeS_32.value)
#         f32_FeS = np.where(FeS_val_np > 1e-3, FeS_32_val / (FeS_val_np + 1e-30), f32_hs)
#         f32_FeS = np.clip(f32_FeS, 0.5, 1.5)

#         add_implicit_coupling_new(
#             CROSS,
#             RATES,
#             LHS,
#             target_species="TS2_32",
#             source_species="FeS_32",
#             coeff=diss_coeff_FeS,
#             rate=diss_rate_FeS * f32_FeS,
#             mp=mp,
#             has_solid=has_solid,
#             add_lhs_sink=True,
#             stoich_ratio=1.0,
#         )

#         # Verbose debug logger (disabled by default to prevent log bloat)
#         import os
#         if getattr(mp, "debug_verbose_fes", False) or os.environ.get("DEBUG_VERBOSE_FES"):
#             _debug_fes_isotopes_report(c, mp, omega, is_prec_active, is_diss_active, f32_hs, f32_FeS)


def _debug_fes_isotopes_report(c, mp, omega, is_prec_active, is_diss_active, f32_hs, f32_FeS):
    import numpy as np
    from fipyrite.diff_lib import get_delta
    
    d_TS2 = get_delta(c.TS2.value, c.TS2_32.value, mp.VCDT)
    d_FeS = get_delta(c.FeS.value, c.FeS_32.value, mp.VCDT)
    omega_np = np.asarray(omega)
    fes_val = np.asarray(c.FeS.value)
    ts2_val = np.asarray(c.TS2.value)
    
    experiment_name = getattr(mp, 'experiment', 'pyrite_model')
    log_file = f"{experiment_name}.log"
    
    max_omega = float(np.max(omega_np)) if len(omega_np) > 0 else 0.0
    max_fes = float(np.max(fes_val)) if len(fes_val) > 0 else 0.0
    valid_d_fes = d_FeS[~np.isnan(d_FeS)]
    max_d_fes = float(np.max(valid_d_fes)) if len(valid_d_fes) > 0 else np.nan
    min_d_fes = float(np.min(valid_d_fes)) if len(valid_d_fes) > 0 else np.nan

    # Identify cells where FeS is present (> 1e-8) OR Omega > 0.5
    mask = (fes_val > 1e-8) | (omega_np >= 0.5)
    idx_cells = np.where(mask)[0]
    
    if len(idx_cells) > 0:
        with open(log_file, "a") as f:
            f.write(f"  [DEBUG FeS ISOTOPES] max_Omega={max_omega:.4f}, max_FeS={max_fes:.3e}, d_FeS_range=[{min_d_fes:.1f}, {max_d_fes:.1f}]‰ ({len(idx_cells)} active cells)\n")
            for i in idx_cells[:20]:
                f.write(
                    f"    Cell {i:4d}: Omega={omega_np[i]:.4f}, "
                    f"TS2={ts2_val[i]:.3e}, d_TS2={d_TS2[i]:.2f}‰, "
                    f"FeS={fes_val[i]:.3e}, d_FeS={d_FeS[i]:.2f}‰, "
                    f"f32_hs={f32_hs[i]:.6f}, f32_FeS={f32_FeS[i]:.6f}\n"
                )


def FeS_precipitation_dissolution_symmetrical_picard(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    """
    FeS precipitation / dissolution with Symmetrical Picard Linearization.

    In FeS precipitation: Fe2⁺ + HS⁻ → FeS(s)
    Both Fe2⁺ and HS⁻ are depleted simultaneously with 1:1 stoichiometry.
    In standard Picard linearization, the implicit diagonal sink is placed
    solely on TS2, leaving Fe2⁺ with zero diagonal damping and vulnerable to
    severe negative overshoots when Fe2⁺ is the limiting reactant.

    Symmetrical Picard Linearization partitions the precipitation rate symmetrically
    (or according to a user-specified weighting parameter w_Fe) across both reactants:
        R_prec = R_prec,Fe + R_prec,TS2
               = prec_coeff_Fe2 · Fe2_total^{n+1} + prec_coeff_TS2 · TS2^{n+1}
    where:
        prec_coeff_Fe2 = w_Fe · R_prec / (Fe2_total + 1e-30)
        prec_coeff_TS2 = (1 - w_Fe) · R_prec / (TS2 + 1e-30)
        (default w_Fe = 0.5)

    Both Fe2_total and TS2 receive:
        - A positive diagonal sink on LHS proportional to 1/C (infinite damping barrier as C -> 0)
        - Symmetrical off-diagonal cross coupling to the other reactant in CROSS
    FeS receives coupled production from both Fe2_total and TS2 in CROSS.

    At convergence:
        prec_coeff_Fe2 · Fe2_total + prec_coeff_TS2 · TS2 = R_prec (exact rate, 0 RHS residual).

    Dissolution (Ω < 1) remains unimolecular in FeS(s), which is already naturally
    symmetric across its two 1:1 products (Fe2⁺ and HS⁻).
    """
    import numpy as np

    phi = mp.phi
    has_solid = True  # True for dissolution branch (solid reactant FeS)
    has_solid_prec = (
        False  # False for precipitation branch (liquid reactants Fe2_pw, HS-)
    )

    # ── Current sweep iterate ─────────────────────────────────────────────
    Fe2_val = nx.maximum(c.Fe2_total.value, 1e-20)
    TS2_val = nx.maximum(c.TS2.value, 1e-20)
    FeS_val = nx.maximum(c.FeS.value, 1e-20)

    Fe2_pw = Fe2_val * mp.Fe2_diss
    hs_val = TS2_val * mp.hs_frac
    omega_den = k.Hplus * k.FeS_sp + 1e-30
    omega = Fe2_pw * hs_val / omega_den

    # ── Per-cell branch selector ──────────────────────────────────────────
    is_prec = nx.where(omega >= 1.0, 1.0, 0.0)
    is_diss = nx.where(omega < 1.0, 1.0, 0.0)

    Km = 0.5

    # ══════════════════════════════════════════════════════════════════════
    # PRECIPITATION BRANCH (Symmetrical Picard Form: split across Fe2 & TS2)
    # ══════════════════════════════════════════════════════════════════════
    k_prec_eff = k.FeS_isp * is_prec

    df = nx.maximum(omega - 1.0, 0.0)
    mm_factor_prec = df / (Km + df)

    # Reactant weighting:
    # - "auto" (default): Patankar limiting-reactant weighting:
    #       w_Fe = TS2^p / (Fe2^p + TS2^p)
    #       w_TS2 = Fe2^p / (Fe2^p + TS2^p)
    #   When Fe2 >> TS2 (e.g. at depth): w_TS2 -> 1, w_Fe -> 0 (diagonal sink on limiting TS2,
    #   breaking two-way off-diagonal feedback and eliminating spatial rate oscillations).
    #   When TS2 >> Fe2: w_Fe -> 1, w_TS2 -> 0 (diagonal sink on limiting Fe2, preventing negative overshoots).
    #   When Fe2 ~ TS2: smooth 50/50 symmetric split.
    # - float: User-specified constant weighting (e.g. 0.5 for fixed 50/50 split).
    weight_fe_cfg = getattr(mp, "fes_picard_weight_fe", "auto")
    if isinstance(weight_fe_cfg, str) and weight_fe_cfg.lower() == "auto":
        p = float(getattr(mp, "fes_picard_auto_power", 2.0))
        fe2_p = Fe2_val ** p
        ts2_p = TS2_val ** p
        sum_p = fe2_p + ts2_p + 1e-30
        w_Fe = ts2_p / sum_p
        w_TS2 = fe2_p / sum_p
    else:
        w_Fe = float(weight_fe_cfg)
        w_TS2 = 1.0 - w_Fe

    # Total precipitation rate in porewater [mol / (m^3_pw s)]
    R_prec = k_prec_eff * mm_factor_prec

    # Picard pseudo-first-order coefficients [1/s]
    prec_coeff_Fe2 = (w_Fe * R_prec) / Fe2_val
    prec_rate_Fe2 = prec_coeff_Fe2 * Fe2_val  # w_Fe * R_prec

    prec_coeff_TS2 = (w_TS2 * R_prec) / TS2_val
    prec_rate_TS2 = prec_coeff_TS2 * TS2_val  # w_TS2 * R_prec

    # 1. FeS solid source coupled to both Fe2_total and TS2
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="FeS",
        source_species="Fe2_total",
        coeff=prec_coeff_Fe2,
        rate=prec_rate_Fe2,
        mp=mp,
        has_solid=has_solid_prec,
        add_lhs_sink=False,
        stoich_ratio=1.0,
    )
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="FeS",
        source_species="TS2",
        coeff=prec_coeff_TS2,
        rate=prec_rate_TS2,
        mp=mp,
        has_solid=has_solid_prec,
        add_lhs_sink=False,
        stoich_ratio=1.0,
    )

    # 2. Reactants diagonal sinks (LHS)
    add_implicit_sink(
        LHS,
        RATES,
        "Fe2_total",
        prec_coeff_Fe2,
        prec_rate_Fe2,
        mp=mp,
        has_solid=has_solid_prec,
    )
    add_implicit_sink(
        LHS,
        RATES,
        "TS2",
        prec_coeff_TS2,
        prec_rate_TS2,
        mp=mp,
        has_solid=has_solid_prec,
    )

    # 3. Reactants off-diagonal cross sinks (CROSS)
    # Fe2_total consumed by TS2 portion of the reaction:
    CROSS["Fe2_total"].append(("TS2", -prec_coeff_TS2 * phi))
    RATES["Fe2_total"] -= prec_rate_TS2 * phi

    # TS2 consumed by Fe2 portion of the reaction:
    CROSS["TS2"].append(("Fe2_total", -prec_coeff_Fe2 * phi))
    RATES["TS2"] -= prec_rate_Fe2 * phi

    # ══════════════════════════════════════════════════════════════════════
    # DISSOLUTION BRANCH (Picard Form: unimolecular in FeS, exact 1:1:1 stoichiometry)
    # ══════════════════════════════════════════════════════════════════════
    k_diss_eff = k.FeS_isd * is_diss

    us = nx.maximum(1.0 - omega, 0.0)
    mm_factor_diss = us / (Km + us)

    diss_coeff_FeS = k_diss_eff * mm_factor_diss
    diss_rate_FeS = diss_coeff_FeS * FeS_val

    # (a) FeS diagonal sink & Fe2_total source
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="Fe2_total",
        source_species="FeS",
        coeff=diss_coeff_FeS,
        rate=diss_rate_FeS,
        mp=mp,
        has_solid=has_solid,
        add_lhs_sink=True,
        stoich_ratio=1.0,
    )

    # (b) TS2 source coupled to FeS
    add_implicit_coupling_new(
        CROSS,
        RATES,
        LHS,
        target_species="TS2",
        source_species="FeS",
        coeff=diss_coeff_FeS,
        rate=diss_rate_FeS,
        mp=mp,
        has_solid=has_solid,
        add_lhs_sink=False,
        stoich_ratio=1.0,
    )

    # ══════════════════════════════════════════════════════════════════════
    # ISOTOPES
    # ══════════════════════════════════════════════════════════════════════
    if mp.isotopes:
        # 1. Precipitation (TS2_32 -> FeS_32)
        # Symmetrical Picard operator mirroring:
        # In bulk, TS2 receives diagonal sink prec_coeff_TS2 and off-diagonal cross sink from Fe2_total (prec_coeff_Fe2).
        # To maintain identical time evolution of 32S and prevent severe artificial d34S drift (-700‰),
        # TS2_32 must mirror this exact operator structure:
        #   - Diagonal sink on TS2_32 using prec_coeff_TS2 (scaled by ratio_hs_ts2)
        #   - Off-diagonal sink on TS2_32 coupled to Fe2_total using prec_coeff_Fe2 * f32_hs
        # FeS_32 receives coupled source terms from both TS2_32 and Fe2_total.
        hs_32 = partition_equilibrium_isotope_32(
            c.TS2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
        )
        hs_val_np = np.asarray(hs_val)
        hs_32_val = np.asarray(hs_32)
        TS2_val_np = np.asarray(c.TS2.value)
        TS2_32_val = np.asarray(c.TS2_32.value)

        f32_default = 1.0 / (1.0 + mp.VCDT)
        f32_hs = np.where(hs_val_np > 1e-6, hs_32_val / (hs_val_np + 1e-30), f32_default)
        f32_hs = np.clip(f32_hs, 0.5, 1.5)

        # Bulk ratio of total sulfide for porewater sink (prevents d34S_TS2 distortion)
        f32_TS2 = np.where(TS2_val_np > 1e-6, TS2_32_val / (TS2_val_np + 1e-30), f32_default)
        f32_TS2 = np.clip(f32_TS2, 0.5, 1.5)

        ratio_hs_ts2 = np.where(f32_TS2 > 1e-10, f32_hs / f32_TS2, 1.0)
        prec_coeff_TS2_32 = prec_coeff_TS2 * ratio_hs_ts2

        # (a) FeS_32 solid source coupled to TS2_32 and Fe2_total
        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            target_species="FeS_32",
            source_species="TS2_32",
            coeff=prec_coeff_TS2_32,
            rate=prec_rate_TS2 * f32_hs,
            mp=mp,
            has_solid=has_solid_prec,
            add_lhs_sink=False,
            stoich_ratio=1.0,
        )
        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            target_species="FeS_32",
            source_species="Fe2_total",
            coeff=prec_coeff_Fe2 * f32_hs,
            rate=prec_rate_Fe2 * f32_hs,
            mp=mp,
            has_solid=has_solid_prec,
            add_lhs_sink=False,
            stoich_ratio=1.0,
        )

        # (b) TS2_32 porewater diagonal sink
        add_implicit_sink(
            LHS,
            RATES,
            "TS2_32",
            prec_coeff_TS2_32,
            prec_rate_TS2 * f32_hs,
            mp=mp,
            has_solid=has_solid_prec,
        )

        # (c) TS2_32 off-diagonal cross sink coupled to Fe2_total
        CROSS["TS2_32"].append(("Fe2_total", -prec_coeff_Fe2 * f32_hs * phi))
        RATES["TS2_32"] -= prec_rate_Fe2 * f32_hs * phi

        # 2. Dissolution (FeS_32 -> TS2_32)
        FeS_val_np = np.asarray(c.FeS.value)
        FeS_32_val = np.asarray(c.FeS_32.value)
        f32_FeS = np.where(FeS_val_np > 1e-3, FeS_32_val / (FeS_val_np + 1e-30), f32_hs)
        f32_FeS = np.clip(f32_FeS, 0.5, 1.5)

        add_implicit_coupling_new(
            CROSS,
            RATES,
            LHS,
            target_species="TS2_32",
            source_species="FeS_32",
            coeff=diss_coeff_FeS,
            rate=diss_rate_FeS * f32_FeS,
            mp=mp,
            has_solid=has_solid,
            add_lhs_sink=True,
            stoich_ratio=1.0,
        )

        # Verbose debug logger (disabled by default to prevent log bloat)
        import os
        if getattr(mp, "debug_verbose_fes", False) or os.environ.get("DEBUG_VERBOSE_FES"):
            _debug_fes_isotopes_report(c, mp, omega, mm_factor_prec > 0, mm_factor_diss > 0, f32_hs, f32_FeS)


FeS_precipitation_dissolution_symmetrical = FeS_precipitation_dissolution_symmetrical_picard


# def sulfide_mediated_iron_reduction_old(c, k, lim, LHS, RHS, RATES, CROSS, mp):
#     """Fe3 iron reduction via HS-.

#     0.5 HS- + Fe3+ -> 0.5 S0 + Fe2+

#     The reaction is doubly-capped so that at most 70% of either Fe3 or TS2
#     is consumed in a single timestep, guaranteeing perfect stoichiometry and
#     unconditional positivity. Coupling uses a single master implicit variable
#     (Fe3) to ensure exact mass balance across all species.
#     """
#     has_solid = True  # True if the reactants contain a solid phase species
#     # ------------------------------------------------------------------
#     # 1. Current state (using FiPy variables directly for consistency)
#     # ------------------------------------------------------------------
#     TS2_val = nx.maximum(c.TS2, 0.0)
#     hs_val = TS2_val * mp.hs_frac
#     Fe3_val = nx.maximum(c.Fe3, 0.0)

#     # ------------------------------------------------------------------
#     # 2. Rate Calculation and Multi-Reactant Capping
#     # ------------------------------------------------------------------
#     k_base_val = k.Fe3_hs * lim["O2_inhibit"] * lim["Fe3_implicit"]

#     # Uncapped reaction rate driven by Fe3 consumption [mmol/L_solid/s]
#     rate_uncapped = k_base_val * hs_val * Fe3_val

#     # Total available TS2 reservoir over the next timestep (including chemical production)
#     # RATES contains rates in bulk units (mmol/L_bulk/s), so we divide by porosity to get mmol/L_liquid/s
#     TS2_prod_pw = RATES["TS2"] / (mp.phi + 1e-30)
#     dt_val = getattr(mp, "current_dt", 0.0)
#     dt_safe = nx.maximum(dt_val, 1e-12)
#     TS2_available = nx.maximum(TS2_val + TS2_prod_pw * dt_safe, 0.0)

#     # Limit by Fe3 depletion (at most 70% per timestep)
#     max_rate_Fe3 = 0.7 * Fe3_val / dt_safe

#     # Limit by TS2 depletion (0.5 mole TS2 consumed per 1 mole Fe3)
#     # 0.5 * Rate * dt <= 0.7 * TS2_available -> Rate <= 1.4 * TS2_available / dt
#     max_rate_TS2 = 1.4 * TS2_available / dt_safe

#     # Actual capped rate
#     rate_actual = nx.minimum(rate_uncapped, nx.minimum(max_rate_Fe3, max_rate_TS2))
#     # rate_actual = rate_uncapped

#     # Single master coefficient based on Fe3 [1/s]
#     coeff_master = rate_actual / (Fe3_val + 1e-30)

#     # ------------------------------------------------------------------
#     # 3. Fe3 sink (Master Variable) — EXACTLY 1:1
#     # ------------------------------------------------------------------
#     add_implicit_sink(
#         LHS,
#         RATES,
#         "Fe3",
#         coeff_master,
#         rate_actual,
#         mp=mp,
#         has_solid=has_solid,
#         c=c,
#         reaction="sulfide_mediated_iron_reduction",
#     )

#     # ------------------------------------------------------------------
#     # 4. Fe2 source (Coupled to Fe3_new) — EXACTLY 1:1
#     # ------------------------------------------------------------------
#     add_implicit_coupling_new(
#         CROSS,
#         RATES,
#         LHS,
#         target_species="Fe2_total",
#         source_species="Fe3",
#         coeff=coeff_master,
#         rate=rate_actual,
#         mp=mp,
#         has_solid=has_solid,
#         c=c,
#         add_lhs_sink=False,
#         stoich_ratio=1.0,
#         reaction="sulfide_mediated_iron_reduction",
#     )

#     # ------------------------------------------------------------------
#     # 5. TS2 sink (Coupled to Fe3_new) — EXACTLY 0.5:1
#     # ------------------------------------------------------------------
#     add_implicit_coupling_new(
#         CROSS,
#         RATES,
#         LHS,
#         target_species="TS2",
#         source_species="Fe3",
#         coeff=coeff_master,
#         rate=rate_actual,
#         mp=mp,
#         has_solid=has_solid,
#         c=c,
#         add_lhs_sink=False,
#         stoich_ratio=-0.5,
#         reaction="sulfide_mediated_iron_reduction",
#     )

#     # ------------------------------------------------------------------
#     # 6. S0 source (Coupled to Fe3_new) — EXACTLY 0.5:1
#     # ------------------------------------------------------------------
#     add_implicit_coupling_new(
#         CROSS,
#         RATES,
#         LHS,
#         target_species="S0",
#         source_species="Fe3",
#         coeff=coeff_master * 0.5,
#         rate=0.5 * rate_actual,
#         mp=mp,
#         has_solid=has_solid,
#         c=c,
#         add_lhs_sink=False,
#         stoich_ratio=1.0,
#         reaction="sulfide_mediated_iron_reduction",
#     )

#     # ------------------------------------------------------------------
#     # 7. Isotopes (32S)
#     # ------------------------------------------------------------------
#     if mp.isotopes:
#         hs_32 = partition_equilibrium_isotope_32(
#             c.TS2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
#         )
#         f32 = hs_32 / (TS2_val * mp.hs_frac + 1e-30)

#         rate_32 = 0.5 * rate_actual * f32
#         coeff_32 = 0.5 * coeff_master * f32

#         # TS2_32 sink
#         add_implicit_coupling_new(
#             CROSS,
#             RATES,
#             LHS,
#             target_species="TS2_32",
#             source_species="Fe3",
#             coeff=coeff_32,
#             rate=rate_32,
#             mp=mp,
#             has_solid=has_solid,
#             c=c,
#             add_lhs_sink=False,
#             stoich_ratio=-1.0,
#         )

#         # S0_32 source
#         add_implicit_coupling_new(
#             CROSS,
#             RATES,
#             LHS,
#             target_species="S0_32",
#             source_species="Fe3",
#             coeff=coeff_32,
#             rate=rate_32,
#             mp=mp,
#             has_solid=has_solid,
#             c=c,
#             add_lhs_sink=False,
#             stoich_ratio=1.0,
#         )




