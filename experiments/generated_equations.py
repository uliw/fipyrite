# Auto-generated reaction functions

from fipyrite.diff_lib import add_coupled_reaction, add_implicit_sink, add_monod_sink, calculate_fractionated_coeff_32, partition_equilibrium_isotope_32

def aerobic_respiration(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = True
    poc_species = k.get('poc_species', 'POC_fast')
    poc_k_name = k.get('poc_k', 'POC_fast')
    k_val = mp.k.get(poc_k_name) if hasattr(mp, 'k') else k.get(poc_k_name, 0.0)
    rate_base = k_val * c.O2 * getattr(c, poc_species) * lim['O2_implicit']
    R_max_O2 = ((1) / 1.0) * k_val * 1.0 * getattr(c, poc_species)
    add_monod_sink(
        LHS=LHS, RHS=RHS, RATES=RATES,
        species='O2',
        conc=c.O2,
        K_m=mp.K_O2 / mp.phi,
        R_max=R_max_O2,
        mp=mp,
        has_solid=has_solid,
        scheme=getattr(mp, 'monod_scheme', 'hybrid'),
        c=c,
    )
    coeff_POC = k_val * c.O2 * 1.0 * lim['O2_implicit']
    add_implicit_sink(LHS, RATES, poc_species, coeff_POC, rate_base, mp=mp, has_solid=has_solid, c=c)


def dissimilatory_iron_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = True
    poc_species = k.get('poc_species', 'POC_fast')
    poc_k_name = k.get('poc_k', 'POC_fast')
    k_val = mp.k.get(poc_k_name) if hasattr(mp, 'k') else k.get(poc_k_name, 0.0)
    rate_base = k_val * c.Fe3 * getattr(c, poc_species) * lim['O2_inhibit'] * lim['Fe3_diss_red_implicit']
    rate_master = rate_base
    coeff_master = k_val * 1.0 * getattr(c, poc_species) * lim['O2_inhibit'] * lim['Fe3_diss_red_implicit']
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species={'Fe3': 4},
        reactants={poc_species: 1},
        products={'Fe2_total': 4},
        coeff_master=coeff_master,
        rate_master=rate_master,
        has_solid=has_solid,
        reaction_name='dissimilatory_iron_reduction',
        ref_species='POC',
        stoich_ref=1.0,
    )
    coeff_POC = k_val * c.Fe3 * 1.0 * lim['O2_inhibit'] * lim['Fe3_diss_red_implicit']
    add_implicit_sink(LHS, RATES, poc_species, coeff_POC, rate_base, mp=mp, has_solid=has_solid, c=c)


def sulfate_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = True
    poc_species = k.get('poc_species', 'POC_fast')
    poc_k_name = k.get('poc_k', 'POC_fast')
    k_val = mp.k.get(poc_k_name) if hasattr(mp, 'k') else k.get(poc_k_name, 0.0)
    rate_base = k_val * getattr(c, poc_species) * c.SO4 * lim['O2_inhibit'] * lim['SO4_implicit'] * lim['Fe3_diss_red_inhib']
    rate_master = rate_base
    coeff_master = k_val * getattr(c, poc_species) * 1.0 * lim['O2_inhibit'] * lim['SO4_implicit'] * lim['Fe3_diss_red_inhib']
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species={'SO4': 1},
        reactants={poc_species: 2},
        products={'TS2': 1},
        coeff_master=coeff_master,
        rate_master=rate_master,
        has_solid=has_solid,
        reaction_name='sulfate_reduction',
        ref_species='POC',
        stoich_ref=2.0,
    )
    coeff_POC = k_val * 1.0 * c.SO4 * lim['O2_inhibit'] * lim['SO4_implicit'] * lim['Fe3_diss_red_inhib']
    add_implicit_sink(LHS, RATES, poc_species, coeff_POC, rate_base, mp=mp, has_solid=has_solid, c=c)
    if mp.isotopes:
        alpha = 1.0 + (mp.msr_alpha - 1.0) * lim['SO4_alpha_explicit']
        coeff_master_32 = calculate_fractionated_coeff_32(
            coeff_master, c.SO4, c.SO4_32, alpha, eps=1e-30
        )
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species={'SO4_32': 1},
            reactants={},
            products={'TS2_32': 1},
            coeff_master=coeff_master_32,
            rate_master=coeff_master_32 * c.SO4_32,
            has_solid=has_solid,
            reaction_name='sulfate_reduction_32',
            ref_species='POC',
            stoich_ref=2.0,
        )


def hs_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = False
    HS = c.TS2 * mp.hs_frac
    k_val = mp.k.get('TS2_O2') if hasattr(mp, 'k') else k.get('TS2_O2', 0.0)
    rate_base = k_val * HS * c.O2
    rate_master = rate_base
    coeff_master = k_val * 1.0 * mp.hs_frac * c.O2
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species={'TS2': 2},
        reactants={},
        products={'S0': 2},
        coeff_master=coeff_master,
        rate_master=rate_master,
        has_solid=has_solid,
        reaction_name='hs_oxidation',
        ref_species='TS2',
        stoich_ref=2.0,
    )
    coeff_O2 = ((1) / 2.0) * k_val * HS * 1.0
    add_implicit_sink(LHS, RATES, 'O2', coeff_O2, ((1) / 2.0) * rate_base, mp=mp, has_solid=has_solid, c=c)
    if mp.isotopes:
        alpha = 1.0 + (mp.TS2_O2_alpha - 1.0) * lim['TS2_alpha_explicit']
        hs_32 = partition_equilibrium_isotope_32(
            c.TS2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
        )
        coeff_master_32 = calculate_fractionated_coeff_32(
            coeff_master, c.TS2 * mp.hs_frac, hs_32, alpha, eps=1e-30
        )
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species={'TS2_32': 2},
            reactants={},
            products={'S0_32': 2},
            coeff_master=coeff_master_32,
            rate_master=coeff_master_32 * c.TS2_32,
            has_solid=has_solid,
            reaction_name='hs_oxidation_32',
            ref_species='TS2',
            stoich_ref=2.0,
        )


def hs_oxidation_velde(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = False
    HS = c.TS2 * mp.hs_frac
    k_val = mp.k.get('TS2_O2') if hasattr(mp, 'k') else k.get('TS2_O2', 0.0)
    rate_base = k_val * HS * c.O2 * lim['O2_implicit_TS2']
    rate_master = rate_base
    coeff_master = k_val * 1.0 * mp.hs_frac * c.O2 * lim['O2_implicit_TS2']
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species={'TS2': 1},
        reactants={},
        products={'SO4': 1},
        coeff_master=coeff_master,
        rate_master=rate_master,
        has_solid=has_solid,
        reaction_name='hs_oxidation_velde',
        ref_species='TS2',
        stoich_ref=1.0,
    )
    R_max_O2 = ((2) / 1.0) * k_val * HS * 1.0
    add_monod_sink(
        LHS=LHS, RHS=RHS, RATES=RATES,
        species='O2',
        conc=c.O2,
        K_m=mp.K_O2_TS2,
        R_max=R_max_O2,
        mp=mp,
        has_solid=has_solid,
        scheme=getattr(mp, 'monod_scheme', 'hybrid'),
        c=c,
    )
    if mp.isotopes:
        alpha = 1.0 + (mp.TS2_O2_alpha - 1.0) * lim['TS2_alpha_explicit']
        hs_32 = partition_equilibrium_isotope_32(
            c.TS2_32, mp.hs_frac, mp.h2s_frac, mp.h2s_hs_alpha
        )
        coeff_master_32 = calculate_fractionated_coeff_32(
            coeff_master, c.TS2 * mp.hs_frac, hs_32, alpha, eps=1e-30
        )
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species={'TS2_32': 1},
            reactants={},
            products={'SO4_32': 1},
            coeff_master=coeff_master_32,
            rate_master=coeff_master_32 * c.TS2_32,
            has_solid=has_solid,
            reaction_name='hs_oxidation_velde_32',
            ref_species='TS2',
            stoich_ref=1.0,
        )


def elemental_sulfur_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = True
    k_val = mp.k.get('S0_O2') if hasattr(mp, 'k') else k.get('S0_O2', 0.0)
    rate_base = k_val * c.O2 * c.S0
    rate_master = rate_base
    coeff_master = k_val * c.O2 * 1.0
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species={'S0': 2},
        reactants={},
        products={'SO4': 2},
        coeff_master=coeff_master,
        rate_master=rate_master,
        has_solid=has_solid,
        reaction_name='elemental_sulfur_oxidation',
        ref_species='S0',
        stoich_ref=2.0,
    )
    coeff_O2 = ((3) / 2.0) * k_val * 1.0 * c.S0
    add_implicit_sink(LHS, RATES, 'O2', coeff_O2, ((3) / 2.0) * rate_base, mp=mp, has_solid=has_solid, c=c)
    if mp.isotopes:
        coeff_master_32 = coeff_master
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species={'S0_32': 2},
            reactants={},
            products={'SO4_32': 2},
            coeff_master=coeff_master_32,
            rate_master=coeff_master_32 * c.S0_32,
            has_solid=has_solid,
            reaction_name='elemental_sulfur_oxidation_32',
            ref_species='S0',
            stoich_ref=2.0,
        )


def sulfide_mediated_iron_reduction(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = True
    HS = c.TS2 * mp.hs_frac
    k_val = mp.k.get('Fe3_hs') if hasattr(mp, 'k') else k.get('Fe3_hs', 0.0)
    rate_base = k_val * c.Fe3 * HS * lim['O2_inhibit'] * lim['Fe3_implicit']
    rate_master = rate_base
    coeff_master = k_val * c.Fe3 * 1.0 * mp.hs_frac * lim['O2_inhibit'] * lim['Fe3_implicit']
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species={'TS2': 1},
        reactants={'Fe3': 2},
        products={'Fe2_total': 2, 'S0': 1},
        coeff_master=coeff_master,
        rate_master=rate_master,
        has_solid=has_solid,
        reaction_name='sulfide_mediated_iron_reduction',
        ref_species='Fe3',
        stoich_ref=2.0,
    )
    if mp.isotopes:
        coeff_master_32 = coeff_master
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species={'TS2_32': 1},
            reactants={},
            products={'S0_32': 1},
            coeff_master=coeff_master_32,
            rate_master=coeff_master_32 * c.TS2_32,
            has_solid=has_solid,
            reaction_name='sulfide_mediated_iron_reduction_32',
            ref_species='Fe3',
            stoich_ref=2.0,
        )


def sulfide_mediated_iron_reduction_velde(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = True
    HS = c.TS2 * mp.hs_frac
    k_val = mp.k.get('Fe3_hs') if hasattr(mp, 'k') else k.get('Fe3_hs', 0.0)
    rate_base = k_val * c.Fe3 * HS * lim['O2_inhibit'] * lim['Fe3_implicit']
    rate_master = rate_base
    coeff_master = k_val * c.Fe3 * 1.0 * mp.hs_frac * lim['O2_inhibit'] * lim['Fe3_implicit']
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species={'TS2': 1},
        reactants={'Fe3': 8},
        products={'Fe2_total': 8, 'SO4': 1},
        coeff_master=coeff_master,
        rate_master=rate_master,
        has_solid=has_solid,
        reaction_name='sulfide_mediated_iron_reduction_velde',
        ref_species='Fe3',
        stoich_ref=8.0,
    )
    if mp.isotopes:
        coeff_master_32 = coeff_master
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species={'TS2_32': 1},
            reactants={},
            products={'SO4_32': 1},
            coeff_master=coeff_master_32,
            rate_master=coeff_master_32 * c.TS2_32,
            has_solid=has_solid,
            reaction_name='sulfide_mediated_iron_reduction_velde_32',
            ref_species='Fe3',
            stoich_ref=8.0,
        )


def Fe2_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = False
    k_val = mp.k.get('Fe2_O2') if hasattr(mp, 'k') else k.get('Fe2_O2', 0.0)
    rate_base = k_val * c.Fe2_total * c.O2
    rate_master = rate_base
    coeff_master = k_val * 1.0 * c.O2
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species={'Fe2_total': 4},
        reactants={},
        products={'Fe3': 4},
        coeff_master=coeff_master,
        rate_master=rate_master,
        has_solid=has_solid,
        reaction_name='Fe2_oxidation',
        ref_species='Fe2_total',
        stoich_ref=4.0,
    )
    coeff_O2 = ((1) / 4.0) * k_val * c.Fe2_total * 1.0
    add_implicit_sink(LHS, RATES, 'O2', coeff_O2, ((1) / 4.0) * rate_base, mp=mp, has_solid=has_solid, c=c)


def FeS_oxidation(c, k, lim, LHS, RHS, RATES, CROSS, mp):
    has_solid = True
    k_val = mp.k.get('FeS_O2') if hasattr(mp, 'k') else k.get('FeS_O2', 0.0)
    rate_base = k_val * c.FeS * c.O2
    rate_master = rate_base
    coeff_master = k_val * 1.0 * c.O2
    add_coupled_reaction(
        CROSS=CROSS,
        LHS=LHS,
        RATES=RATES,
        mp=mp,
        master_species={'FeS': 4},
        reactants={},
        products={'Fe3': 4, 'SO4': 4},
        coeff_master=coeff_master,
        rate_master=rate_master,
        has_solid=has_solid,
        reaction_name='FeS_oxidation',
        ref_species='FeS',
        stoich_ref=4.0,
    )
    coeff_O2 = ((9) / 4.0) * k_val * c.FeS * 1.0
    add_implicit_sink(LHS, RATES, 'O2', coeff_O2, ((9) / 4.0) * rate_base, mp=mp, has_solid=has_solid, c=c)
    if mp.isotopes:
        coeff_master_32 = coeff_master
        add_coupled_reaction(
            CROSS=CROSS,
            LHS=LHS,
            RATES=RATES,
            mp=mp,
            master_species={'FeS_32': 4},
            reactants={},
            products={'SO4_32': 4},
            coeff_master=coeff_master_32,
            rate_master=coeff_master_32 * c.FeS_32,
            has_solid=has_solid,
            reaction_name='FeS_oxidation_32',
            ref_species='FeS',
            stoich_ref=4.0,
        )


