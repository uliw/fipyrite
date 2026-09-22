"""
Define a reaction-transport model that computes pyrite precipitation.

as a function of organic matter availability, including isotopes.  Model units are
meter/second, concentrations are given mmol/liter (mol/m^3) and solids are expressed as
concentration per unit of solid volume (mmol/L_solid).

This keeps the physics of the "solid phase" independent of how much water is currently
squeezing around it.  If the sediment compacts (porosity ϕ decreases), the amount of
organic matter per gram of rock doesn't change, but the amount of organic matter per
liter of bulk sediment does.  ​

As such, a reaction between a liquid and a solid needs to be scaled

f = k * [SO4] * (1 - phi)/phi * [OM]

"""


def pyrite_model(p_dict: dict, plot_queue=None, experiment="pyrite"):
    """Model pyrite precipitation.

    As a function of organic matter availability, including isotopes
    Model units are meter/second, mmol/liter, and meter
    """
    # import numpy as np
    import pandas as pd
    import pint

    # from reactions_new import diagenetic_reactions
    import reactions_new as rn
    from fipy import CellVariable
    from fipy.tools import numerix as np

    import fipyrite.plot_data_new as plot_data_new
    from fipyrite.diff_lib import (
        check_peclet_numbers,
        compute_bio_irrigation_alpha,
        compute_sigmoidal_db,
        data_container,
        diff_coeff,
        get_l_mass,
        make_grid2,
        read_state,
    )
    from fipyrite.live_plot_lib import LivePlotter, capture_state
    from fipyrite.solver_calls import (
        run_non_steady_state_solver_coupled,
    )

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    mp = data_container({
        # -------- File Names & output --------------------
        "plot_name": f"{experiment}",
        "state_data": None,  # f"{experiment}_state.npz",
        "layout_file": "plot_layout.py",
        # "process_monitor": "none",  # gui | video | none
        # "process_monitor": "gui",  # gui | video | none
        "process_monitor": "video",  # gui | video | none
        "report_step": 10,  # how often to update plot
        "backup_step": 1000,  # create backups every nth step
        "title": None,  # defaults to current time
        "start_time": 0,  # i.e., when starting from a previous state
        # --------- Model Geometry --------------------------- #
        "max_depth": 2,  # meters
        "initial_spacing": 0.0001,  # meters
        "reaction_zone_spacing": 0.0001,  # meters
        "max_spacing": 0.1,  # meters, None = no cap
        "reaction_zone": (0.0, 0.1),  # in meters
        # ------ boundary conditions ------------------------ #
        "temp": [10.0, 10.1],  # temp top, bottom, in C
        "w": Q_("0.2 cm/yr").to("m/s").m,  # sedimentation rate in m/s
        "advection": 0,  # upward directed flow component
        "pH": 7.5,  # porewater pH, Velde et al.
        "phi": 0.8,
        "bc_O2": 0.28,  # mmmol/l
        "bc_SO4": 28.2,  # mmol/l
        "bc_TS2": 0.0,  # mmol/l # Total S2-
        "bc_S0": 0.0,  # mmol/l
        "bc_POC_fast": Q_("365 umol/(cm^2 * year)").to("mol/(m^2 * second)").magnitude,
        "bc_POC_slow": Q_("183 umol/(cm^2 * year)").to("mol/(m^2 * second)").magnitude,
        "bc_Fe3": Q_("12 umol/(cm^2 * year)").to("mol/(m^2 * second)").magnitude,
        "bc_Fe2": 0,  # wt% Fe2
        "bc_Fe2_p": 0,  # wt% sorbed Fe2
        "POC_O2_ratio": 1.27,  # 1.27, Velde uses 1.0
        # ---------  Monod constants -------------------------- #
        # Note, unlike the k-values in reaction_constants.py
        # These may need to be corrected to phase specific values
        # i.e., Velde et al report their k-values in bulk units
        # since phi is not yet known, we apply this correction
        # in the reactions_new.py file.
        "K_O2": Q_("0.001 umol/cm^3").to("mol/m^3").m,  # Monod constant
        "K_O2_TS2": Q_("0.001 umol/cm^3").to("mol/m^3").m,  # Monod constant
        "K_TS2": Q_("0.1 umol/cm^3").to("mol/m^3").m,  # Monod constant
        "K_SO4": Q_("0.9 umol/cm^3").to("mol/m^3").m,  # Monod constant
        "K_Fe3_diss_red": Q_("10.4 umol/cm^3")
        .to("mol/m^3")
        .m,  # Monod constant diss Fe3 reduc
        "K_Fe3": 1e-3,  # Monod constant Fe3 H2S reduc
        # -------- benthic activity ---------------------------- #
        "BT0": Q_("4 cm^2/year").to("m^2/second").magnitude * 0,
        "BT_depth": Q_("7.6 cm").to("meter").magnitude,  # Bioturbation depth in m
        "BT_attenuation": Q_("2 cm").to("meter").magnitude,  # xbm of Velde et al.
        "BI0": 1e-6 * 0,  # should be < 1e-5
        "BI_depth": 0.0,  # Irrigation depth (0 = off)
        # --------- Isotopes ----------------------------------- #
        "isotopes": True,
        "SO4_d": 21,  # seawater delta
        "S0_d": 8, 
        "msr_alpha": 1.07,  # MSR enrichment factor in mUr
        "TS2_O2_alpha": 0.995,  # sulfide oxidation enrichment factor in mUr
        "S0_O2_alpha": 1,  # sulfide oxidation enrichment factor in mUr
        "dispro_SO4_alpha": 1.02,  # about +20 mUr
        "dispro_hs_alpha": 0.993,  # about -7 mUr
        "dispro_SO4_hs_split": 0.5,  # i.e. 2 parts SO4, 1 part H2S
        "h2s_hs_alpha": 0.99991542,  # equilibrium fractionation factor between H2S and HS- for 32S (derived from alpha_34 = 1.002)
        "VCDT": 0.044162589,  # VCDT reference ratio
        "K_epsilon_msr": 0.2,  # limit MSR fractionation below 0.2 mmol/L
        "K_epsilon_TS2_O2": 0.01,  # limit HS fractionation below 0.01 mmol/L
        # --------- Solver Parameters -------------------------- #
        "max_steps": 20,  # max number of iterations
        "t_end": Q_("1 kyr").to("seconds").magnitude,
        "dt_min": Q_("1 minute").to("seconds").magnitude,  # time step in years
        "dt_init": Q_("1 month").to("seconds").magnitude,  # initial dt
        "dt_max": Q_("1 year").to("seconds").magnitude,  # time step in years
        "tolerance": 1e-12,  # convergence criterion
        "dt_tolerance": 1e-12,  # steady state threshold (stop simulation)
        # parameters controlling the dynamic time step adaption
        "dt_target_change": 100,  # target change per step (for dt adaptation)
        "solver_backend": "default",  # see solver_calls for options
        "solver_backend": "LinearGMRESSolver",  # see solver_calls for options
        #  "enable_failure_ceiling": True,
        # "failure_ceiling_factor": 0.7,           # cap dt at 70% of failed dt (e.g. 35h -> 24.5h)
        # "failure_hold_steps": 10,                # hold ceiling for 10 successful steps
        # "ceiling_growth_factor": 1.05,           # cautiously relax ceiling by 5% per step after hold
        # "enable_rate_adaptation": True,
        # "enable_rate_magnitude_check": True,
        # "rate_threshold": 1e-6,
        # "rate_sign_min_change": 1e-9,            # mol/(m^3*s) minimum rate change to consider oscillation
        # "rate_sign_min_consecutive_cells": 10,
        # # "enable_isotope_dt_limiter": True,
        # "isotope_limiter_species": "FeS",        # default
        # --------- Inner Sweeping Parameters (Method 1) ------- #
      
        # ---------  Other ------------------------------------ #
        "current_dt": 0.0,  # place holder
        "display_length": 2,  #
    })

    if "reaction_constants" in p_dict:
        get_reaction_constants = p_dict["reaction_constants"]
    else:
        from reaction_constants_slow import get_reaction_constants

    # get initial k-values.
    # Use pH and phi from p_dict if it exists, otherwise use the default from mp.
    pH = p_dict.get("pH", mp.pH)
    phi = p_dict.get("phi", mp.phi)
    k = data_container()
    _k1, k = get_reaction_constants(pH, phi, k_values=k)
    mp.k = k

    # add reactions as needed. This dict entry is a list of lists, where the first entry
    # is the function handle, and the second entry is data container with config/k values.
    # You can pass different k value containers as needed.
    mp["diagenetic_reactions"] = [
        # fast poc
        [rn.aerobic_respiration, {"poc_species": "POC_fast", "poc_k": "POC_fast"}],
        [rn.dissimilatory_iron_reduction, {"poc_species": "POC_fast", "poc_k": "POC_fast"}],
        [rn.sulfate_reduction, {"poc_species": "POC_fast", "poc_k": "POC_fast"}],
        # slow poc
        [rn.aerobic_respiration, {"poc_species": "POC_slow", "poc_k": "POC_slow"}],
        [rn.dissimilatory_iron_reduction, {"poc_species": "POC_slow", "poc_k": "POC_slow"}],
        [rn.sulfate_reduction, {"poc_species": "POC_slow", "poc_k": "POC_slow"}],
        [rn.hs_oxidation, k],
        [rn.elemental_sulfur_oxidation, k],
        [rn.sulfide_mediated_iron_reduction, k],
        # [rn.Fe2_oxidation, k],
        # [rn.FeS_precipitation_dissolution_linearized, k],
        # [rn.FeS_precipitation_terminal, k],
        # [rn.FeS_dissolution, k],
        # [rn.FeS_oxidation, k],
        # [rn.pyrite_formation_S0, k],
        # [rn.pyrite_formation_FeS_TS2, k],
        # [rn.pyrite_oxidation, k],
        # [rn.S0_disproportionation, k],
    ]

    mp["instantenous_reactions"] = [
        # [rn.Fe2_sorption_clip, 1.0],
        # [rn.sulfide_speciation_clip, 1.0],
    ]

    # update with values passed from calling program
    mp.update(p_dict)

    # -----------------------------------------------------------------------------
    # 2. MESH GENERATION (Variable Grid)
    # -----------------------------------------------------------------------------
    # mesh, z = make_grid(mp.max_depth, mp.initial_spacing, mp.max_spacing)
    mesh, z = make_grid2(
        mp.max_depth,
        mp.initial_spacing,
        mp.reaction_zone_spacing,
        mp.max_spacing,
        mp.reaction_zone,
    )
    mp.z = z
    mp.grid_points = len(z)
    mp.phi = CellVariable(name="porosity", mesh=mesh, value=mp.phi)

    # update k-values after we initialized phi as an array. This is needed for
    # models that use depth dependent phi values
    _k1, k = get_reaction_constants(mp.pH, mp.phi, k_values=k)

    # get delta values for sulfate/sulfide boundary conditions.
    mp.bc_SO4_32 = get_l_mass(mp.bc_SO4, mp.SO4_d, mp.VCDT)
    mp.bc_TS2_32 = get_l_mass(mp.bc_TS2, 0.0, mp.VCDT)  # Assume 0 delta for bc_h2s
    mp.bc_S0_32 = get_l_mass(mp.bc_S0, getattr(mp, "S0_d", 8.0), mp.VCDT)

    # -----------------------------------------------------------------------------
    # 3. VARIABLES & DIFFUSION PROFILES
    # -----------------------------------------------------------------------------
    # Species that are part of the transport system
    species_list_partial = [
        "SO4",
        "TS2",  # Total S2-
        "O2",
        "POC_fast",
        "POC_slow",
        "Fe2_total",
        "Fe3",
        "FeS",
        "S0",
        "FeS2",
    ]

    if mp.isotopes:
        species_list_partial = species_list_partial + [
            "SO4_32",
            "TS2_32",  # Total S2- 32S
            "FeS_32",
            "S0_32",
            "FeS2_32",
        ]

    # Species that we use for reporting only
    report_species = [
        "Fe2",
        "Fe2_p",
        "Hplus",
        "hs",  # HS-
        "hs_32",  # HS- 32S
        "h2s",
        "h2s_32",
    ]

    # these are not part of the T & R equation system
    species_list_full = species_list_partial + report_species

    # ---- calculate some helper coefficients ----- #
    # Note: All of these assume that porosity does not change with time!

    # Porosity correction factor
    mp.fac_s = mp.phi.value / (1.0 - mp.phi.value)

    # Fe2 sorption fraction. Since sorption is faster than transport we treat it as
    # instantenous, i.e. it is just a function of concentration K_ads = k.Fe2_p_eq which
    # is unitless (Conc_solid_vol / Conc_liquid_vol)
    # Note that Fe2_total is considered a liquid species!
    # fraction of Fe2+ in porewater mmol/L_pw
    mp.Fe2_diss = 1 / (1 + k.Fe2_p_eq)

    # 2. Fraction of Fe2+ in sediment (mmol/L_solid)
    # You must still apply the volume ratio to convert the liquid-based inventory
    # back into solid-phase concentration units for your solid-state reactions
    vol_ratio = mp.phi.value / (1.0 - mp.phi.value)
    mp.Fe2_sorb = k.Fe2_p_eq * vol_ratio * mp.Fe2_diss

    # calculate H2S/HS- speciation. Note that H in mol/l, and different
    # from k.Hplus which is in mol/m^3.
    pKa1 = 7.0
    Ka1 = 10 ** (-pKa1)
    H = 10 ** (-mp.pH)
    mp.h2s_frac = H / (H + Ka1)
    mp.hs_frac = Ka1 / (H + Ka1)
    
    # ---- Initialize CellVariables and diffusion coefficients ---- #
    D_mol = data_container()
    c = data_container()
    zeros = np.zeros(mp.grid_points)
    for species_name in species_list_full:
        setattr(D_mol, species_name, zeros)
        setattr(
            c,
            species_name,
            CellVariable(name=species_name, mesh=mesh, value=0.0, hasOld=True),
        )

    # -- Temperature & Porosity Profiles --
    T_profile = np.linspace(mp.temp[0], mp.temp[1], mp.grid_points)
    # phi_profile = np.ones(mp.grid_points) * mp.phi

    # ----- diffusion coefficients for liquid species ------ #
    D_mol.SO4 = diff_coeff(T_profile, 4.88, 0.232, mp.phi)
    D_mol.TS2 = diff_coeff(T_profile, 43.3, 0.85, mp.phi)
    D_mol.Fe2 = diff_coeff(T_profile, 27.7, 1, mp.phi)
    D_mol.O2 = (
        (0.2604 + 0.006363 * ((T_profile + 273.15) / 1))
        * 1e-9
        / (1 - np.log(mp.phi.value**2))
    )
    if mp.isotopes:
        D_mol.SO4_32 = D_mol.SO4
        D_mol.TS2_32 = D_mol.TS2

    # -- Bioturbation and Irrigation Profiles (Robust Sigmoid) --
    D_mol.D_irr = compute_bio_irrigation_alpha(z, mp.BI0, mp.BI_depth)
    D_mol.D_bio = compute_sigmoidal_db(z, mp.BT0, mp.BT_depth, mp.BT_attenuation)
    # lumped modeling of Fe2 liq and Fe2 adsorbed
    D_mol.Fe2_total = D_mol.Fe2 * mp.Fe2_diss

    # -----------------------------------------------------------------------------
    # 4. BOUNDARY CONDITIONS
    # -----------------------------------------------------------------------------
    bc_map = {
        "SO4": {"top": mp.bc_SO4, "type": "dissolved"},
        "TS2": {"top": mp.bc_TS2, "type": "dissolved"},
        "POC_fast": {"top": mp.bc_POC_fast, "type": "particulate"},
        "POC_slow": {"top": mp.bc_POC_slow, "type": "particulate"},
        "O2": {"top": mp.bc_O2, "type": "dissolved"},
        "S0": {"top": mp.bc_S0, "type": "particulate"},
        "Fe2_total": {"top": mp.bc_Fe2, "type": "dissolved"},
        "Fe3": {"top": mp.bc_Fe3, "type": "particulate"},
        "FeS": {"top": 0.0, "type": "particulate"},
        "FeS2": {"top": 0.0, "type": "particulate"},
    }

    if mp.isotopes:
        bc_map.update({
            "SO4_32": {"top": mp.bc_SO4_32, "type": "dissolved"},
            "TS2_32": {"top": mp.bc_TS2_32, "type": "dissolved"},
            "S0_32": {"top": mp.bc_S0_32, "type": "particulate"},
            "FeS_32": {"top": 0.0, "type": "particulate"},
            "FeS2_32": {"top": 0.0, "type": "particulate"},
        })

    for species_name, props in bc_map.items():
        var = getattr(c, species_name)

        if props["type"] == "particulate":
            # For particulate species, top is a flux in mol/(m²·s) bulk.
            # The solver transport term is weighted by (1-φ), so the CellVariable
            # must hold a solid-phase concentration: C_solid = J / (w * (1-φ)).
            # Robin BC: J_in = (1-φ) * (v_burial * C_solid - D * dC_solid/dx)
            # Therefore: dC_solid/dx = (v_burial * C_solid - J_in/(1-φ)) / D
            phi_top = mp.phi.value[0]  # porosity at the top face
            J_solid = props["top"] / (
                1.0 - phi_top
            )  # convert bulk flux → solid-phase flux

            D_total = getattr(D_mol, species_name, 0.0) + D_mol.D_bio
            if not isinstance(D_total, CellVariable):
                D_total = CellVariable(mesh=mesh, value=D_total)

            d_left = D_total.faceValue[mesh.facesLeft.value][0]
            if d_left > 1e-20:
                var.faceGrad.constrain(
                    [(mp.w * var.faceValue - J_solid) / D_total.faceValue],
                    mesh.facesLeft,
                )
                # if species_name == "Fe3":
                #     var.setValue(J_solid / mp.w if mp.w > 0 else 0.0)
            else:
                # Pure advection -> Dirichlet C_solid = J_solid / w
                val = J_solid / mp.w if mp.w > 0 else 0.0
                var.setValue(val)
                var.constrain(val, mesh.facesLeft)
        else:
            var.setValue(props["top"])
            var.constrain(props["top"], mesh.facesLeft)

        var.faceGrad.constrain([0.0], mesh.facesRight)

    if mp.state_data:
        print(f"Reading state from {mp.state_data}")
        read_state(c, mp.state_data)

    check_peclet_numbers(mesh, mp, D_mol, species_list_partial, bc_map)

    plotter = None
    if plot_queue is None and mp.process_monitor != "none":
        output_path = f"{mp.plot_name}.pdf"
        import os

        # For "gui", we also want video output
        video_path = None
        if mp.process_monitor in ["video", "gui"]:
            video_path = os.path.abspath(f"{mp.plot_name}.mp4")

        gui_enabled = mp.process_monitor == "gui"

        plotter = LivePlotter(
            layout_path=mp.layout_file,
            display_length=mp.display_length,
            output_path=output_path,
            video_path=video_path,
            gui=gui_enabled,
            report_step=getattr(mp, "report_step", 1),
        )
        print(f"[Parent] Starting LivePlotter (gui={gui_enabled})...", flush=True)
        plotter.start()
        print(f"[Parent] LivePlotter started. Queue: {plotter.queue}", flush=True)
        plot_queue = plotter.queue

    print(f"[Parent] Calling solver...", flush=True)
    step, max_change = run_non_steady_state_solver_coupled(
        mp,
        c,
        species_list_full,
        species_list_partial,
        k,
        rn.diagenetic_reactions,
        rn.equilibrium_reactions,
        mesh,
        D_mol,
        bc_map,
        z,
        plot_queue=plot_queue,
    )

    if plotter:
        plotter.stop()
    elif mp.process_monitor == "none":
        # Produce final plot even if monitoring was disabled
        print(f"[Parent] Producing final PDF plot: {mp.plot_name}.pdf")
        final_data = capture_state(
            mp,
            c,
            k,
            species_list_full,
            z,
            D_mol,
            rn.diagenetic_reactions,
            rn.equilibrium_reactions,
            current_dt=0.0,  # dt doesn't matter for final static plot
        )
        final_df = pd.DataFrame(final_data)
        measured_path = mp.measured_data_path if hasattr(mp, "measured_data_path") else None
        plt_desc = plot_data_new.load_layout_from_file(final_df, mp.layout_file, measured_path)
        plot_data_new.plot(
            final_df,
            mp.display_length,
            outfile=f"{mp.plot_name}.pdf",
            show=False,
            plot_description=plt_desc,
            measured_data_path=measured_path,
        )

    converged = "Yes" if step < mp.max_steps else "No"
    total_time = 0.0

    return (
        mp,
        c,
        k,
        species_list_full,
        z,
        D_mol,
        rn.diagenetic_reactions,
        converged,
        step,
        total_time,
    )
