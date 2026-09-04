#!/usr/bin/env python
"""
Define a specific modeling scenario.

Note: Model units are meter/second, concentrations are given mmol/liter (mol/m^3) and
solids are expressed as concentration per unit of solid volume (mmol/L_solid).

This keeps the physics of the "solid phase" independent of how much water is currently
squeezing around it.  If the sediment compacts (porosity ϕ decreases), the amount of
organic matter per gram of rock doesn't change, but the amount of organic matter per
liter of bulk sediment does.  ​
"""

if __name__ == "__main__":
    import faulthandler
    from pathlib import Path

    import pint
    import reactions_new as rn
    from pyrite_base_model import pyrite_model
    from reaction_constants_slow import get_reaction_constants

    from fipyrite.diff_lib import data_container, get_total_delta, save_data, save_state

    faulthandler.enable()

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    # set experiment name equal to script name
    experiment = Path(__file__).stem
    state_out = f"{experiment}_state.npz"

    p_dict = {
        "experiment": experiment,
        "state_data": "run_velde_slow_ic.npz",
        # "state_data": "run_velde_slow_picard_state.npz",
        "process_monitor": "video",  # gui | video | none
        # "process_monitor": "gui",  # gui | video | none
        "layout_file": "plot_layout.py",
        # Solver Parameters
        "enable_inner_sweeping": True,
        "enable_rate_adaptation": True,
        "enable_rate_magnitude_check": True,
        "rate_threshold": 1e-6,
        "rate_sign_min_change": 1e-9,  # mol/(m^3*s) minimum rate change to consider oscillation
        "rate_sign_min_consecutive_cells": 10,  # minimum number of consecutive cells affected
        # Failure Ceiling & Hold Parameters
        "enable_failure_ceiling": True,
        "failure_ceiling_factor": 0.7,  # cap dt at 70% of failed dt (e.g. 23.8h -> 16.67h)
        "failure_hold_steps": 10,  # hold ceiling for 10 successful steps AT the ceiling
        "ceiling_growth_factor": 1.05,  # increase ceiling by 20% after 10 successful steps at ceiling
        "max_steps": int(1e6),  # max number of iterations
        "max_depth": 0.5,  # meters
        "t_end": Q_("1000 years").to("seconds").magnitude,
        "dt_min": Q_("1 seconds").to("seconds").magnitude,  # time step in years
        "dt_init": Q_("1 minute").to("seconds").magnitude,  # initial dt
        "dt_max": Q_("1 year").to("seconds").magnitude,  # time step in years
        "dt_target_change": 1,  # target change per step (for dt adaptation)
        "report_step": 10,  # how often to update plot
        "BT0": Q_("4 cm^2/year").to("m^2/second").magnitude,
        "BT_depth": Q_("6 cm").to("meter").magnitude,  # Bioturbation depth in m
        "BT_attenuation": Q_("2 cm").to("meter").magnitude,  # xbm of Velde et al.
        "POC_O2_ratio": 1,  # Velde uses a 1:1 ratio
        "isotopes": True,
        "reaction_constants": get_reaction_constants,  # see imports to select a different one
        "solver_backend": "LinearGMRESSolver",  # see solver_calls for options
        "solver_precon": "ilu",  # Bypasses expensive Hypre BoomerAMG setup
        "debug_fes_isotopes": False,
        # "enable_isotope_dt_limiter": True,
        "isotope_limiter_species": "FeS",  # default
        "isotope_onset_threshold": 1e-5,  # default (mmol/L)
    }

    k = data_container()
    _k1, k = get_reaction_constants(7.5, 0.8, k_values=k)

    p_dict["diagenetic_reactions"] = [
        # fast poc
        [rn.aerobic_respiration, {"poc_species": "POC_fast", "poc_k": "POC_fast"}],
        [
            rn.dissimilatory_iron_reduction,
            {"poc_species": "POC_fast", "poc_k": "POC_fast"},
        ],
        [rn.sulfate_reduction, {"poc_species": "POC_fast", "poc_k": "POC_fast"}],
        # slow poc
        [rn.aerobic_respiration, {"poc_species": "POC_slow", "poc_k": "POC_slow"}],
        [
            rn.dissimilatory_iron_reduction,
            {"poc_species": "POC_slow", "poc_k": "POC_slow"},
        ],
        [rn.sulfate_reduction, {"poc_species": "POC_slow", "poc_k": "POC_slow"}],
        [rn.hs_oxidation_velde, k],
        [rn.Fe2_oxidation, k],
        [rn.sulfide_mediated_iron_reduction_velde, k],
        [rn.FeS_precipitation_dissolution_symmetrical_picard, k],
        [rn.FeS_oxidation, k],
    ]

    (
        mp,
        c,
        k,
        species_list,
        z,
        D_mol,
        diagenetic_reactions,
        converged,
        step,
        total_time,
    ) = pyrite_model(p_dict, experiment=experiment)

    # -----------------------------------------------------------------------------
    # 8. EXPORT DATA
    # -----------------------------------------------------------------------------
    df, fqfn = save_data(
        mp, c, k, species_list, z, D_mol, diagenetic_reactions, rn.equilibrium_reactions
    )

    if mp.isotopes:
        print(f"d34S = {get_total_delta(c, mp):.2f}")
        print(f"d34S = {get_total_delta(c, mp):.2f}")

    total_iron = df.c_Fe2_total + df.c_Fe3 + df.c_FeS + df.c_FeS2
    print(f"total_iron diff = {total_iron.max() - total_iron.min():.2e}")

    if state_out:
        save_state(c, state_out)
