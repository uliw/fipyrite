#!/usr/bin/env python
"""run_velde_model.py

10-meter Velde Sediment Diagenesis Simulation with Pyrite Precipitation and Sulfur Isotopes.

Solves the coupled reactive-transport system using the SymPy analytical Jacobian
and Numba Block TDMA solver (solve_block_tridiagonal_thomas), driven by the
declarative DiageneticModel interface.

Usage:
------
    python run_velde_model.py                       # Run standard simulation
    python run_velde_model.py --help                # Show CLI options
    python run_velde_model.py --video "10 year"     # Enable live video recording
    python run_velde_model.py --numerical-jacobian  # Use finite-difference Jacobian
"""

from mantis import DiageneticModel
from mantis.diff_lib import get_total_delta

if __name__ == "__main__":
    # 1. Initialize model (automatically detects species.py, reactions, etc. in same dir,
    # and derives experiment name from the script filename 'run_velde_model')
    model = DiageneticModel(
        reactions="fes_reactions.py",
        constants="kinetic_constants_slow.py",
        layout="plot_layout.py",
    )

    # 2. Configure environmental and run parameters
    model.configure(
        # video_dt = "1 year",
        max_depth="0.5 m",
        display_length="0.5 m",
        t_end="0.5 kyr",
        dt_max="1 year",
        dt_init="1 day",
        dt_min="1 second",
        dt_tolerance = 1e-5, # steady state criterium
        w="0.2 cm/yr",
        phi=0.8,
        isotopes=True,
        bioturbation={
            "BT0": "4 cm^2/year",
            "depth": "6 cm",
            "attenuation": "2 cm",
        },
        bioirrigation={
            "BI0": 0.0,  # explicitly disabled
            "depth": "0 cm",
        },
    )

    # Allow CLI argument overrides (e.g. --video, --solver-tol)
    model.parse_cli_args()

    # 3. Run simulation
    result = model.run()

    # 4. Save results & render summary plot
    df, fqfn = result.save_csv()
    result.save_state()

    if result.mp.isotopes:
        print(f"d34S = {get_total_delta(result.c, result.mp):.2f}")

    total_iron = df.c_Fe2_total + df.c_Fe3 + df.c_FeS + df.c_FeS2
    print(f"total_iron diff = {total_iron.max() - total_iron.min():.2e}")
    print(f"max FeS = {df.c_FeS.max():.4e} mmol/L")
    print(f"max TS2 = {df.c_TS2.max():.4e} mmol/L")
    print(f"max Fe2 = {df.c_Fe2_total.max():.4e} mmol/L")

    result.plot_summary()
