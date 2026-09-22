"""Define reaction constants.

Note, Velde et al likely have a typo in the FeS equilibrium constant.
Rickard (2006) defines two reaction constants, one with Ksp = 3.5
and the other with Ksp = - 3.5. Here we use the latter since otherwise
we would never precipitate FeS unless H2S > 1 mmol/l
"""

from fipy import CellVariable


def get_reaction_constants(
    pH: float, phi: float | CellVariable, k_values: dict = {}
) -> dict:
    """Convert reaction_constants into model units.

        It is assumed that the reaction constants are in phase specific units.
        This can be tricky, as Velde et al, state that their k-values are in
        bulk units, but then they convert it before using (see Table 3). E.g.
        they write:

       (1−ϕ)⋅kSIR⋅[FeOOH]⋅[HS−]
    ​
        which suggests that the (1−ϕ) is used to convert FeOOH from solid
        phase concentration to bulk concentration. Since this code
        tracks all concentrations in phase specific units, converting the
        k-value by deviding by (1−ϕ), would cancel out, and we can use the
        k-value as published.
    """
    import pint

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    if isinstance(phi, CellVariable):
        phi = phi.value

    velde: dict = {
        "POC_fast": [10, "1/year", "1/second"],  # highly reactive OM
        "POC_slow": [0.1, "1/year", "1/second"],  # reactive OM, Velde et al
        "Fe2_p_eq": [
            696,
            "dimensionless",
            "dimensionless",
        ],  # sorbed vs Fe2+ Fe2+ liq.
        "FeS2_O2": [
            1e-10,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
        ],  # FeS2 + O2 -> SO4, Halevy et al
        "FeS_S0": [  # FeS + S0 -> FeS2
            5e-8,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
        ],  # FeS + HS -> FeS2
        "FeS_TS2": [
            5e-8,
            "m^3/(mol*second)",
            "m^3/(mol*second)",
        ],  # FeS + H2S -> FeS2, at 10C -> notes.org
        "Hplus": [10 ** (-pH), "mol/l", "mol/m^3"],
        # Oxidation reactions after Velde at eal 2016
        "TS2_O2": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        "Fe3_hs": [
            4,
            "cm^3/(umol*year)",
            "m^3/(mol*second)",
        ],  # check with halevy
        "FeS_O2": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        "Fe2_O2": [1e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        # "Fe2p_ox": [1e73, "cm^3/(umol*year)", "m^3/(mol*second)"],
        "S0_O2": [4e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
        # Iron sulfide reactions after Velde at eal 2016
        "FeS_isp": [
            1e-2 * 100, # After Velde.
            "umol/((cm^3*year))",
            "mol/(m^3 *second)",
        ],  # FeS precipitation
        "FeS_isd": [3/10, "1/year", "1/second"],  # FeS dissolution
        "FeS_sp": [  # FeS equilibrium constant
            10**-3.5,
            "mol/l",
            "mol/m^3 ",
        ],  # Dispro rate, 4 times that of HS oxidation
        "S0_dispro": [4e7, "cm^3/(umol*year)", "m^3/(mol*second)"],
    }

    for k, v in velde.items():
        k_values[k] = Q_(f"{v[0]} {v[1]}").to(f"{v[2]}").magnitude

    return velde, k_values


if __name__ == "__main__":
    import pint

    ureg = pint.UnitRegistry()
    Q_ = ureg.Quantity

    velde, k_values = get_reaction_constants(7.5, 0.8)
    ureg = pint.UnitRegistry()
    for k, v in velde.items():
        n = Q_(f"{v[0]} {v[1]}").to(f"{v[2]}")
        print(f"{k} = {n.magnitude:.2e} [{n.units:~P}]")

    print(f"k.FeS_sp * k.Hplus: {k_values['FeS_sp'] * k_values['Hplus']:.2e}")
    print(f"k.Hplus: {k_values['Hplus']:.2e}")
