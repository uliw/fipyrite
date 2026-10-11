- [Introduction: What is MANTIS?](#org5dbc758)
- [Overall Model Architecture: What is Where?](#org4417d27)
- [Step 1: Defining Chemical Species (`species.py`)](#orgc16f49a)
  - [Understanding Phases: Dissolved vs. Solid vs. Mixed](#orgba057ca)
  - [Example: Standard Species Configuration](#org10d01d0)
  - [Handling Diffusion Coefficients](#orgd0812cc)
    - [Custom Diffusion Functions (e.g., Dissolved Oxygen)](#org69559bb)
- [Step 2: Defining Kinetic Constants (`kinetic_constants_*.py`)](#org809df99)
  - [Structure of `kinetic_constants_slow.py`](#org3c64ffa)
- [Step 3: Defining Reactions & Limiters (`fes_reactions.py`)](#org1fecbeb)
  - [Understanding Limiters & Inhibitors](#org739ad2b)
  - [Declaring Chemical Reactions](#org5336c1e)
- [Step 4: The Master Run Script (`run_velde_model.py`)](#org21fe2f4)
- [Step 5: Designing Plot Layouts (`plot_layout.py`)](#orgf185086)
  - [Converting Model Units to Geochemical Units](#orgaefde21)
  - [Defining Multi-Panel Profiles](#orgf5ff558)
- [Step 6: Running the Model & Producing Time-Lapse Videos](#orge7acd7b)
  - [Standard Execution](#orgee831e4)
  - [Generating a Time-Lapse Video](#orgcf4fea3)
  - [Command-Line Options Reference](#orgf872aa8)
- [Solver Mechanics & Physical Realism (Geochemist's Alert!)](#orgfeddb58)
  - [How to Ensure Physical Realism:](#org9a13232)
- [Summary & Next Steps](#org0cb5c1f)



<a id="org5dbc758"></a>

# Introduction: What is MANTIS?

Welcome to **MANTIS** (\*M\*onolithic \*A\*ssembly of \*N\*onlinear \*T\*ransport \*I\*nteractions \*S\*olver)! If you are a geochemist, sedimentologist, or marine geologist, you are likely familiar with the classic picture of early diagenesis in marine sediments:

-   Organic matter rains down onto the seafloor and is buried.
-   Microorganisms degrade this carbon using a succession of electron acceptors in order of decreasing thermodynamic yield (Oxygen $\to$ Nitrate $\to$ Manganese/Iron oxides $\to$ Sulfate $\to$ Methanogenesis).
-   Dissolved porewater species (like $\text{SO}_4^{2-}$, $\text{HS}^-$, and $\text{Fe}^{2+}$) diffuse and are bio-irrigated, while particulate phases (organic carbon, reactive iron oxides, pyrite) are buried and mixed by burrowing organisms (bioturbation).
-   Mineral phases precipitate (e.g., mackinawite $\text{FeS}$ and pyrite $\text{FeS}_2$) or dissolve, and stable isotope systems (such as $\delta^{34}\text{S}$) record these metabolic pathways in the geologic record.

Traditionally, translating these natural concepts into a computer simulation required writing intricate numerical solvers, wrestling with differential equation discretization, and manually deriving Jacobian matrices.

**MANTIS** changes this workflow. It is designed to decouple the **science** from the **numerical mechanics**:

1.  You describe the **species**, **reactions**, **rate constants**, and **environmental boundary conditions** in clean, human-readable Python files.
2.  MANTIS parses the stoichiometry, calculates exact mathematical Jacobians under the hood using symbolic computer algebra (SymPy), and solves the coupled reactive-transport system using an accelerated monolithic Block Newton solver.
3.  You get publication-ready depth profiles, spreadsheets, and time-lapse videos of your sediment column.

This guide walks you through setting up, configuring, running, and interpreting your own diagenetic simulations.


<a id="org4417d27"></a>

# Overall Model Architecture: What is Where?

A typical MANTIS project directory contains five primary configuration files, alongside the master run script:

```
my_project/
├── species.py                 <- Who lives in the sediment? (chemical species, phases, diffusion, BCs)
├── fes_reactions.py           <- What reactions happen? (stoichiometry, kinetics, limiters)
├── kinetic_constants_slow.py  <- How fast do reactions happen? (rate constants, half-saturation values)
├── plot_layout.py             <- How should the profiles look? (panels, colors, geochemist units like wt%)
└── run_velde_model.py         <- Master driver script (sediment accumulation rate, depth, duration)
```

When you execute `python run_velde_model.py`, MANTIS automatically detects these files, ensures the underlying equation system is up to date, solves the time evolution of the sediment column, and produces your outputs:

-   `<model_name>.pdf`: A multi-panel summary plot of porewater and solid-phase profiles.
-   `<model_name>.csv`: A tabular export of concentrations, reaction rates, and isotope ratios across depth.
-   `<model_name>_state.npz`: The complete numerical state of the simulation for restarting or continuing runs.
-   `<model_name>.mp4` (optional): An animated video showing the transient evolution over hundreds or thousands of years.


<a id="orgc16f49a"></a>

# Step 1: Defining Chemical Species (`species.py`)

The `species.py` file is your geochemical inventory. It declares every chemical entity that exists in your model domain.


<a id="orgba057ca"></a>

## Understanding Phases: Dissolved vs. Solid vs. Mixed

In a sediment column, species exist in different physical compartments:

1.  **Liquid (Porewater)**: Solutes dissolved in the pore fluid (e.g., $\text{SO}_4^{2-}$, $\text{O}_2$, $\text{HS}^-$). Their concentrations are expressed per unit volume of porewater ($\text{mmol/L}_{\text{pw}}$) and their inventory is scaled by the sediment porosity ($\phi$). They move by molecular diffusion, porewater advection, and bio-irrigation.
2.  **Solid (Particulate)**: Minerals and particulate organic carbon bound in the sediment framework (e.g., $\text{FeS}_2$, reactive iron $\text{Fe}^{3+}$, organic carbon $\text{POC}$). Their inventory is scaled by the solid fraction ($1 - \phi$). They move solely via sediment accumulation (burial) and macrofaunal bioturbation.
3.  **Mixed Pools**: Lumped entities like total ferrous iron ($\text{Fe}^{2+}_{\text{total}}$) that partition rapidly into a dissolved fraction and a surface-adsorbed fraction.


<a id="org10d01d0"></a>

## Example: Standard Species Configuration

Here is how common species are declared inside the `species` dictionary in `species.py`:

```python
species = {
    # -------------------------------------------------------------------------
    # Dissolved Porewater Solutes
    # -------------------------------------------------------------------------
    "SO4": {
        "phase": "liquid",
        "include": True,               # True = solve transport PDE; False = diagnostic/fixed
        "formula": "SO4",
        "diff_coeff_function": "diff_coeff",
        "diffusion": {"m0": 4.88, "m1": 0.232},  # Boudreau (1996) diffusion parameters
        "bc": {"top": 28.2, "type": "dissolved"}, # 28.2 mmol/L seawater concentration at SWI
        "isotope": "SO4_32",           # Companion light-sulfur isotope species
    },

    "SO4_32": {
        "phase": "liquid",
        "include": True,
        "formula": "SO4_32",
        "diff_coeff_function": "diff_coeff",
        "diffusion": "SO4",            # Inherits the molecular diffusion of parent SO4
        "bc": {"top": 27.007, "type": "dissolved"},
        "isotope_of": "SO4",
    },

    # -------------------------------------------------------------------------
    # Solid Minerals and Particulate Matter
    # -------------------------------------------------------------------------
    "Fe3": {
        "phase": "solid",
        "include": True,
        "formula": "Fe3",
        "diffusion": None,             # Solids do not have molecular diffusion!
        "bc": {"top": 120.0, "type": "particulate"}, # Solid input at sediment-water interface
    },

    "FeS2": {
        "phase": "solid",
        "include": True,
        "formula": "FeS2",
        "diffusion": None,
        "bc": {"top": 0.0, "type": "particulate"},   # Formed authigenically in sediment
        "isotope": "FeS2_32",
    },

    # -------------------------------------------------------------------------
    # System Master Variables (e.g., pH)
    # -------------------------------------------------------------------------
    "Hplus": {
        "phase": "liquid",
        "include": False,              # Fixed background parameter, not an active transport PDE
        "formula": "H",
        "bc": {"pH": 7.5},             # Set porewater pH as the single source of truth!
    },
}
```


<a id="orgd0812cc"></a>

## Handling Diffusion Coefficients

For dissolved ions, molecular diffusion coefficients in sediment are calculated using the empirical linear temperature formulation from **Boudreau (1996)** corrected for tortuosity via Archie's law: $$ D_0(T) = (m_0 + m_1 \cdot T) \times 10^{-10} \quad [\text{m}^2/\text{s}] $$ $$ D_{\text{sed}} = \frac{D_0(T)}{1 - \ln(\phi^2)} $$

For ions following this relationship, you simply set:

```python
"diff_coeff_function": "diff_coeff",
"diffusion": {"m0": 4.88, "m1": 0.232},
```


<a id="org69559bb"></a>

### Custom Diffusion Functions (e.g., Dissolved Oxygen)

Some species, such as dissolved $\text{O}_2$, require a non-standard formulation. You can declare your own custom diffusion function directly at the top of `species.py` and point to it by name:

```python
def boudreau_o2(T_profile, phi):
    """Temperature and Archie porosity corrected O2 molecular diffusion."""
    phi_val = phi.value if hasattr(phi, "value") else phi
    return (
        (0.2604 + 0.006363 * (T_profile + 273.15))
        * 1e-9
        / (1.0 - np.log(phi_val**2))
    )

species = {
    "O2": {
        "phase": "liquid",
        "include": True,
        "formula": "O2",
        "diff_coeff_function": "boudreau_o2",  # Calls the custom function defined above!
        "diffusion": None,
        "bc": {"top": 0.28, "type": "dissolved"}, # 280 uM typical bottom water O2
    },
    # ...
}
```


<a id="org809df99"></a>

# Step 2: Defining Kinetic Constants (`kinetic_constants_*.py`)

The kinetic constants file defines how fast reactions occur and the affinity thresholds of microbial communities.

In diagenesis, we typically distinguish:

1.  **Rate Constants ($k$)**: The maximum rate of a reaction under substrate-saturated conditions (e.g., $k_{\text{POC}}$ in $\text{yr}^{-1}$ or $\text{s}^{-1}$, or mineral precipitation rates $k_{\text{prec}}$).
2.  **Half-Saturation Constants ($K_m$)**: The concentration of an electron acceptor at which the microbial degradation rate reaches $50\%$ of its maximum. Lower $K_m$ means the microbe is efficient at scavenging low concentrations.
3.  **Inhibition Constants**: The threshold concentration of a more energetically favorable oxidant that shuts down a less favorable pathway (e.g., $O_2$ inhibiting iron reduction or sulfate reduction).


<a id="org3c64ffa"></a>

## Structure of `kinetic_constants_slow.py`

Constants are encapsulated in a callable function `get_reaction_constants(pH, phi)` so that pH-dependent speciation and porosity-dependent conversions can be calculated automatically:

```python
def get_reaction_constants(pH, phi):
    """Returns reaction rate constants and half-saturation values for the diagenetic system."""

    # 1. Primary organic carbon degradation constants
    velde = {
        "POC_fast": 1.2e-9,        # Organic carbon degradation rate (s^-1)
        "POC_slow": 3.8e-11,       # Refractory carbon degradation rate (s^-1)
        "TS2_O2": 1.5e-6,          # Sulfide oxidation rate
        "Fe3_hs": 8.0e-7,          # Sulfide-mediated iron reduction rate
        "Fe2_O2": 2.5e-5,          # Ferrous iron oxidation rate
    }

    # 2. Mineral kinetics (Precipitation, Dissolution, Solubility Products)
    velde["FeS_isp"] = 2.0e-3      # FeS apparent precipitation rate
    velde["FeS_isd"] = 1.0e-5      # FeS apparent dissolution rate
    velde["FeS_sp"] = 0.3162       # FeS apparent solubility product

    # 3. Microorganism half-saturation & inhibition thresholds (mmol/L)
    k_values = {
        "K_O2": 0.001,             # O2 limitation / inhibition threshold (1 uM)
        "K_TS2": 0.05,             # Sulfide tolerance threshold
        "K_SO4": 0.5,              # Sulfate limitation threshold (0.5 mM)
        "K_Fe3_diss_red": 50.0,    # Reactive iron limitation threshold (solid phase)
        "K_O2_TS2": 1e-4,          # O2 threshold for sulfide oxidation
    }

    return velde, k_values
```


<a id="org1fecbeb"></a>

# Step 3: Defining Reactions & Limiters (`fes_reactions.py`)

The reactions file is where you express your geochemical hypotheses. It contains two main sections:

1.  `limiters`: Kinetic switches and fractionating terms.
2.  `reactions`: The list of stoichiometric equations and their regulatory controls.


<a id="org739ad2b"></a>

## Understanding Limiters & Inhibitors

In sediment columns, microbes follow the classic redox ladder (Froelich et al., 1979). To model this mathematically:

-   When a microbe **needs** an electron acceptor (like $O_2$ or $SO_4^{2-}$), its rate is multiplied by a **Monod Limiter**: $$ L_{\text{Monod}}(C) = \frac{C}{C + K} $$
-   When a pathway is **suppressed** because a higher-energy oxidant is present, its rate is multiplied by an **Inhibition Term**: $$ I(C) = \frac{K}{C + K} = 1 - \frac{C}{C + K} $$

In MANTIS, you do **not** need to write out explicit fraction formulas for every reaction. You define clean base limiters in a dictionary:

```python
limiters = {
    # Implicit form: 1 / (C + K)
    # MANTIS automatically generates the limiter [C / (C + K)] and inhibitor [K / (C + K)]!
    "O2": "1.0 / (c.O2 + K_O2)",
    "O2_TS2": "1.0 / (c.O2 + mp.K_O2_TS2)",
    "TS2": "1.0 / (c.TS2 + K_TS2)",
    "SO4": "1.0 / (c.SO4 + K_SO4)",
    "Fe3_diss_red": "1.0 / (np.maximum(c.Fe3, 0.0) + K_Fe3_diss_red)",

    # Custom isotope explicit fractionation terms
    "SO4_alpha_explicit": "c.SO4 / (c.SO4 + 1e-3)",
    "TS2_alpha_explicit": "c.TS2 / (c.TS2 + 1e-3)",
}
```


<a id="org5336c1e"></a>

## Declaring Chemical Reactions

Reactions are declared as a list of dictionaries with natural chemical formulas. Notice how clean and readable the syntax is:

```python
reactions = [
    # -------------------------------------------------------------------------
    # 1. Aerobic Respiration: POC + O2 -> CO2
    # Limited by O2 availability
    # -------------------------------------------------------------------------
    {
        "reaction_name": "aerobic_respiration_fast",
        "reaction": "POC_fast + O2 -> CO2",
        "k_value_name": "POC_fast",
        "limiters": ["O2"],
    },

    # -------------------------------------------------------------------------
    # 2. Dissimilatory Iron Reduction: POC + 4 Fe3 -> 4 Fe2
    # Limited by reactive iron, INHIBITED by oxygen
    # -------------------------------------------------------------------------
    {
        "reaction_name": "dissimilatory_iron_reduction_fast",
        "reaction": "POC_fast + 4 Fe3 -> 4 Fe2_total",
        "k_value_name": "POC_fast",
        "limiters": ["Fe3_diss_red"],
        "inhibitors": ["O2"],
    },

    # -------------------------------------------------------------------------
    # 3. Sulfate Reduction: 2 POC + SO4 -> TS2 (Total Sulfide)
    # Limited by SO4, INHIBITED by oxygen and reactive iron
    # Includes sulfur isotope fractionation
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 4. Secondary Sulfide Oxidation: HS- + 2 O2 -> SO4
    # -------------------------------------------------------------------------
    {
        "reaction_name": "hs_oxidation",
        "reaction": "1 HS + 2 O2 -> 1 SO4",
        "k_value_name": "TS2_O2",
        "limiters": ["O2_TS2"],
        "isotope_species": {"SO4": "SO4_32", "TS2": "TS2_32", "HS": "HS_32"},
    },

    # -------------------------------------------------------------------------
    # 5. Mineral Precipitation & Dissolution (e.g. Iron Monosulfide FeS)
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
```


<a id="org21fe2f4"></a>

# Step 4: The Master Run Script (`run_velde_model.py`)

The driver script is your experiment control panel. It specifies the sediment accumulation rate, bioturbation mixing, domain length, and simulation duration.

```python
#!/usr/bin/env python
from mantis import DiageneticModel
from mantis.diff_lib import get_total_delta

if __name__ == "__main__":
    # 1. Initialize the model
    # MANTIS automatically associates species.py in the same folder
    model = DiageneticModel(
        reactions="fes_reactions.py",
        constants="kinetic_constants_slow.py",
        layout="plot_layout.py",
    )

    # 2. Configure sediment environmental parameters using human-friendly units!
    model.configure(
        max_depth="0.5 m",             # Total sediment column depth (50 cm)
        display_length="0.5 m",        # Depth shown in plots
        t_end="2 kyr",                 # Integrate for 2,000 years
        dt_max="1 year",               # Maximum allowed timestep
        dt_init="1 day",               # Initial startup timestep
        dt_min="1 second",             # Minimum timestep before solver gives up
        w="0.2 cm/yr",                 # Sediment accumulation / burial rate
        phi=0.8,                       # Porosity (80% porewater, 20% solids)
        isotopes=True,                 # Enable 34S/32S isotope tracking

        # Macrofaunal bioturbation mixing (sediment surface down to 6 cm)
        bioturbation={
            "BT0": "4 cm^2/year",      # Surface bio-diffusion coefficient
            "depth": "6 cm",           # Mixed layer depth
            "attenuation": "2 cm",     # Transition attenuation width
        },

        # Porewater bio-irrigation (burrow flushing by worms/crustaceans)
        bioirrigation={
            "BI0": 0.0,                # Set to 0.0 to disable; or e.g. "20 /year"
            "depth": "0 cm",
        },
    )

    # 3. Allow CLI overrides (e.g. running --video from the terminal)
    model.parse_cli_args()

    # 4. Run the simulation
    result = model.run()

    # 5. Save outputs and print summary
    df, csv_path = result.save_csv()
    result.save_state()

    if result.mp.isotopes:
        print(f"Buried Delta34S = {get_total_delta(result.c, result.mp):.2f} permil")

    # Render final publication plot
    result.plot_summary()
```


<a id="orgf185086"></a>

# Step 5: Designing Plot Layouts (`plot_layout.py`)

Geochemists usually want to see:

-   Dissolved porewater concentrations in $\text{mmol/L}$ (or $\mu\text{mol/L}$).
-   Solid minerals and organic carbon in **dry weight percent** ($\text{wt}\%$) or $&mu;\text{mol/g}$.
-   Stable isotopes in per mil ($\delta^{34}\text{S}$ in $\text{\permil}$).

The `plot_layout.py` file gives you complete control over figure layout, colors, and line attributes.


<a id="orgaefde21"></a>

## Converting Model Units to Geochemical Units

In the model solver, all concentrations are tracked in volumetric $\text{mmol/L}$. To convert them to $\text{wt}\%$ for your figures, MANTIS provides helper functions:

```python
from mantis.diff_lib import solid_conc_to_wt_percent, liquid_conc_to_wt_percent

# Convert solid FeS (in mmol/L) to wt% dry sediment:
# solid_conc_to_wt_percent(C_solid, molecular_weight, grain_density_g_cm3, porosity)
FeS_wt_percent = solid_conc_to_wt_percent(df.c_FeS, mw=56.0, d=2.6, phi=df.phi)
```


<a id="orgf5ff558"></a>

## Defining Multi-Panel Profiles

In `get_layout(df)`, you return a dictionary defining the figure geometry:

```python
def get_layout(df):
    phi = df.phi
    FeS_fe = solid_conc_to_wt_percent(df.c_FeS, 56, 2.6, phi)
    FeS2_fe = solid_conc_to_wt_percent(df.c_FeS2, 56, 2.6, phi)

    plot_description = {
        "figsize": (12, 8),
        "panels": [
            # Panel 1: Dissolved Oxidants (O2 and Sulfate)
            {
                "title": "Porewater Oxidants",
                "xlabel": "Concentration (mmol/L)",
                "curves": [
                    {"data": df.c_O2, "label": "O2", "color": "purple", "lw": 2},
                    {"data": df.c_SO4, "label": "SO4", "color": "blue", "lw": 2},
                ],
                "xlim": (0, 30),
            },

            # Panel 2: Authigenic Iron Minerals
            {
                "title": "Iron Minerals (wt% Fe)",
                "xlabel": "wt % Fe",
                "curves": [
                    {"data": FeS_fe, "label": "FeS", "color": "crimson", "lw": 2},
                    {"data": FeS2_fe, "label": "FeS2 (Pyrite)", "color": "darkorange", "lw": 2},
                ],
                "xlim": (0, 1.5),
            },

            # Panel 3: Sulfur Isotope Composition
            {
                "title": "d34S (VCDT)",
                "xlabel": "d34S (permil)",
                "curves": [
                    {"data": df.d_SO4, "label": "d34S-SO4", "color": "blue", "lw": 2},
                    {"data": df.d_TS2, "label": "d34S-Sulfide", "color": "orange", "lw": 2},
                ],
                "xlim": (-40, 40),
            },
        ],
    }
    return plot_description
```


<a id="orge7acd7b"></a>

# Step 6: Running the Model & Producing Time-Lapse Videos


<a id="orgee831e4"></a>

## Standard Execution

To run your simulation, simply run the driver script in your terminal:

```sh
python run_velde_model.py
```

During execution, MANTIS outputs progressive milestones indicating physical time reached, current timestep $\Delta t$, and Newton iterations:

```
Grid generated with 1108 points.
Starting Monolithic Block Newton-Raphson Solver. dt_init: 24.00 h
Monolithic solver: Analytical chemical Jacobian enabled.
Step 25, time=8.6 week (0%), dt=36.68 h, iters=6
Step 50, time=20.2 week (0%), dt=5.52 d, iters=5
Step 100, time=26.7 month (0%), dt=25.23 d, iters=6
...
Step 2129, time=2.0 ka (100%), dt=52.18 week, iters=1
Final Report: Maximum steps or end time reached in 2129 steps. Total Wall Time: 26.2s
State saved to run_velde_model_state.npz
Saved final PDF plot: run_velde_model.pdf
d34S = -23.59 permil
```


<a id="orgcf4fea3"></a>

## Generating a Time-Lapse Video

To observe the dynamics as reaction fronts establish, add the `--video` CLI flag followed by the recording interval:

```sh
python run_velde_model.py --video "1 year"
```

This spawns an asynchronous background video worker that renders each snapshot into an MP4 file (`run_velde_model.mp4`) without slowing down the core simulation solver!


<a id="orgf872aa8"></a>

## Command-Line Options Reference

You can query all options and their configured defaults at any time:

```sh
python run_velde_model.py --help
```

Key flags include:

-   `--t-end "5000 year"`: Override the simulation duration.
-   `--video "5 year"`: Enable video rendering at the specified interval.
-   `--solver-tol 1e-3`: Adjust the Newton convergence tolerance.
-   `--numerical-jacobian`: Run with finite-difference Jacobians instead of analytical SymPy Jacobians.


<a id="orgfeddb58"></a>

# Solver Mechanics & Physical Realism (Geochemist's Alert!)

A critical strength of MANTIS's monolithic implicit Block Newton solver is that it is **unconditionally numerically stable**: unlike classical explicit methods (which blow up if the timestep $\Delta t$ exceeds seconds or hours), MANTIS will **not crash** even when taking timesteps of $\Delta t = 1\text{ year}$ or more.

However, **numerical stability is not the same as physical realism**. Here is why this distinction matters:

> **CAUTION: Large $\Delta t$ and Reaction Front Smearing** When a sediment column contains extremely fast, irreversible reactions between opposing fronts—such as the suboxic-anoxic transition zone where dissolved hydrogen sulfide ($\text{HS}^-$) meets oxygen ($\text{O}_2$) or ferric iron ($\text{Fe}^{3+}$)—the natural reaction zone may physically be only a few millimeters thick.
> 
> If the solver takes very large time steps ($\Delta t = 1\text{ year}$), diffusion carries $\text{HS}^-$ and $\text{O}_2$ across several grid cells within a single time step. While the implicit solver computes a mathematically stable solution, it will **artificially broaden (smear)** the reaction front. In the simulation, sulfide and oxygen might appear to coexist over several centimeters, whereas in nature they react immediately to near-zero levels within millimeters.


<a id="org9a13232"></a>

## How to Ensure Physical Realism:

1.  **Cap $\Delta t_{\text{max}}$**: If you are studying transient front migration or sharp redox boundaries, set `dt_max`"1 month"= or `dt_max`"6 month"= rather than multiple years.
2.  **Refine the Reaction Zone Grid**: Use MANTIS's `make_grid2` with high resolution (e.g. $\Delta z \le 0.5\text{ mm}$) across the critical depth where the redox front sits.
3.  **Check Front Coexistence**: Verify that reactants do not unnaturally overlap by running diagnostic checks across your front.


<a id="org0cb5c1f"></a>

# Summary & Next Steps

You now have a complete overview of MANTIS's modular modeling system:

1.  **Species** are declared in `species.py` with phases, diffusion laws, and boundary conditions.
2.  **Kinetics** are set in `kinetic_constants_*.py` with half-saturation and rate values.
3.  **Reactions** are written in `fes_reactions.py` with natural chemistry and Monod/inhibition controls.
4.  **Environmental Settings** (depth, burial, time, bioturbation) are configured in your driver script.
5.  **Visuals** are customized in `plot_layout.py` in geochemist-friendly units ($\text{wt}\%$, $\text{\permil}$).

Happy modeling!
