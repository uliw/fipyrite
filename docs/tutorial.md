# MANTIS quickstart guide 

## Overall Model Architecture: What is Where?

A typical MANTIS project directory contains five primary configuration files, alongside the master run script. An working example is provided in the ~./examples~ directory 

```
./velde_et_al_2016/
├── species.py                 <- Who lives in the sediment? (chemical species, phases, diffusion, BCs)
├── fes_reactions.py           <- What reactions happen? (stoichiometry, kinetics, limiters)
├── kinetic_constants_fast.py  <- How fast do reactions happen? (rate constants, half-saturation values)
├── plot_layout.py             <- How should the profiles look? (panels, colors, geochemist units like wt%)
└── run_velde_model_fast.py         <- Master driver script (sediment accumulation rate, depth, duration)
```

When you execute `python run_velde_model_fast.py`, MANTIS automatically detects these files, ensures the underlying equation system is up to date, solves the time evolution of the sediment column, and produces your outputs:

-   `<model_name>.pdf`: A multi-panel summary plot of porewater and solid-phase profiles.
-   `<model_name>.csv`: A tabular export of concentrations, reaction rates, and isotope ratios across depth.
-   `<model_name>_state.npz`: The complete numerical state of the simulation for restarting or continuing runs.
-   `<model_name>.mp4` (optional): An animated video showing the transient evolution


## Step 1: Defining Chemical Species (`species.py`)

The `species.py` file is your geochemical inventory. It declares every chemical entity that exists in your model domain.

### Understanding Phases: Dissolved vs. Solid vs. Mixed

In a sediment column, species exist in different physical compartments:

1.  **Liquid (Porewater)**: Solutes dissolved in the pore fluid (e.g., $\text{SO}_4^{2-}$, $\text{O}_2$, $\text{HS}^-$). Their concentrations are expressed per unit volume of porewater ($\text{mmol/L}_{\text{pw}}$) and their inventory is scaled by the sediment porosity ($\phi$). They move by molecular diffusion, porewater advection, and bio-irrigation.
2.  **Solid (Particulate)**: Minerals and particulate organic carbon bound in the sediment framework (e.g., $\text{FeS}_2$, reactive iron $\text{Fe}^{3+}$, organic carbon $\text{POC}$). Their inventory is scaled by the solid fraction ($1 - \phi$). They move solely via sediment accumulation (burial) and macrofaunal bioturbation.
3.  **Mixed Pools**: Lumped entities like total ferrous iron ($\text{Fe}^{2+}_{\text{total}}$) that partition rapidly into a dissolved fraction and a surface-adsorbed fraction.


### Example: Standard Species Configuration

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


### Handling Diffusion Coefficients

For dissolved ions, molecular diffusion coefficients in sediment are calculated using the empirical linear temperature formulation from **Boudreau (1996)** corrected for tortuosity via Archie's law: $$ D_0(T) = (m_0 + m_1 \cdot T) \times 10^{-10} \quad [\text{m}^2/\text{s}] $$ $$ D_{\text{sed}} = \frac{D_0(T)}{1 - \ln(\phi^2)} $$

For ions following this relationship, you simply set:

```python
"diff_coeff_function": "diff_coeff",
"diffusion": {"m0": 4.88, "m1": 0.232},
```



#### Custom Diffusion Functions (e.g., Dissolved Oxygen)

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

## Concentration Units & Conversions: Inputs, Internal Engine, and Outputs

A common source of confusion in diagenetic modeling is the distinction between porewater solute concentrations, solid sediment mineral concentrations, and geochemical laboratory units (such as $\text{wt}\%$ or $\mu\text{mol/g}$). MANTIS handles physical units transparently while maintaining strict mass conservation across phases.

### 1. Default Input Units

When defining your model setup and boundary conditions:

- **Driver Script Parameters (`run_velde_model_fast.py`)**:
  MANTIS supports human-readable strings parsed via Python's `pint` physical unit library:
  - Lengths and depths: `"0.5 m"`, `"50 cm"`
  - Times and time steps: `"0.5 kyr"`, `"1 year"`, `"24 hours"`
  - Sedimentation rates: `"0.2 cm/yr"`, `"1 mm/a"`
  - Bioturbation coefficients: `"4 cm^2/year"`, `"1e-9 m^2/s"`

- **Boundary Conditions (`species.py`)**:
  - **Dissolved Solutes (`type: "dissolved"`)**:
    Upper boundary concentrations can be specified as a numeric float or a unit string. The default numeric input is **$\text{mmol/L}$** (which is identical to $\text{mol/m}^3$ or $\text{mM}$).
    ```text
    "bc": {"top": 28.2, "type": "dissolved"}         # 28.2 mmol/L (seawater sulfate)
    "bc": {"top": "280 umol/L", "type": "dissolved"} # parsed automatically to 0.28 mmol/L
    ```
  - **Particulate Solids (`type: "particulate"`)**:
    Deposition at the sediment-water interface is specified as a depositional surface flux:
    ```text
    "bc": {"top": "12 umol/(cm^2 * yr)", "type": "particulate"}
    ```
    MANTIS automatically converts this to internal flux units ($\text{mol}/(\text{m}^2 \cdot \text{s})$) during initialization.

### 2. How Units Are Handled Internally by the Solver

Under the hood, MANTIS standardizes all transport and reaction PDEs into SI-derived volumetric units:
$$\text{Internal Concentration Unit} = \mathbf{mol/m^3} \quad (\equiv 1\ \text{mmol/L} \equiv 1\ \text{mM} \equiv 1000\ \mu\text{mol/L} \equiv 1\ \mu\text{mol/cm}^3)$$

Crucially, concentrations are defined **per unit volume of their respective phase**:
1. **Liquid Solutes**: Expressed per volume of porewater ($\text{mol / m}^3_{\text{liquid}} = \text{mmol / L}_{\text{porewater}}$). The bulk sediment inventory is $\phi \cdot C_{\text{liquid}}$.
2. **Solid Minerals**: Expressed per volume of pure solid sediment ($\text{mol / m}^3_{\text{solid}} = \text{mmol / L}_{\text{solid}}$). The bulk sediment inventory is $(1 - \phi) \cdot C_{\text{solid}}$.
3. **Phase Transfer & Reactions**: Whenever a reaction couples a solid phase and a liquid phase (for example, $\text{Fe}^{2+} + \text{HS}^- \to \text{FeS}$), MANTIS scales the source and sink terms with the appropriate phase ratio ($(1-\phi)/\phi$ or $\phi/(1-\phi)$) so that mass is strictly conserved.
4. **Reaction Rates**: Solved internally in $\text{mol} / (\text{m}^3_{\text{phase}} \cdot \text{s})$.
5. **Depths and Transport**: Depth $z$ is in meters ($\text{m}$), time $t$ in seconds ($\text{s}$), and diffusion coefficients in $\text{m}^2/\text{s}$.

### 3. Output Units in Result Files (`.csv`)

When the simulation completes or exports data (`<model_name>.csv`):
- `c_<species>`: Reported directly in **$\text{mmol/L}$** (numerically equivalent to $\text{mol/m}^3$):
  - Solutes (e.g., `c_SO4`, `c_O2`, `c_TS2`) are in **$\text{mmol / L of porewater}$**.
  - Solids (e.g., `c_FeS`, `c_Fe3`, `c_POC_slow`) are in **$\text{mmol / L of pure solid}$**.
- `f_<species>`: Net production/consumption reaction rates in **$\text{mol} / (\text{m}^3 \cdot \text{s})$**.
- `d_<species>`: Isotope delta values (e.g., `d_SO4`, `d_FeS`) in **$\permil$** (per mil relative to VCDT).
- `z`: Depth below sediment-water interface in **meters** ($\text{m}$).
- `phi` and `w`: Porosity (fraction, dimensionless) and burial velocity ($\text{m/s}$).

### 4. Converting Solid Concentrations to Standard Geochemical Units

Geochemists typically report solid mineral abundances as dry-sediment mass fractions ($\text{wt}\%$) or mass-normalized molar concentrations ($\mu\text{mol/g}$). MANTIS includes built-in conversion utilities in `mantis.diff_lib`:

#### A. Converting to $\mu\text{mol / g}$ (micromoles per gram of dry sediment)
To convert a solid concentration $C_{\text{solid}}$ ($\text{mmol/L of solid} = \mu\text{mol/cm}^3_{\text{solid}}$) to $\mu\text{mol/g}$ of dry sediment, divide by the mineral grain density $\rho_{\text{sed}}$ (typically $2.6\ \text{g/cm}^3$ for siliciclastic sediments):
$$C_{\mu\text{mol/g}} = \frac{C_{\text{solid}}}{\rho_{\text{sed}}}$$

In Python:
```python
from mantis.diff_lib import solid_conc_to_umol_per_g

# df.c_FeS is in mmol/L of solid; density in g/cm^3
c_fes_umol_g = solid_conc_to_umol_per_g(df.c_FeS, d=2.6)
```

#### B. Converting to Dry Weight Percent ($\text{wt}\%$)
To convert $C_{\text{solid}}$ to weight percent of dry sediment matrix given molecular weight $M_w$ ($\text{g/mol}$):
$$\text{wt}\% = \frac{C_{\text{solid}} \cdot M_w}{\rho_{\text{sed}} \cdot 10^4}$$

In Python:
```python
from mantis.diff_lib import solid_conc_to_wt_percent

# Example: FeS (Mw = 87.91 g/mol, density = 2.6 g/cm^3)
fes_wt_pct = solid_conc_to_wt_percent(df.c_FeS, mw=87.91, d=2.6, phi=df.phi)
```


## Step 2: Defining Kinetic Constants (`kinetic_constants_*.py`)

The kinetic constants file defines how fast reactions occur and the affinity thresholds of microbial communities.

In diagenesis, we typically distinguish:

1.  **Rate Constants ($k$)**: The maximum rate of a reaction under substrate-saturated conditions (e.g., $k_{\text{POC}}$ in $\text{yr}^{-1}$ or $\text{s}^{-1}$, or mineral precipitation rates $k_{\text{prec}}$).
2.  **Half-Saturation Constants ($K_m$)**: The concentration of an electron acceptor at which the microbial degradation rate reaches $50\%$ of its maximum. Lower $K_m$ means the microbe is efficient at scavenging low concentrations.
3.  **Inhibition Constants**: The threshold concentration of a more energetically favorable oxidant that shuts down a less favorable pathway (e.g., $O_2$ inhibiting iron reduction or sulfate reduction).


### Structure of `kinetic_constants_slow.py`

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



## Step 3: Defining Reactions & Limiters (`fes_reactions.py`)

The reactions file is where you express your geochemical hypotheses. It contains two main sections:

1.  `limiters`: Kinetic switches and fractionating terms.
2.  `reactions`: The list of stoichiometric equations and their regulatory controls.



### Understanding Limiters & Inhibitors

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



### Declaring Chemical Reactions

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



## Step 4: The Master Run Script (`run_velde_model_fast.py`)

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

        # Spatial grid & reaction zone refinement (optional, defaults provided)
        initial_spacing="0.1 mm",      # Spacing at sediment-water interface (z=0)
        reaction_zone=("0.1 cm", "1.0 cm"), # Critical depth window of sharp redox fronts
        reaction_zone_spacing="0.05 mm",    # Dense, uniform spacing in the reaction zone
        max_spacing="1.0 cm",          # Coarser spacing in deep sediment
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


### Spatial Grid Configuration & Reaction Zone Resolution

Diagenetic models must resolve spatial scales ranging from sub-millimeter oxygen penetration depths to deep sediment columns spanning decimeters to hundreds of meters. A uniform fine grid across a 50-meter column would require millions of cells and slow down calculations unnecessarily.

MANTIS solves this using an adaptive geometric-mesh builder (`make_grid2`) that refines where steep gradients occur and expands into the deep sediment.

### 1. How Grid Generation Works (`make_grid2`)
The grid structure is divided into four seamless stages from the sediment surface to depth:
1. **Surface Section**: Starts at `initial_spacing` and geometrically refines cells downward toward `reaction_zone[0]`.
2. **Reaction Zone Section**: Maintains a constant, ultra-fine cell size (`reaction_zone_spacing`) across the depth interval `reaction_zone = (z_start, z_end)`.
3. **Coarsening Section**: Below the reaction zone, cell size expands geometrically by factor $r = 1.05$ per cell.
4. **Deep Uniform Section**: Once cell size reaches `max_spacing`, the remainder of the domain down to `max_depth` uses uniform large cells.

#### 2. The Geochemical Importance of Reaction Zone Resolution
Why does grid spacing in the reaction zone matter so much?
Fast bimolecular redox reactions—such as ferrous iron oxidation ($\text{Fe}^{2+} + \text{O}_2 \to \text{Fe}^{3+}$) or sulfide oxidation—occur across exceedingly narrow interfaces. The physical reaction front width $\delta_{\text{phys}}$ is governed by the balance between molecular diffusion and reaction kinetics:
$$\delta_{\text{phys}} \sim \sqrt{\frac{D_{\text{eff}}}{k \cdot C_{\text{scale}}}}$$
For transport-controlled reactions with large kinetic constants ($k = 10^7\ \text{M}^{-1}\text{s}^{-1}$), $\delta_{\text{phys}}$ is often **under $0.05\text{ mm}$**.

If your numerical grid cell size $\Delta z$ is larger than $2 \delta_{\text{phys}}$, the local Damköhler number is large ($Da_{II} \gg 1$). The finite-volume discretization cannot resolve the sharp counter-gradients, producing a numerical artifact called **artificial reactant coexistence**:
- $\text{Fe}^{2+}$ and $\text{O}_2$ artificially overlap in the same cell.
- Reactants diffuse past each other instead of reacting completely at the interface.
- Apparent reaction rates become smeared over several millimeters to centimeters.

#### 3. Automated Coexistence Diagnostic
MANTIS automatically runs a post-simulation front diagnostic (`check_reactant_coexistence`) at the end of every run. If any reaction front is under-resolved, MANTIS outputs a clear warning:
```text
[WARNING: Numerical Reactant Co-existence Detected]
  Reaction: 'Fe2 Oxidation by O2' (fe2 + O2)
  Front location (peak): z = 0.21 cm (depth cell index: 21)
  Local grid resolution: dz = 0.10 mm
  Estimated physical front width: 0.01 mm
  Front cell span (N_cells): 6 cell(s)
  Reasons: Local grid cell dz (0.10 mm) > 2x physical front width (0.01 mm)
  --> Action recommended: In make_grid2(), refine the mesh around z = [0.19, 0.23] cm to reaction_zone_spacing <= 0.01 mm.
```

When you see this warning, simply update your driver script's `model.configure()` with the recommended reaction zone window and spacing:
```python
model.configure(
    reaction_zone=("0.15 cm", "0.30 cm"),
    reaction_zone_spacing="0.01 mm",
)
```


## Step 5: Designing Plot Layouts (`plot_layout.py`)

Geochemists usually want to see:

-   Dissolved porewater concentrations in $\text{mmol/L}$ (or $\mu\text{mol/L}$).
-   Solid minerals and organic carbon in **dry weight percent** ($\text{wt}\%$) or $&mu;\text{mol/g}$.
-   Stable isotopes in per mil ($\delta^{34}\text{S}$ in $\text{\permil}$).

The `plot_layout.py` file gives you complete control over figure layout, colors, and line attributes.


### Converting Model Units to Geochemical Units

In the model solver, all concentrations are tracked in volumetric $\text{mmol/L}$. To convert them to $\text{wt}\%$ for your figures, MANTIS provides helper functions:

```python
from mantis.diff_lib import solid_conc_to_wt_percent, liquid_conc_to_wt_percent

# Convert solid FeS (in mmol/L) to wt% dry sediment:
# solid_conc_to_wt_percent(C_solid, molecular_weight, grain_density_g_cm3, porosity)
FeS_wt_percent = solid_conc_to_wt_percent(df.c_FeS, mw=56.0, d=2.6, phi=df.phi)
```


### Defining Multi-Panel Profiles

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


## Step 6: Running the Model & Producing Time-Lapse Videos

### Standard Execution

To run your simulation, simply run the driver script in your terminal:

```sh
python run_velde_model_fast.py
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


### Generating a Time-Lapse Video

To observe the dynamics as reaction fronts establish, add the `--video` CLI flag followed by the recording interval:

```sh
python run_velde_model_fast.py --video "1 year"
```

This spawns an asynchronous background video worker that renders each snapshot into an MP4 file (`run_velde_model.mp4`) without slowing down the core simulation solver!


## Command-Line Options Reference

You can query all options and their configured defaults at any time:

```sh
python run_velde_model_fast.py --help
```

Key flags include:

-   `--t-end "5000 year"`: Override the simulation duration.
-   `--video "5 year"`: Enable video rendering at the specified interval.
-   `--solver-tol 1e-3`: Adjust the Newton convergence tolerance.
-   `--numerical-jacobian`: Run with finite-difference Jacobians instead of analytical SymPy Jacobians.


## Solver Mechanics & Physical Realism (Geochemist's Alert!)

A critical strength of MANTIS's monolithic implicit Block Newton solver is that it is **unconditionally numerically stable**: unlike classical explicit methods (which blow up if the timestep $\Delta t$ exceeds seconds or hours), MANTIS will **not crash** even when taking timesteps of $\Delta t = 1\text{ year}$ or more.

However, **numerical stability is not the same as physical realism**. Here is why this distinction matters:

> **CAUTION: Large $\Delta t$ and Reaction Front Smearing** When a sediment column contains extremely fast, irreversible reactions between opposing fronts—such as the suboxic-anoxic transition zone where dissolved hydrogen sulfide ($\text{HS}^-$) meets oxygen ($\text{O}_2$) or ferric iron ($\text{Fe}^{3+}$)—the natural reaction zone may physically be only a few millimeters thick.
> 
> If the solver takes very large time steps ($\Delta t = 1\text{ year}$), diffusion carries $\text{HS}^-$ and $\text{O}_2$ across several grid cells within a single time step. While the implicit solver computes a mathematically stable solution, it will **artificially broaden (smear)** the reaction front. In the simulation, sulfide and oxygen might appear to coexist over several centimeters, whereas in nature they react immediately to near-zero levels within millimeters.

### How to Ensure Physical Realism:

1.  **Cap $\Delta t_{\text{max}}$**: If you are studying transient front migration or sharp redox boundaries, set `dt_max`"1 month"= or `dt_max`"6 month"= rather than multiple years.
2.  **Refine the Reaction Zone Grid**: Use MANTIS's `make_grid2` with high resolution (e.g. $\Delta z \le 0.5\text{ mm}$) across the critical depth where the redox front sits.
3.  **Check Front Coexistence**: Verify that reactants do not unnaturally overlap by running diagnostic checks across your front.



# Summary & Next Steps

You now have a complete overview of MANTIS's modular modeling system:

1.  **Species** are declared in `species.py` with phases, diffusion laws, and boundary conditions.
2.  **Kinetics** are set in `kinetic_constants_*.py` with half-saturation and rate values.
3.  **Reactions** are written in `fes_reactions.py` with natural chemistry and Monod/inhibition controls.
4.  **Environmental Settings** (depth, burial, time, bioturbation) are configured in your driver script.
5.  **Visuals** are customized in `plot_layout.py` in geochemist-friendly units ($\text{wt}\%$, $\text{\permil}$).

Happy modeling!
