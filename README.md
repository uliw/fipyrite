# MANTIS: A *M*onolithic *A*ssembly of *N*onlinear *T*ransport *I*nteractions *S*olver

**MANTIS** is a high-performance 1D reactive-transport modeling (RTM) framework written in Python, designed to simulate stiff, tightly coupled advection-diffusion-reaction networks.

---

## What is MANTIS?

Reactive-transport systems in natural environments—such as marine sediments, groundwater aquifers, and geochemical reactors—involve simultaneous transport (advection, molecular diffusion, bio-irrigation, and bioturbation) coupled with complex networks of nonlinear chemical, mineral, and biological reactions.

MANTIS provides:
- **Declarative Geochemical Model Definition**: Define chemical species, phases, reaction stoichiometries, and kinetic limiters (Monod, inhibition, threshold laws) in intuitive, human-readable Python configuration files.
- **Automated Symbolic Jacobian Generation**: Automatically parses reaction networks and compiles analytical chemical Jacobians using symbolic computer algebra (`SymPy`), eliminating error-prone manual differentiation.
- **Monolithic Block Newton-Raphson Solver**: Solves all chemical species and transport equations simultaneously in a coupled block-tridiagonal system using a Numba-accelerated Block Thomas algorithm.
- **Adaptive Time Stepping**: Features PID-controlled adaptive time stepping with automatic step rejection and convergence recovery.
- **Unified Declarative Interface**: Configure and run simulations through `DiageneticModel`, complete with automatic CLI argument handling, checkpointing, publication-ready summary plots, and asynchronous time-lapse video generation.

---

## How is MANTIS Different from Other Python RTM Packages?

Most existing Python and geochemical reactive-transport modeling packages rely on **operator splitting** (such as Sequential Non-Iterative Approach [SNIA] or Sequential Iterative Approach [SIA]) or **decoupled Picard (Gauss-Seidel) iterations**:

1. **Monolithic Coupling vs. Operator Splitting**:
   - *Operator splitting* alternates between solving pure transport and pure batch chemistry over a time step. While simple, it introduces splitting errors and often requires tiny time steps ($\Delta t$ on the order of seconds or hours) to avoid artificial oscillations or numerical divergence when reactions are fast and stiff.
   - *Picard iterations* solve each chemical species sequentially, lagging cross-species couplings from the previous iteration. At sharp reaction fronts (e.g., where sulfide meets oxygen or ferric iron), this decoupling frequently causes severe numerical lag oscillations.
   - **MANTIS solves the entire system monolithically**: Transport and reactions for all species are assembled into a single unified block-tridiagonal linear system at each Newton step. This achieves true simultaneous coupling and quadratic Newton convergence.

2. **Exact Symbolic Jacobians vs. Finite Differences**:
   - Rather than approximating derivatives via costly numerical perturbations (finite differences) or requiring users to manually code and maintain hundreds of lines of Jacobian terms, MANTIS computes exact analytical Jacobians symbolically via `SymPy` ahead of time and compiles them to optimized machine code.

3. **High-Performance Native Numba Engine**:
   - The core block-tridiagonal solver and residual evaluations run directly in compiled machine code via Numba. This delivers C-like performance without external heavy PDE framework dependencies.

4. **Unconditional Numerical Stability**:
   - Thanks to its fully implicit monolithic formulation, MANTIS does not suffer from Courant-Friedrichs-Lewy (CFL) or diffusion time step stability restrictions, enabling long-term simulations (centuries to millennia) with large time steps.

---

## Scope and General Applicability

While the bundled examples and default templates are geared toward **marine sediment early diagenesis** (e.g., organic matter mineralization cascades, dissimilatory iron and sulfate reduction, iron sulfide and pyrite precipitation, and stable sulfur isotope fractionation $\delta^{34}\text{S}$), **the underlying numerical formulation is domain-agnostic**.

The monolithic advection-diffusion-reaction solver can be directly applied to any 1D reactive transport problem involving:
- Stiff, multi-species nonlinear reaction networks
- Combined particulate (solid) and solute (porewater/fluid) phases
- Advection, molecular and eddy diffusion, and nonlocal exchange processes
- Stable isotope fractionation and multi-isotope tracking

---

## Current Limitations

While MANTIS provides robust and unconditionally stable solutions for stiff reactive transport, the current release has the following limitations:

1. **Time-Invariant Spatial Mesh**:
   - The spatial grid $z$ is static throughout the simulation. Dynamic grid adaptation, moving boundaries, or expanding domains (e.g., physical sediment column growth over time) are not currently supported.
2. **Time-Invariant Porosity**:
   - The porosity profile $\phi(z)$ is stationary in time (steady-state compaction). Transient sediment compaction or compaction-driven dynamic fluid expulsion over time is not currently modeled.
3. **Time-Invariant Temperature**:
   - The temperature profile $T(z)$ is steady-state. Dynamic thermal evolution and time-dependent diffusion coefficients or rate constants driven by non-isothermal thermal transients are not currently supported.

---

## Installation

MANTIS requires Python 3.12 or newer. Clone the repository and install in editable mode:

```bash
git clone https://github.com/uliw/mantis.git
cd mantis
pip install -e .
```

Dependencies include `numpy`, `pandas`, `matplotlib`, `numba`, `sympy`, `chempy`, and `watchdog`.

---

## Quick Start Example

A typical simulation directory contains declarative configuration files defining species, reactions, rate constants, and plot layouts:

```python
from mantis import DiageneticModel

# 1. Initialize the model (auto-detects species.py, reactions, constants in the directory)
model = DiageneticModel(
    reactions="fes_reactions.py",
    constants="kinetic_constants_slow.py",
    layout="plot_layout.py",
    experiment="marine_diagenesis",
)

# 2. Configure environmental and grid parameters
model.configure(
    max_depth="0.5 m",
    t_end="2 kyr",
    dt_init="1 day",
    dt_max="1 year",
    w="0.2 cm/yr",
    phi=0.8,
    bioturbation={
        "BT0": "4 cm^2/year",
        "depth": "6 cm",
        "attenuation": "2 cm",
    },
)

# 3. Enable CLI flag overrides (--video, --solver-tol, --t-end, etc.)
model.parse_cli_args()

# 4. Run the simulation
result = model.run()

# 5. Export results and summary plots
df, csv_path = result.save_csv()
result.plot_summary()
```

---

## Running Tests

Run the unit test suite using `pytest`:

```bash
pytest
```

---

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.