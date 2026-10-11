"""model.py

Unified, declarative DiageneticModel interface for FiPyrite reaction-transport simulations.

Provides a clean object-oriented entry point that:
- Automatically resolves paths relative to the calling script.
- Centralizes species, phases, dynamic speciation, and default boundary conditions.
- Extracts pH from species.py (Hplus['bc']['pH']) as a single source of truth.
- Sets bioirrigation explicitly to zero by default.
- Provides built-in CLI argument parsing (--help, --solver, --video).
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pint

from fipyrite.diff_lib import (
    VariableArray as CellVariable,
    check_peclet_numbers,
    compute_bio_irrigation_alpha,
    compute_sigmoidal_db,
    data_container,
    diff_coeff,
    get_l_mass,
    get_total_delta,
    make_grid2,
    read_state,
    save_data,
    save_state,
)
from fipyrite.generate_equation_system import resolve_and_prepare_equations
from fipyrite.live_plot_lib import LivePlotter
from fipyrite.solver_calls import run_non_steady_state_solver_coupled as run_solver


class DiageneticModelResult:
    """Encapsulates the output of a completed DiageneticModel simulation."""

    def __init__(
        self,
        mp: data_container,
        c: data_container,
        k: data_container,
        species_list: List[str],
        z: np.ndarray,
        D_mol: data_container,
        diagenetic_reactions: Any,
        converged: str,
        step: int,
        total_time: float,
        df: Any = None,
        fqfn: Optional[str] = None,
    ):
        self.mp = mp
        self.c = c
        self.k = k
        self.species_list = species_list
        self.z = z
        self.D_mol = D_mol
        self.diagenetic_reactions = diagenetic_reactions
        self.converged = converged
        self.step = step
        self.total_time = total_time
        self.df = df
        self.fqfn = fqfn

    def save_csv(self) -> Tuple[Any, str]:
        """Saves tabular concentration and rate data to CSV."""
        df, fqfn = save_data(
            self.mp, self.c, self.k, self.species_list, self.z, self.D_mol, self.diagenetic_reactions
        )
        self.df = df
        self.fqfn = fqfn
        return df, fqfn

    def save_state(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Saves complete CellVariable array state to a compressed .npz archive."""
        if filepath is None:
            filepath = f"{self.mp.plot_name}_state.npz"
        filepath_str = str(filepath)
        save_state(self.c, filepath_str)
        return filepath_str

    def plot_summary(self, layout_file: Optional[Union[str, Path]] = None, outfile: Optional[str] = None) -> Optional[str]:
        """Renders the final summary plot PDF."""
        import fipyrite.plot_data_new as pdn
        layout = layout_file or getattr(self.mp, "layout_file", "plot_layout.py")
        if not Path(layout).exists() and hasattr(self.mp, "_caller_dir"):
            candidate = Path(self.mp._caller_dir) / Path(layout).name
            if candidate.exists():
                layout = str(candidate)
        if Path(layout).exists() and self.df is not None:
            pdf_out = outfile or f"{self.mp.plot_name}.pdf"
            try:
                plt_desc = pdn.load_layout_from_file(self.df, layout)
                pdn.plot(self.df, self.mp.max_depth, outfile=pdf_out, show=False, plot_description=plt_desc)
                return pdf_out
            except Exception as e:
                print(f"[DiageneticModel] Plot rendering skipped or failed: {e}")
                return None
        return None


class DiageneticModel:
    """Declarative Diagenetic Model Runner."""

    def __init__(
        self,
        species: Union[str, Path] = "species.py",
        reactions: Union[str, Path] = "reactions.py",
        constants: Union[str, Path] = "reaction_constants.py",
        equations: Optional[Union[str, Path]] = None,
        limiters: Optional[Union[str, Path]] = None,
        layout: Union[str, Path] = "plot_layout.py",
        experiment: Optional[str] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ):
        # 1. Determine caller directory and caller frame automatically
        caller_frame = None
        for frame_info in inspect.stack()[1:]:
            frame_file = frame_info.filename
            frame_name = Path(frame_file).name
            if (
                frame_name not in ("model.py", "__init__.py")
                and "_pytest" not in frame_file
                and "pluggy" not in frame_file
                and frame_name not in ("pytest", "pytest.py")
            ):
                caller_frame = frame_info
                break

        if base_dir is not None:
            self.base_dir = Path(base_dir).resolve()
        elif caller_frame is not None and caller_frame.filename != "<stdin>":
            self.base_dir = Path(caller_frame.filename).resolve().parent
        else:
            self.base_dir = Path.cwd()

        # 2. Resolve filenames relative to base_dir or cwd
        self.species_file = self._resolve(species)
        self.reactions_file = self._resolve(reactions)
        self.constants_file = self._resolve(constants)
        self.equations_file = self._resolve(equations) if equations else None
        self.limiters_file = self._resolve(limiters) if limiters else None
        self.layout_file = self._resolve(layout)

        # Derive experiment name automatically from caller filename stem
        if experiment is not None:
            self.experiment = experiment
        elif caller_frame is not None and caller_frame.filename not in ("<stdin>", "") and not Path(caller_frame.filename).name.startswith("pytest"):
            self.experiment = Path(caller_frame.filename).stem
        elif sys.argv and sys.argv[0] and not Path(sys.argv[0]).name.startswith("pytest"):
            self.experiment = Path(sys.argv[0]).stem
        else:
            self.experiment = "diagenetic_run"

        self.ureg = pint.UnitRegistry()
        self.Q_ = self.ureg.Quantity

        # Master parameter container
        self.mp = data_container()
        self.p_dict: Dict[str, Any] = {}

        # Default parameters
        self._init_defaults()

    def _resolve(self, path_like: Union[str, Path]) -> Path:
        p = Path(path_like)
        if p.is_absolute() and p.exists():
            return p
        if (self.base_dir / p).exists():
            return (self.base_dir / p).resolve()
        if (Path.cwd() / p).exists():
            return (Path.cwd() / p).resolve()
        return (self.base_dir / p).resolve()

    def _init_defaults(self) -> None:
        """Initializes default physical, geometric, and numerical solver parameters."""
        Q_ = self.Q_
        self.p_dict.update({
            "experiment": self.experiment,
            "plot_name": self.experiment,
            "state_data": None,
            "layout_file": str(self.layout_file),
            "video_dt": None,
            "video_dpi": 120,
            "video_queue_size": 100,
            "video_resume_threshold": 50,
            "video_workers": None,
            "start_time": 0,
            # Geometry
            "max_depth": 2.0,  # meters
            "initial_spacing": 0.0001,
            "reaction_zone_spacing": 0.0001,
            "max_spacing": 0.1,
            "reaction_zone": (0.0, 0.1),
            "display_length": 2.0,
            # Environmental defaults
            "temp": [10.0, 10.1],
            "w": Q_("0.2 cm/yr").to("m/s").magnitude,
            "advection": 0.0,
            "phi": 0.8,
            "POC_O2_ratio": 1.27,
            # Monod defaults (bulk or phase corrected)
            "K_O2": Q_("0.001 umol/cm^3").to("mol/m^3").magnitude,
            "K_O2_TS2": Q_("0.001 umol/cm^3").to("mol/m^3").magnitude,
            "K_TS2": Q_("0.1 umol/cm^3").to("mol/m^3").magnitude,
            "K_SO4": Q_("0.9 umol/cm^3").to("mol/m^3").magnitude,
            "K_Fe3_diss_red": Q_("10.4 umol/cm^3").to("mol/m^3").magnitude,
            "K_Fe3": 1e-3,
            # Benthic activity (bioirrigation explicitly disabled by default)
            "BT0": 0.0,
            "BT_depth": Q_("7.6 cm").to("meter").magnitude,
            "BT_attenuation": Q_("2 cm").to("meter").magnitude,
            "BI0": 0.0,  # explicitly 0.0
            "BI_depth": 0.0,
            # Isotopes
            "isotopes": True,
            "SO4_d": 21.0,
            "S0_d": 8.0,
            "msr_alpha": 1.07,
            "TS2_O2_alpha": 0.995,
            "S0_O2_alpha": 1.0,
            "dispro_hs_alpha": 0.993,
            "dispro_SO4_hs_split": 3.0,
            "h2s_hs_alpha": 0.99991542,
            "VCDT": 0.044162589,
            "K_epsilon_msr": 0.2,
            "K_epsilon_TS2_O2": 0.01,
            # Solver parameters
            "solver": "monolithic",
            "t_end": Q_("1 kyr").to("seconds").magnitude,
            "dt_min": Q_("1 minute").to("seconds").magnitude,
            "dt_init": Q_("1 month").to("seconds").magnitude,
            "dt_max": Q_("1 year").to("seconds").magnitude,
            "tolerance": 1e-12,
            "dt_tolerance": 1e-12,
            # Monolithic Newton Settings
            "max_newton_iters": 8,
            "newton_tol": 1e-3,
            "newton_norm": "linf",
            "newton_target_optimal": 4,
            "newton_max_acceptable": 6,
            "use_numerical_jacobian": False,
        })

    def configure(self, **kwargs) -> DiageneticModel:
        """Sets or overrides model configuration parameters with automatic unit handling.

        Supported parameters include:
        - max_depth: e.g. "10 m" or 10.0
        - t_end: e.g. "10 kyr" or seconds
        - dt_max: e.g. "1 year"
        - dt_init: e.g. "1 day"
        - dt_min: e.g. "1 second"
        - w: sedimentation rate, e.g. "0.2 cm/yr"
        - phi: sediment porosity (float)
        - pH: (float) will update Hplus boundary condition
        - isotopes: bool
        - bioturbation: dict with 'BT0', 'depth', 'attenuation'
        - bioirrigation: dict with 'BI0', 'depth' (defaults to 0.0)
        - bc_overrides: dict mapping species names to concentration/flux overrides
        """
        Q_ = self.Q_

        def _to_sec(val: Any) -> float:
            if isinstance(val, str):
                return Q_(val).to("seconds").magnitude
            return float(val)

        def _to_m(val: Any) -> float:
            if isinstance(val, str):
                return Q_(val).to("meter").magnitude
            return float(val)

        def _to_vel(val: Any) -> float:
            if isinstance(val, str):
                return Q_(val).to("m/s").magnitude
            return float(val)

        for key, val in kwargs.items():
            if key == "max_depth":
                self.p_dict["max_depth"] = _to_m(val)
            elif key == "display_length":
                self.p_dict["display_length"] = _to_m(val)
            elif key == "t_end":
                self.p_dict["t_end"] = _to_sec(val)
            elif key == "dt_max":
                self.p_dict["dt_max"] = _to_sec(val)
            elif key == "dt_init":
                self.p_dict["dt_init"] = _to_sec(val)
            elif key == "dt_min":
                self.p_dict["dt_min"] = _to_sec(val)
            elif key == "w":
                self.p_dict["w"] = _to_vel(val)
            elif key == "bioturbation" and isinstance(val, dict):
                if "BT0" in val:
                    b0 = val["BT0"]
                    self.p_dict["BT0"] = Q_(b0).to("m^2/second").magnitude if isinstance(b0, str) else float(b0)
                if "depth" in val:
                    self.p_dict["BT_depth"] = _to_m(val["depth"])
                if "attenuation" in val:
                    self.p_dict["BT_attenuation"] = _to_m(val["attenuation"])
            elif key == "bioirrigation" and isinstance(val, dict):
                if "BI0" in val:
                    b0 = val["BI0"]
                    self.p_dict["BI0"] = Q_(b0).to("1/second").magnitude if isinstance(b0, str) else float(b0)
                if "depth" in val:
                    self.p_dict["BI_depth"] = _to_m(val["depth"])
            elif key == "video" or key == "video_dt":
                self.p_dict["video_dt"] = val
            elif key == "pH":
                self.p_dict["pH"] = float(val)
            else:
                self.p_dict[key] = val

        return self

    def parse_cli_args(self, args: Optional[List[str]] = None) -> None:
        """Parses CLI arguments if requested, supporting category inspection and option overrides."""
        cur_solver = self.p_dict.get("solver", "monolithic")
        cur_tol = self.p_dict.get("newton_tol", 1e-3)
        cur_jac = self.p_dict.get("use_numerical_jacobian", False)
        cur_max_iters = self.p_dict.get("max_newton_iters", 8)
        cur_norm = self.p_dict.get("newton_norm", "linf")
        cur_opt_iters = self.p_dict.get("newton_target_optimal", 4)
        cur_max_acc_iters = self.p_dict.get("newton_max_acceptable", 6)

        cur_video = self.p_dict.get("video_dt", "None")
        cur_dpi = self.p_dict.get("video_dpi", 120)

        cur_tend = self.p_dict.get("t_end", "None")
        cur_dtmax = self.p_dict.get("dt_max", "None")
        cur_dtinit = self.p_dict.get("dt_init", "None")
        cur_dtmin = self.p_dict.get("dt_min", "None")
        cur_max_depth = self.p_dict.get("max_depth", "None")
        cur_disp_len = self.p_dict.get("display_length", "None")

        solver_section = [
            ("--solver <backend>", cur_solver, "Backend solver choice ('monolithic' or 'coupled')"),
            ("--solver-tol <float>", cur_tol, "Newton convergence tolerance (e.g. 1e-3)"),
            ("--numerical-jacobian", cur_jac, "Use finite-difference Jacobian instead of SymPy analytical"),
            ("--max-newton-iters <int>", cur_max_iters, "Maximum Newton iterations allowed per timestep"),
            ("--newton-norm <choice>", cur_norm, "Residual norm formulation ('linf' or 'wrms')"),
            ("--newton-target-optimal <int>", cur_opt_iters, "Target Newton iterations for timestep expansion"),
            ("--newton-max-acceptable <int>", cur_max_acc_iters, "Max acceptable iterations before timestep reduction"),
        ]

        video_section = [
            ("--video <interval>", cur_video, "Live video recording frame interval (e.g. '10 year', 'none')"),
            ("--video-dpi <int>", cur_dpi, "Resolution (DPI) of recorded video frames"),
        ]

        run_section = [
            ("--t-end <duration>", f"{cur_tend} s" if isinstance(cur_tend, (int, float)) else str(cur_tend), "Total duration of simulation (e.g. '10 kyr')"),
            ("--dt-max <timestep>", f"{cur_dtmax} s" if isinstance(cur_dtmax, (int, float)) else str(cur_dtmax), "Maximum adaptive timestep cap (e.g. '1 year')"),
            ("--dt-init <timestep>", f"{cur_dtinit} s" if isinstance(cur_dtinit, (int, float)) else str(cur_dtinit), "Initial timestep size (e.g. '1 day')"),
            ("--dt-min <timestep>", f"{cur_dtmin} s" if isinstance(cur_dtmin, (int, float)) else str(cur_dtmin), "Minimum allowable timestep before aborting"),
            ("--max-depth <length>", f"{cur_max_depth} m" if isinstance(cur_max_depth, (int, float)) else str(cur_max_depth), "Total depth of sediment model column (e.g. '10 m')"),
            ("--display-length <length>", f"{cur_disp_len} m" if isinstance(cur_disp_len, (int, float)) else str(cur_disp_len), "Plotted column depth for video and PDF summaries"),
        ]

        parser = argparse.ArgumentParser(
            description=(
                f"FiPyrite Diagenetic Model: {self.experiment}\n\n"
                "Interactive Category Query:\n"
                "  Passing category flags without arguments displays all configurable options and current defaults:\n"
                "    %(prog)s --solver        Show all solver options and current settings\n"
                "    %(prog)s --video         Show all video output options\n"
                "    %(prog)s --run           Show all time, grid, and domain options\n"
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=True,
        )

        # Solver options
        parser.add_argument(
            "--solver",
            nargs="?",
            const="__SHOW__",
            metavar="BACKEND",
            help=f"Solver backend: 'monolithic' or 'coupled' (or query section: pass flag with no value)",
        )
        parser.add_argument(
            "--solver-tol",
            "--newton-tol",
            nargs="?",
            const="__SHOW__",
            metavar="TOL",
            help=f"Newton convergence tolerance (default: {cur_tol})",
        )
        parser.add_argument(
            "--numerical-jacobian",
            action="store_true",
            help=f"Use finite-difference Jacobian instead of SymPy analytical (default: {cur_jac})",
        )
        parser.add_argument(
            "--max-newton-iters",
            type=int,
            metavar="ITERS",
            help=f"Max Newton iterations per step (default: {cur_max_iters})",
        )
        parser.add_argument(
            "--newton-norm",
            choices=["linf", "wrms"],
            metavar="NORM",
            help=f"Residual norm metric: 'linf' or 'wrms' (default: {cur_norm})",
        )
        parser.add_argument(
            "--newton-target-optimal",
            type=int,
            metavar="ITERS",
            help=f"Target iterations for timestep expansion (default: {cur_opt_iters})",
        )
        parser.add_argument(
            "--newton-max-acceptable",
            type=int,
            metavar="ITERS",
            help=f"Max iterations before timestep reduction (default: {cur_max_acc_iters})",
        )

        # Video options
        parser.add_argument(
            "--video",
            nargs="?",
            const="__SHOW__",
            metavar="INTERVAL",
            help=f"Live video recording interval, e.g. '10 year' or 'none' (or query section: pass flag with no value)",
        )
        parser.add_argument(
            "--video-dpi",
            nargs="?",
            const="__SHOW__",
            metavar="DPI",
            help=f"Video DPI resolution (default: {cur_dpi})",
        )

        # Run / Domain options
        parser.add_argument(
            "--run",
            nargs="?",
            const="__SHOW__",
            metavar="SETTINGS",
            help="Query time and domain run settings (pass flag with no value)",
        )
        parser.add_argument(
            "--t-end",
            metavar="DURATION",
            help=f"Simulation duration, e.g. '10 kyr' (default: {cur_tend} s)",
        )
        parser.add_argument(
            "--dt-max",
            metavar="TIMESTEP",
            help=f"Maximum timestep, e.g. '1 year' (default: {cur_dtmax} s)",
        )
        parser.add_argument(
            "--dt-init",
            metavar="TIMESTEP",
            help=f"Initial timestep, e.g. '1 day' (default: {cur_dtinit} s)",
        )
        parser.add_argument(
            "--dt-min",
            metavar="TIMESTEP",
            help=f"Minimum timestep, e.g. '1 second' (default: {cur_dtmin} s)",
        )
        parser.add_argument(
            "--max-depth",
            metavar="LENGTH",
            help=f"Sediment column depth, e.g. '10 m' (default: {cur_max_depth} m)",
        )
        parser.add_argument(
            "--display-length",
            metavar="LENGTH",
            help=f"Plotted display depth, e.g. '0.5 m' (default: {cur_disp_len} m)",
        )

        parsed, _ = parser.parse_known_args(args)

        # Check section queries first
        def _print_section(title: str, rows: List[Tuple[str, Any, str]]) -> None:
            print(f"\n[{self.experiment}] {title}:")
            for flag, cur_val, desc in rows:
                print(f"  {flag:<30} : {str(cur_val):<12} ({desc})")
            print()

        show_solver = parsed.solver in ("__SHOW__", "help", "?", "show") or parsed.solver_tol == "__SHOW__"
        show_video = parsed.video in ("__SHOW__", "help", "?", "show") or parsed.video_dpi == "__SHOW__"
        show_run = parsed.run in ("__SHOW__", "help", "?", "show")

        if show_solver or show_video or show_run:
            if show_solver:
                _print_section("Solver Configuration & Configurable Options", solver_section)
            if show_video:
                _print_section("Video Output Configuration & Configurable Options", video_section)
            if show_run:
                _print_section("Simulation Time & Domain Configuration & Configurable Options", run_section)
            sys.exit(0)

        # Apply solver overrides
        if parsed.solver is not None:
            if parsed.solver in ("monolithic", "coupled"):
                self.p_dict["solver"] = parsed.solver
            else:
                parser.error(f"argument --solver: invalid choice: '{parsed.solver}' (choose from 'monolithic', 'coupled')")

        if parsed.solver_tol is not None:
            self.p_dict["newton_tol"] = float(parsed.solver_tol)
        if parsed.numerical_jacobian:
            self.p_dict["use_numerical_jacobian"] = True
        if parsed.max_newton_iters is not None:
            self.p_dict["max_newton_iters"] = parsed.max_newton_iters
        if parsed.newton_norm is not None:
            self.p_dict["newton_norm"] = parsed.newton_norm
        if parsed.newton_target_optimal is not None:
            self.p_dict["newton_target_optimal"] = parsed.newton_target_optimal
        if parsed.newton_max_acceptable is not None:
            self.p_dict["newton_max_acceptable"] = parsed.newton_max_acceptable

        # Apply video overrides
        if parsed.video is not None:
            self.p_dict["video_dt"] = None if parsed.video.lower() in ("none", "null", "false") else parsed.video
        if parsed.video_dpi is not None:
            self.p_dict["video_dpi"] = int(parsed.video_dpi)

        # Apply run / domain overrides
        if parsed.t_end is not None:
            self.configure(t_end=parsed.t_end)
        if parsed.dt_max is not None:
            self.configure(dt_max=parsed.dt_max)
        if parsed.dt_init is not None:
            self.configure(dt_init=parsed.dt_init)
        if parsed.dt_min is not None:
            self.configure(dt_min=parsed.dt_min)
        if parsed.max_depth is not None:
            self.configure(max_depth=parsed.max_depth)
        if parsed.display_length is not None:
            self.configure(display_length=parsed.display_length)

    def _load_species_definitions(self) -> Dict[str, Any]:
        """Loads and parses the species definitions dictionary from species.py."""
        spec = importlib.util.spec_from_file_location("species_mod", self.species_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load species file: {self.species_file}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "species") or not isinstance(mod.species, dict):
            raise ValueError(f"Species file {self.species_file} must define a 'species' dictionary.")
        return mod.species

    def setup(self, plot_queue=None) -> Tuple[data_container, data_container, data_container, List[str], np.ndarray, data_container, Any]:
        """Assembles grid, species variables, transport coefficients, and compiles reaction equations."""
        self.mp.update(self.p_dict)
        self.mp._caller_dir = str(self.base_dir)

        # 1. Parse species definitions
        species_defs = self._load_species_definitions()

        # 2. Extract pH from Hplus if present
        if "Hplus" in species_defs and "bc" in species_defs["Hplus"]:
            h_bc = species_defs["Hplus"]["bc"]
            if isinstance(h_bc, dict) and "pH" in h_bc:
                if "pH" not in self.p_dict:
                    self.mp.pH = float(h_bc["pH"])

        # 3. Compile or load reaction equations
        prep_dict = dict(self.mp)
        prep_dict["reactions"] = str(self.reactions_file)
        prep_dict["species"] = str(self.species_file)
        prep_dict["kinetic_constants"] = str(self.constants_file)
        if self.limiters_file and self.limiters_file.exists():
            prep_dict["limiters"] = str(self.limiters_file)
        if self.equations_file:
            prep_dict["equations"] = str(self.equations_file)

        eq_mod, _rxn_p, _eq_p = resolve_and_prepare_equations(
            prep_dict, experiment_name=self.experiment, base_dir=self.base_dir
        )
        self.mp.reactions_module = eq_mod
        if hasattr(eq_mod, "get_limiters"):
            self.mp.get_limiters = eq_mod.get_limiters

        # 4. Load reaction constants
        spec_rc = importlib.util.spec_from_file_location("rc_mod", self.constants_file)
        if spec_rc is None or spec_rc.loader is None:
            raise ImportError(f"Cannot load reaction constants file: {self.constants_file}")
        mod_rc = importlib.util.module_from_spec(spec_rc)
        spec_rc.loader.exec_module(mod_rc)

        if hasattr(mod_rc, "get_reaction_constants"):
            get_rc_fn = mod_rc.get_reaction_constants
        else:
            raise AttributeError(f"{self.constants_file} must define get_reaction_constants(pH, phi, k_values).")

        # 5. Grid generation
        mesh, z = make_grid2(
            self.mp.max_depth,
            self.mp.initial_spacing,
            self.mp.reaction_zone_spacing,
            self.mp.max_spacing,
            self.mp.reaction_zone,
        )
        self.mp.mesh = mesh
        self.mp.z = z
        self.mp.grid_points = len(z)
        self.mp.phi = CellVariable(name="porosity", mesh=mesh, value=self.mp.phi)

        # 6. Evaluate k values
        k = data_container()
        _k1, k = get_rc_fn(self.mp.pH, self.mp.phi, k_values=k)
        self.mp.k = k

        # 7. Species lists & phase classification
        species_transport: List[str] = []
        species_all: List[str] = []
        bc_map: Dict[str, Dict[str, Any]] = {}
        D_mol = data_container()
        c = data_container()

        # Porosity volume ratios
        phi_val = self.mp.phi.value if hasattr(self.mp.phi, "value") else self.mp.phi
        self.mp.fac_s = phi_val / (1.0 - phi_val)

        # Sorption / speciation fractions
        self.mp.Fe2_diss = 1.0 / (1.0 + k.Fe2_p_eq) if hasattr(k, "Fe2_p_eq") else 1.0
        self.mp.Fe2_sorb = (k.Fe2_p_eq * self.mp.fac_s * self.mp.Fe2_diss) if hasattr(k, "Fe2_p_eq") else 0.0

        pKa1 = 7.0
        Ka1 = 10 ** (-pKa1)
        H_conc = 10 ** (-self.mp.pH)
        self.mp.h2s_frac = H_conc / (H_conc + Ka1)
        self.mp.hs_frac = Ka1 / (H_conc + Ka1)

        # Initialize CellVariables
        zeros = np.zeros(self.mp.grid_points)
        T_profile = np.linspace(self.mp.temp[0], self.mp.temp[1], self.mp.grid_points)

        # Sort species: included in transport vs diagnostic
        for sp_name, meta in species_defs.items():
            species_all.append(sp_name)
            setattr(D_mol, sp_name, zeros.copy())
            setattr(c, sp_name, CellVariable(name=sp_name, mesh=mesh, value=0.0, hasOld=True))

            is_inc = meta.get("include", True)
            is_iso = "isotope_of" in meta

            if is_inc:
                if not is_iso or self.mp.isotopes:
                    species_transport.append(sp_name)

            # Build default boundary conditions
            bc_meta = meta.get("bc")
            if bc_meta and isinstance(bc_meta, dict):
                top_val = bc_meta.get("top", 0.0)
                if isinstance(top_val, str):
                    # convert flux or concentration units
                    q_val = self.Q_(top_val)
                    if "yr" in top_val or "year" in top_val or "second" in top_val:
                        top_val = q_val.to("mol/(m^2 * second)").magnitude
                    else:
                        top_val = q_val.to("mol/m^3").magnitude
                bc_map[sp_name] = {
                    "top": top_val,
                    "type": bc_meta.get("type", "dissolved"),
                }

        # Apply any explicit user overrides to boundary conditions
        bc_overrides = self.mp.get("bc_overrides", {})
        for sp_k, ov in bc_overrides.items():
            if sp_k in bc_map:
                if isinstance(ov, dict):
                    bc_map[sp_k].update(ov)
                else:
                    bc_map[sp_k]["top"] = float(ov)

        # 8. Compute molecular diffusion coefficients from species recipes
        for sp_name, meta in species_defs.items():
            diff_spec = meta.get("diffusion")
            if diff_spec is None:
                continue
            if isinstance(diff_spec, str):
                # Inherit from another species
                if hasattr(D_mol, diff_spec):
                    setattr(D_mol, sp_name, getattr(D_mol, diff_spec))
            elif isinstance(diff_spec, dict):
                d_type = diff_spec.get("type", "diff_coeff")
                if d_type == "diff_coeff":
                    m0 = diff_spec.get("m0", 0.0)
                    m1 = diff_spec.get("m1", 0.0)
                    setattr(D_mol, sp_name, diff_coeff(T_profile, m0, m1, self.mp.phi))
                elif d_type == "boudreau_o2":
                    d_o2 = (
                        (0.2604 + 0.006363 * ((T_profile + 273.15) / 1.0))
                        * 1e-9
                        / (1.0 - np.log(phi_val**2))
                    )
                    setattr(D_mol, sp_name, d_o2)

        # Lumped Fe2_total / mixed-phase diffusion scaling (porewater retardation)
        for sp_name, meta in species_defs.items():
            if meta.get("phase") == "mixed" and hasattr(D_mol, sp_name):
                diss_frac = 1.0
                if sp_name == "Fe2_total":
                    diss_frac = getattr(self.mp, "Fe2_diss", 1.0)
                else:
                    part = meta.get("partition", {})
                    if "dissolved" in part:
                        try:
                            diss_frac = float(eval(part["dissolved"], {"k": k, "mp": self.mp, "np": np}))
                        except Exception:
                            diss_frac = 1.0
                setattr(D_mol, sp_name, getattr(D_mol, sp_name) * diss_frac)

        if hasattr(D_mol, "Fe2") and hasattr(D_mol, "Fe2_total"):
            D_mol.Fe2_total = D_mol.Fe2 * getattr(self.mp, "Fe2_diss", 1.0)

        # Bioturbation & Bio-irrigation profiles
        D_mol.D_irr = compute_bio_irrigation_alpha(z, self.mp.BI0, self.mp.BI_depth)
        D_mol.D_bio = compute_sigmoidal_db(z, self.mp.BT0, self.mp.BT_depth, self.mp.BT_attenuation)

        # 9. Isotope boundary conditions companion setup
        if self.mp.isotopes:
            if "SO4" in bc_map:
                self.mp.bc_SO4_32 = get_l_mass(bc_map["SO4"]["top"], self.mp.SO4_d, self.mp.VCDT)
                bc_map["SO4_32"] = {"top": self.mp.bc_SO4_32, "type": "dissolved"}
            if "TS2" in bc_map:
                self.mp.bc_TS2_32 = get_l_mass(bc_map["TS2"]["top"], 0.0, self.mp.VCDT)
                bc_map["TS2_32"] = {"top": self.mp.bc_TS2_32, "type": "dissolved"}
            if "S0" in bc_map:
                self.mp.bc_S0_32 = get_l_mass(bc_map["S0"]["top"], getattr(self.mp, "S0_d", 8.0), self.mp.VCDT)
                bc_map["S0_32"] = {"top": self.mp.bc_S0_32, "type": "particulate"}
            if "FeS" in species_defs and "FeS_32" in species_defs:
                bc_map["FeS_32"] = {"top": 0.0, "type": "particulate"}
            if "FeS2" in species_defs and "FeS2_32" in species_defs:
                bc_map["FeS2_32"] = {"top": 0.0, "type": "particulate"}

        # 10. Initial conditions and boundary constraints
        phi_top = phi_val[0] if hasattr(phi_val, "__getitem__") else phi_val
        for sp_name in species_transport:
            var = getattr(c, sp_name)
            props = bc_map.get(sp_name, {"top": 0.0, "type": "dissolved"})
            if props["type"] == "particulate":
                # Convert bulk surface flux J_bulk to solid-phase flux: J_solid = J_bulk / (1 - phi)
                J_solid = props["top"] / (1.0 - phi_top)

                D_total = getattr(D_mol, sp_name, 0.0) + D_mol.D_bio
                if not isinstance(D_total, CellVariable):
                    D_total = CellVariable(mesh=mesh, value=D_total)

                d_left = D_total.faceValue[mesh.facesLeft.value][0]
                if d_left > 1e-20:
                    # Robin BC: flux = w * C - D * dC/dx -> dC/dx = (w * C - J_solid) / D
                    var.faceGrad.constrain(
                        [(self.mp.w * var.faceValue - J_solid) / D_total.faceValue],
                        mesh.facesLeft,
                    )
                else:
                    val = J_solid / self.mp.w if self.mp.w > 0 else 0.0
                    var.setValue(val)
                    var.constrain(val, mesh.facesLeft)
            else:
                var.setValue(props["top"])
                var.constrain(props["top"], mesh.facesLeft)

            var.faceGrad.constrain([0.0], mesh.facesRight)

        # Restore state if specified
        if self.mp.state_data:
            read_state(c, self.mp.state_data)

        check_peclet_numbers(mesh, self.mp, D_mol, species_transport, bc_map)

        self._c = c
        self._k = k
        self._species_transport = species_transport
        self._species_all = species_all
        self._D_mol = D_mol
        self._bc_map = bc_map
        self._mesh = mesh

        return self.mp, c, k, species_transport, z, D_mol, eq_mod.diagenetic_reactions

    def run(self, plot_queue=None, **override_kwargs) -> DiageneticModelResult:
        """Runs the simulation and returns a DiageneticModelResult object."""
        if override_kwargs:
            self.configure(**override_kwargs)

        self.setup(plot_queue=plot_queue)

        plotter = None
        video_dt = getattr(self.mp, "video_dt", None)
        if plot_queue is None and video_dt is not None:
            output_path = f"{self.mp.plot_name}.pdf"
            video_path = os.path.abspath(f"{self.mp.plot_name}.mp4")
            plotter = LivePlotter(
                layout_path=self.mp.layout_file,
                display_length=self.mp.display_length,
                output_path=output_path,
                video_path=video_path,
                gui=False,
                max_queue_size=getattr(self.mp, "video_queue_size", 100),
                resume_threshold=getattr(self.mp, "video_resume_threshold", 50),
                video_dpi=getattr(self.mp, "video_dpi", 120),
                video_workers=getattr(self.mp, "video_workers", None),
            )
            plotter.start()
            plot_queue = plotter.queue

        try:
            diag_rxn_func = self.mp.reactions_module.diagenetic_reactions

            def default_equilibrium_reactions(mp, c, k, f, RATES, dt):
                inst_rxns = getattr(mp, "instantenous_reactions", [])
                for r in inst_rxns:
                    sig = inspect.signature(r[0])
                    if "f" in sig.parameters:
                        r[0](c, r[1], mp, dt, RATES, f=f)
                    else:
                        r[0](c, r[1], mp, dt, RATES)
                return f, RATES

            step, max_change = run_solver(
                self.mp,
                self._c,
                self._species_all,
                self._species_transport,
                self._k,
                diag_rxn_func,
                default_equilibrium_reactions,
                self._mesh,
                self._D_mol,
                self._bc_map,
                self.mp.z,
                plot_queue=plot_queue,
            )
        finally:
            if plotter:
                plotter.stop()

        max_steps = getattr(self.mp, "max_steps", None)
        converged = "Yes" if (max_steps is None or step < max_steps) else "No"
        total_time = getattr(self.mp, "current_time", getattr(self.mp, "t_end", 0.0))

        result = DiageneticModelResult(
            mp=self.mp,
            c=self._c,
            k=self._k,
            species_list=self._species_transport,
            z=self.mp.z,
            D_mol=self._D_mol,
            diagenetic_reactions=diag_rxn_func,
            converged=converged,
            step=step,
            total_time=total_time,
        )
        return result
