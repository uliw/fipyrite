"""model.py

Unified, declarative DiageneticModel interface for MANTIS reaction-transport simulations.

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

from mantis.diff_lib import (
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
from mantis.generate_equation_system import resolve_and_prepare_equations
from mantis.live_plot_lib import LivePlotter
from mantis.monolithic_solver import run_non_steady_state_solver_monolithic as run_solver


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
        import mantis.plot_data_new as pdn
        layout = layout_file or getattr(self.mp, "layout_file", "plot_layout.py")
        p_layout = Path(layout)
        if not p_layout.is_absolute():
            if (Path.cwd() / p_layout).exists():
                p_layout = Path.cwd() / p_layout
            elif hasattr(self.mp, "_caller_dir") and (Path(self.mp._caller_dir) / p_layout.name).exists():
                p_layout = Path(self.mp._caller_dir) / p_layout.name
            elif (Path.cwd() / p_layout.name).exists():
                p_layout = Path.cwd() / p_layout.name
            elif hasattr(self.mp, "_caller_dir") and (Path(self.mp._caller_dir) / p_layout).exists():
                p_layout = Path(self.mp._caller_dir) / p_layout

        if not p_layout.exists():
            caller_dir = getattr(self.mp, "_caller_dir", Path.cwd())
            print(
                f"\n[MANTIS ERROR] Plot layout file '{layout}' not found.\n"
                f"Searched in:\n"
                f"  • Current working directory: {Path.cwd()}\n"
                f"  • Model directory:           {caller_dir}\n"
                f"Plot summary rendering skipped.\n",
                file=sys.stderr,
            )
            return None

        if self.df is not None:
            pdf_out = outfile or f"{self.mp.plot_name}.pdf"
            try:
                plt_desc = pdn.load_layout_from_file(self.df, str(p_layout))
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

        # Derive experiment name automatically from caller filename stem
        if experiment is not None:
            self.experiment = experiment
        elif caller_frame is not None and caller_frame.filename not in ("<stdin>", "") and not Path(caller_frame.filename).name.startswith("pytest"):
            self.experiment = Path(caller_frame.filename).stem
        elif sys.argv and sys.argv[0] and not Path(sys.argv[0]).name.startswith("pytest"):
            self.experiment = Path(sys.argv[0]).stem
        else:
            self.experiment = "diagenetic_run"

        # 2. Store original user inputs for clear diagnostics
        self._orig_species = species
        self._orig_reactions = reactions
        self._orig_constants = constants
        self._orig_equations = equations
        self._orig_limiters = limiters
        self._orig_layout = layout

        # 3. Resolve filenames relative to cwd or base_dir (wherever the model resides)
        self.species_file = self._resolve(species)
        rxn_fallbacks = [f"{self.experiment}_reactions.py"] if str(reactions) == "reactions.py" else None
        self.reactions_file = self._resolve(reactions, fallbacks=rxn_fallbacks)
        self.constants_file = self._resolve(constants)
        self.equations_file = self._resolve(equations) if equations else None
        self.limiters_file = self._resolve(limiters) if limiters else None
        self.layout_file = self._resolve(layout)

        # 4. If running as a script (not querying CLI help/options or in pytest), validate files immediately
        is_cli_query = any(arg in sys.argv for arg in ("-h", "--help", "--solver", "--video", "--run"))
        in_pytest = "pytest" in sys.modules or (sys.argv and Path(sys.argv[0]).name.startswith("pytest"))
        if not is_cli_query and not in_pytest:
            self.validate_simulation_files()

        self.ureg = pint.UnitRegistry()
        self.Q_ = self.ureg.Quantity

        # Master parameter container
        self.mp = data_container()
        self.p_dict: Dict[str, Any] = {}

        # Default parameters
        self._init_defaults()

    def _exit_with_missing_file_error(self, filename: Union[str, Path], file_description: str) -> None:
        """Prints a user-friendly error message and exits when a simulation file cannot be found."""
        print(f"\n[MANTIS ERROR] Could not find {file_description}: '{filename}'", file=sys.stderr)
        print("Searched locations:", file=sys.stderr)
        print(f"  • Current working directory: {Path.cwd()}", file=sys.stderr)
        if self.base_dir.resolve() != Path.cwd().resolve():
            print(f"  • Model directory:           {self.base_dir}", file=sys.stderr)
        print(
            f"\nPlease ensure that '{filename}' is located in the current working directory "
            f"or the directory where the model script resides ({self.base_dir}).\n",
            file=sys.stderr,
        )
        sys.exit(1)

    def validate_simulation_files(self) -> None:
        """Validates that all required files describing the simulation exist in cwd or model directory."""
        if not self.species_file.exists():
            self._exit_with_missing_file_error(self._orig_species, "species definitions file")
        if not self.reactions_file.exists():
            self._exit_with_missing_file_error(self._orig_reactions, "declarative reactions file")
        if not self.constants_file.exists():
            self._exit_with_missing_file_error(self._orig_constants, "reaction constants file")
        if self._orig_limiters is not None and self.limiters_file is not None and not self.limiters_file.exists():
            self._exit_with_missing_file_error(self._orig_limiters, "limiters file")

    def _resolve(self, path_like: Union[str, Path], fallbacks: Optional[List[str]] = None) -> Path:
        p = Path(path_like)
        # Check absolute path
        if p.is_absolute() and p.exists():
            return p.resolve()

        # Check in current working directory
        if (Path.cwd() / p).exists():
            return (Path.cwd() / p).resolve()

        # Check in model directory (base_dir)
        if (self.base_dir / p).exists():
            return (self.base_dir / p).resolve()

        # Check fallbacks if provided
        if fallbacks:
            for fb in fallbacks:
                fb_p = Path(fb)
                if (Path.cwd() / fb_p).exists():
                    return (Path.cwd() / fb_p).resolve()
                if (self.base_dir / fb_p).exists():
                    return (self.base_dir / fb_p).resolve()

        # Fallback to model directory or cwd
        if self.base_dir != Path.cwd():
            return (self.base_dir / p).resolve()
        return (Path.cwd() / p).resolve()

    def _init_defaults(self) -> None:
        """Initializes default numerical solver, video output, and experiment metadata parameters."""
        Q_ = self.Q_
        self.p_dict.update({
            # Experiment metadata
            "experiment": self.experiment,
            "plot_name": self.experiment,
            "state_data": None,
            "layout_file": str(self.layout_file),
            "start_time": 0,

            # Video recording options
            "video_dt": None,
            "video_dpi": 120,
            "video_queue_size": 100,
            "video_resume_threshold": 50,
            "video_workers": None,

            # Solver parameters
            "solver": "monolithic",
            "t_end": Q_("1 kyr").to("seconds").magnitude,
            "dt_min": Q_("1 minute").to("seconds").magnitude,
            "dt_init": Q_("1 day").to("seconds").magnitude,
            "dt_max": Q_("1 year").to("seconds").magnitude,
            "tolerance": 1e-12,
            "dt_tolerance": 1e-5,

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

        Any parameter passed with units (as a string or Pint Quantity) is automatically
        detected and converted to standard SI simulation units based on its physical dimensionality:
        - [time] -> seconds (e.g. t_end='2 kyr', dt_max='1 year', dt_min='1 second')
        - [length] -> meters (e.g. max_depth='0.5 m', initial_spacing='0.01 mm')
        - [length] / [time] -> m/s (e.g. w='0.2 cm/yr')
        - [length]^2 / [time] -> m^2/s (e.g. BT0='4 cm^2/year')
        - 1 / [time] -> 1/s (e.g. BI0='0.1 /year')
        - [substance] / [length]^3 -> mol/m^3 (e.g. concentration='0.001 umol/cm^3')
        - [substance] / ([length]^2 * [time]) -> mol/(m^2*s) (e.g. flux='1.2 mol/(m^2*yr)')

        Nested structures (dicts, lists, tuples) are handled recursively.
        """
        Q_ = self.Q_

        def _convert_value(val: Any) -> Any:
            if isinstance(val, (list, tuple)):
                converted = [_convert_value(item) for item in val]
                return tuple(converted) if isinstance(val, tuple) else converted
            if isinstance(val, dict):
                return {k: _convert_value(v) for k, v in val.items()}
            if isinstance(val, str):
                try:
                    q = Q_(val)
                    if hasattr(q, "dimensionality"):
                        dim = q.dimensionality
                        if dim == Q_("1 second").dimensionality:
                            return q.to("seconds").magnitude
                        elif dim == Q_("1 meter").dimensionality:
                            return q.to("meter").magnitude
                        elif dim == Q_("1 m/s").dimensionality:
                            return q.to("m/s").magnitude
                        elif dim == Q_("1 m^2/s").dimensionality:
                            return q.to("m^2/second").magnitude
                        elif dim == Q_("1 / second").dimensionality:
                            return q.to("1/second").magnitude
                        elif dim == Q_("1 mol/m^3").dimensionality:
                            return q.to("mol/m^3").magnitude
                        elif dim == Q_("1 mol/(m^2 * second)").dimensionality:
                            return q.to("mol/(m^2 * second)").magnitude
                        elif q.dimensionless and not isinstance(q.magnitude, str):
                            return q.magnitude
                except Exception:
                    pass
            elif hasattr(val, "dimensionality"):
                dim = val.dimensionality
                if dim == Q_("1 second").dimensionality:
                    return val.to("seconds").magnitude
                elif dim == Q_("1 meter").dimensionality:
                    return val.to("meter").magnitude
                elif dim == Q_("1 m/s").dimensionality:
                    return val.to("m/s").magnitude
                elif dim == Q_("1 m^2/s").dimensionality:
                    return val.to("m^2/second").magnitude
                elif dim == Q_("1 / second").dimensionality:
                    return val.to("1/second").magnitude
                elif dim == Q_("1 mol/m^3").dimensionality:
                    return val.to("mol/m^3").magnitude
                elif dim == Q_("1 mol/(m^2 * second)").dimensionality:
                    return val.to("mol/(m^2 * second)").magnitude
                elif val.dimensionless:
                    return val.to_base_units().magnitude
            return val

        for key, val in kwargs.items():
            if key in ("video", "video_dt"):
                self.p_dict["video_dt"] = None if str(val).lower() in ("none", "null", "false") else val
            elif key == "pH":
                self.p_dict["pH"] = float(val)
            elif key in ("steady_state_tolerance", "dt_tolerance"):
                f_val = float(val)
                self.p_dict["steady_state_tolerance"] = f_val
                self.p_dict["dt_tolerance"] = f_val
            elif key == "check_reactant_coexistence":
                self.p_dict["check_reactant_coexistence"] = bool(val)
            elif key == "bioturbation" and isinstance(val, dict):
                c_val = _convert_value(val)
                if "BT0" in c_val:
                    self.p_dict["BT0"] = float(c_val["BT0"])
                if "depth" in c_val:
                    self.p_dict["BT_depth"] = float(c_val["depth"])
                if "attenuation" in c_val:
                    self.p_dict["BT_attenuation"] = float(c_val["attenuation"])
                self.p_dict["bioturbation"] = c_val
            elif key == "bioirrigation" and isinstance(val, dict):
                c_val = _convert_value(val)
                if "BI0" in c_val:
                    self.p_dict["BI0"] = float(c_val["BI0"])
                if "depth" in c_val:
                    self.p_dict["BI_depth"] = float(c_val["depth"])
                self.p_dict["bioirrigation"] = c_val
            else:
                self.p_dict[key] = _convert_value(val)

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

        cur_dt_tol = self.p_dict.get("dt_tolerance", 1e-5)
        cur_vqueue = self.p_dict.get("video_queue_size", 100)
        cur_vresume = self.p_dict.get("video_resume_threshold", 50)
        cur_vworkers = self.p_dict.get("video_workers", None)

        solver_section = [
            ("--solver <backend>", cur_solver, "Backend solver choice ('monolithic' or 'coupled')"),
            ("--solver-tol <float>", cur_tol, "Newton convergence tolerance (e.g. 1e-3)"),
            ("--numerical-jacobian", cur_jac, "Use finite-difference Jacobian instead of SymPy analytical"),
            ("--max-newton-iters <int>", cur_max_iters, "Maximum Newton iterations allowed per timestep"),
            ("--newton-norm <choice>", cur_norm, "Residual norm formulation ('linf' or 'wrms')"),
            ("--newton-target-optimal <int>", cur_opt_iters, "Target Newton iterations for timestep expansion"),
            ("--newton-max-acceptable <int>", cur_max_acc_iters, "Max acceptable iterations before timestep reduction"),
            ("--dt-tolerance <float>", cur_dt_tol, "Adaptive timestep convergence tolerance (e.g. 1e-5)"),
            ("--force-regen", False, "Force regeneration of equation system and analytical Jacobian"),
        ]

        video_section = [
            ("--video <interval>", cur_video, "Live video recording frame interval (e.g. '10 year', 'none')"),
            ("--video-dpi <int>", cur_dpi, "Resolution (DPI) of recorded video frames"),
            ("--video-queue-size <int>", cur_vqueue, "Maximum video frame render queue size"),
            ("--video-resume-threshold <int>", cur_vresume, "Video frame resume queue threshold"),
            ("--video-workers <int>", cur_vworkers, "Dedicated background worker count for video rendering"),
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
                f"MANTIS Diagenetic Model: {self.experiment}\n\n"
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
        parser.add_argument(
            "--dt-tolerance",
            type=float,
            metavar="TOL",
            help=f"Adaptive timestep tolerance (default: {cur_dt_tol})",
        )
        parser.add_argument(
            "--force-regen",
            "--rebuild-equations",
            action="store_true",
            help="Force regeneration of equations and analytical Jacobian",
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
        parser.add_argument(
            "--video-queue-size",
            type=int,
            metavar="SIZE",
            help=f"Max video queue size (default: {cur_vqueue})",
        )
        parser.add_argument(
            "--video-resume-threshold",
            type=int,
            metavar="SIZE",
            help=f"Video queue resume threshold (default: {cur_vresume})",
        )
        parser.add_argument(
            "--video-workers",
            type=int,
            metavar="COUNT",
            help="Worker count for video encoding",
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
        if parsed.dt_tolerance is not None:
            self.configure(dt_tolerance=parsed.dt_tolerance)
        if parsed.force_regen:
            self.p_dict["force_regen"] = True

        # Apply video overrides
        if parsed.video is not None:
            self.p_dict["video_dt"] = None if parsed.video.lower() in ("none", "null", "false") else parsed.video
        if parsed.video_dpi is not None:
            self.p_dict["video_dpi"] = int(parsed.video_dpi)
        if parsed.video_queue_size is not None:
            self.p_dict["video_queue_size"] = int(parsed.video_queue_size)
        if parsed.video_resume_threshold is not None:
            self.p_dict["video_resume_threshold"] = int(parsed.video_resume_threshold)
        if parsed.video_workers is not None:
            self.p_dict["video_workers"] = int(parsed.video_workers)

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
        if not self.species_file.exists():
            self._exit_with_missing_file_error(self._orig_species, "species definitions file")
        spec = importlib.util.spec_from_file_location("species_mod", self.species_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load species file: {self.species_file}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "species") or not isinstance(mod.species, dict):
            raise ValueError(f"Species file {self.species_file} must define a 'species' dictionary.")
        self._species_mod = mod
        return mod.species

    def setup(self, plot_queue=None) -> Tuple[data_container, data_container, data_container, List[str], np.ndarray, data_container, Any]:
        """Assembles grid, species variables, transport coefficients, and compiles reaction equations."""
        self.validate_simulation_files()
        self.mp.update(self.p_dict)
        self.mp._caller_dir = str(self.base_dir)

        # Fallback defaults for geometry if not specified by user configuration
        if not hasattr(self.mp, "max_depth"):
            self.mp.max_depth = 2.0
        if not hasattr(self.mp, "display_length"):
            self.mp.display_length = self.mp.max_depth
        if not hasattr(self.mp, "initial_spacing"):
            self.mp.initial_spacing = 0.0001
        if not hasattr(self.mp, "reaction_zone_spacing"):
            self.mp.reaction_zone_spacing = 0.0001
        if not hasattr(self.mp, "max_spacing"):
            self.mp.max_spacing = 0.1
        if not hasattr(self.mp, "reaction_zone"):
            self.mp.reaction_zone = (0.0, 0.1)

        # Fallback defaults for environmental and transport parameters
        if not hasattr(self.mp, "temp"):
            self.mp.temp = [10.0, 10.1]
        if not hasattr(self.mp, "w"):
            self.mp.w = self.Q_("0.2 cm/yr").to("m/s").magnitude
        if not hasattr(self.mp, "advection"):
            self.mp.advection = 0.0
        if not hasattr(self.mp, "phi"):
            self.mp.phi = 0.8
        if not hasattr(self.mp, "BT0"):
            self.mp.BT0 = 0.0
        if not hasattr(self.mp, "BT_depth"):
            self.mp.BT_depth = self.Q_("7.6 cm").to("meter").magnitude
        if not hasattr(self.mp, "BT_attenuation"):
            self.mp.BT_attenuation = self.Q_("2 cm").to("meter").magnitude
        if not hasattr(self.mp, "BI0"):
            self.mp.BI0 = 0.0
        if not hasattr(self.mp, "BI_depth"):
            self.mp.BI_depth = 0.0
        if not hasattr(self.mp, "isotopes"):
            self.mp.isotopes = True
        if not hasattr(self.mp, "VCDT"):
            self.mp.VCDT = 0.044162589

        # 1. Parse species definitions
        species_defs = self._load_species_definitions()

        # 2. Extract pH from species definitions (e.g. Hplus or any species defining 'pH' under 'bc')
        if "pH" not in self.p_dict:
            for sp_k, s_meta in species_defs.items():
                s_bc = s_meta.get("bc")
                if isinstance(s_bc, dict) and "pH" in s_bc:
                    self.mp.pH = float(s_bc["pH"])
                    break

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

        # Extract reaction_parameters or fractionation_factors from reactions module or file
        rxn_params = getattr(eq_mod, "reaction_parameters", None) or getattr(eq_mod, "fractionation_factors", None)
        if rxn_params is None and hasattr(self, "reactions_file") and self.reactions_file.exists():
            spec_rxn = importlib.util.spec_from_file_location("rxn_param_mod", self.reactions_file)
            if spec_rxn and spec_rxn.loader:
                mod_rxn = importlib.util.module_from_spec(spec_rxn)
                spec_rxn.loader.exec_module(mod_rxn)
                rxn_params = getattr(mod_rxn, "reaction_parameters", None) or getattr(mod_rxn, "fractionation_factors", None)

        if isinstance(rxn_params, dict):
            for rp_k, rp_v in rxn_params.items():
                if not hasattr(self.mp, rp_k):
                    setattr(self.mp, rp_k, rp_v)

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

        # Propagate Monod half-saturation constants (K_*) from k to self.mp if not already configured
        k_items = k.items() if hasattr(k, "items") else k.__dict__.items()
        for k_name, k_val in k_items:
            if k_name.startswith("K_") and not hasattr(self.mp, k_name):
                setattr(self.mp, k_name, k_val)

        # 7. Species lists & phase classification
        species_transport: List[str] = []
        species_all: List[str] = []
        bc_map: Dict[str, Dict[str, Any]] = {}
        D_mol = data_container()
        c = data_container()

        # Porosity volume ratios
        phi_val = self.mp.phi.value if hasattr(self.mp.phi, "value") else self.mp.phi
        self.mp.fac_s = phi_val / (1.0 - phi_val)

        # Dynamic speciation & phase partition calculations from species_defs
        Hplus = 10.0 ** (-float(self.mp.pH))
        eval_ctx: Dict[str, Any] = {
            "k": k,
            "mp": self.mp,
            "np": np,
            "Hplus": Hplus,
            "pH": float(self.mp.pH),
            "fac_s": self.mp.fac_s,
        }

        for sp_name, meta in species_defs.items():
            # Check partition for mixed species (e.g. Fe2_total: dissolved, sorbed)
            if "partition" in meta and isinstance(meta["partition"], dict):
                for p_key, p_expr in meta["partition"].items():
                    val = float(eval(p_expr, eval_ctx))
                    setattr(self.mp, f"{sp_name}_{p_key}", val)
                    eval_ctx[f"{sp_name}_{p_key}"] = val
                    # Backwards compatibility alias for Fe2_total.dissolved -> Fe2_diss
                    if sp_name == "Fe2_total" and p_key == "dissolved":
                        self.mp.Fe2_diss = val
                        eval_ctx["Fe2_diss"] = val
                    elif sp_name == "Fe2_total" and p_key == "sorbed":
                        self.mp.Fe2_sorb = val * self.mp.fac_s
                        eval_ctx["Fe2_sorb"] = self.mp.Fe2_sorb

            # Check fraction for equilibrium speciation (e.g. HS: Ka1 / (Hplus + Ka1))
            if "fraction" in meta:
                frac_ctx = dict(eval_ctx)
                if "constants" in meta and isinstance(meta["constants"], dict):
                    for c_name, c_val in meta["constants"].items():
                        frac_ctx[c_name] = float(c_val)
                        if c_name.startswith("pK"):
                            base_c = c_name[1:]
                            frac_ctx[base_c] = 10.0 ** (-float(c_val))
                f_val = float(eval(meta["fraction"], frac_ctx))
                setattr(self.mp, f"{sp_name.lower()}_frac", f_val)
                eval_ctx[f"{sp_name.lower()}_frac"] = f_val

        # Ensure fallback defaults if not specified in species_defs
        if not hasattr(self.mp, "Fe2_diss"):
            self.mp.Fe2_diss = 1.0 / (1.0 + k.Fe2_p_eq) if hasattr(k, "Fe2_p_eq") else 1.0
        if not hasattr(self.mp, "Fe2_sorb"):
            self.mp.Fe2_sorb = (k.Fe2_p_eq * self.mp.fac_s * self.mp.Fe2_diss) if hasattr(k, "Fe2_p_eq") else 0.0
        if not hasattr(self.mp, "hs_frac"):
            Ka1 = 10.0 ** (-7.0)
            self.mp.hs_frac = Ka1 / (Hplus + Ka1)
        if not hasattr(self.mp, "h2s_frac"):
            Ka1 = 10.0 ** (-7.0)
            self.mp.h2s_frac = Hplus / (Hplus + Ka1)

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
        # Import diff_lib module for diff_coeff functions
        import mantis.diff_lib as diff_lib_mod

        for sp_name, meta in species_defs.items():
            diff_spec = meta.get("diffusion")
            fn_name = meta.get("diff_coeff_function")

            # Check if this species inherits diffusion from another species
            if isinstance(diff_spec, str):
                if hasattr(D_mol, diff_spec):
                    setattr(D_mol, sp_name, getattr(D_mol, diff_spec))
                continue

            # Determine function name from diff_coeff_function or dict 'type'
            if fn_name is None and isinstance(diff_spec, dict):
                fn_name = diff_spec.get("type", "diff_coeff")

            if fn_name is not None:
                # 1. Look up function in species.py module first, then diff_lib
                diff_fn = None
                if hasattr(self, "_species_mod") and hasattr(self._species_mod, fn_name):
                    diff_fn = getattr(self._species_mod, fn_name)
                elif hasattr(diff_lib_mod, fn_name):
                    diff_fn = getattr(diff_lib_mod, fn_name)

                if diff_fn is not None:
                    sig = inspect.signature(diff_fn)
                    kwargs = {}
                    if "T" in sig.parameters:
                        kwargs["T"] = T_profile
                    if "T_profile" in sig.parameters:
                        kwargs["T_profile"] = T_profile
                    if "temp" in sig.parameters:
                        kwargs["temp"] = T_profile
                    if "phi" in sig.parameters:
                        kwargs["phi"] = self.mp.phi
                    if isinstance(diff_spec, dict):
                        if "m0" in sig.parameters and "m0" in diff_spec:
                            kwargs["m0"] = diff_spec["m0"]
                        if "m1" in sig.parameters and "m1" in diff_spec:
                            kwargs["m1"] = diff_spec["m1"]
                    try:
                        d_val = diff_fn(**kwargs)
                        setattr(D_mol, sp_name, d_val)
                    except Exception as e:
                        raise RuntimeError(f"Failed to compute diffusion for '{sp_name}' with {fn_name}(): {e}")
                else:
                    raise AttributeError(f"Diffusion function '{fn_name}' for species '{sp_name}' not found in species.py or diff_lib.")

        # Lumped mixed-phase diffusion scaling (porewater retardation)
        for sp_name, meta in species_defs.items():
            if meta.get("phase") == "mixed" and hasattr(D_mol, sp_name):
                diss_frac = 1.0
                part = meta.get("partition", {})
                if "dissolved" in part:
                    try:
                        diss_frac = float(eval(part["dissolved"], eval_ctx))
                    except Exception:
                        diss_frac = getattr(self.mp, f"{sp_name}_dissolved", getattr(self.mp, "Fe2_diss", 1.0))
                else:
                    diss_frac = getattr(self.mp, f"{sp_name}_dissolved", getattr(self.mp, "Fe2_diss", 1.0))
                setattr(D_mol, sp_name, getattr(D_mol, sp_name) * diss_frac)

        # Bioturbation & Bio-irrigation profiles
        D_mol.D_irr = compute_bio_irrigation_alpha(z, self.mp.BI0, self.mp.BI_depth)
        D_mol.D_bio = compute_sigmoidal_db(z, self.mp.BT0, self.mp.BT_depth, self.mp.BT_attenuation)

        # 9. Isotope boundary conditions companion setup (generalized across all species)
        if self.mp.isotopes:
            for sp_name, meta in species_defs.items():
                iso_target = meta.get("isotope")
                if not iso_target or iso_target not in species_defs:
                    continue
                # Determine delta value: from bc dict, or mp.<sp>_d, or 0.0 default
                delta_val = None
                sp_bc = meta.get("bc", {})
                if isinstance(sp_bc, dict):
                    if "delta" in sp_bc:
                        delta_val = float(sp_bc["delta"])
                if delta_val is None:
                    delta_attr = f"{sp_name}_d"
                    delta_val = float(getattr(self.mp, delta_attr, 0.0))

                iso_meta = species_defs[iso_target]
                iso_phase = iso_meta.get("phase", meta.get("phase", "liquid"))
                iso_bc_type = "particulate" if iso_phase == "solid" else "dissolved"

                if sp_name in bc_map and bc_map[sp_name]["top"] > 0.0:
                    top_mass_32 = get_l_mass(bc_map[sp_name]["top"], delta_val, self.mp.VCDT)
                    setattr(self.mp, f"bc_{iso_target}", top_mass_32)
                    bc_map[iso_target] = {"top": top_mass_32, "type": iso_bc_type}
                else:
                    setattr(self.mp, f"bc_{iso_target}", 0.0)
                    bc_map[iso_target] = {"top": 0.0, "type": iso_bc_type}

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
            if not self.layout_file.exists():
                self._exit_with_missing_file_error(self._orig_layout, "plot layout file (required for video recording)")
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

        # Run post-simulation reactant coexistence diagnostic by default
        if getattr(self.mp, "check_reactant_coexistence", True):
            try:
                from .diff_lib import check_reactant_coexistence
                check_reactant_coexistence(
                    mesh_or_z=self._mesh if self._mesh is not None else self.mp.z,
                    c=self._c,
                    k=self._k,
                    D_mol=self._D_mol,
                    verbose=True,
                )
            except Exception as e:
                print(f"[Warning] Reactant coexistence diagnostic failed: {e}")

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
