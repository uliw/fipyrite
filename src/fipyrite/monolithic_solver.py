"""Monolithic Block Newton-Raphson Solver for FiPyrite.

Solves the coupled non-steady-state reactive-transport system as a monolithic
nonlinear system using a Block Tridiagonal Matrix Algorithm (Block TDMA)
accelerated with Numba.

Eliminates decoupled Picard/Gauss-Seidel cross-species lag oscillations by
solving all chemical species simultaneously with a fully coupled chemical Jacobian.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .diff_lib import (
    ArrayProxy,
    VariableArray,
    data_container,
    get_time_units,
    save_data,
    save_data_async,
    save_state,
)
from .live_plot_lib import parse_time_to_seconds, save_final_pdf, write_to_queue_async
from .solvers_numba import (
    compute_species_residual_linf,
    compute_species_residual_wrms,
    solve_block_tridiagonal_thomas,
)


@dataclass
class AdaptiveDT:
    """Advanced PID-controlled adaptive time stepping.

    Uses standard H211B controller logic for smooth step size adaptation.
    """

    dt_min: float
    dt_max: float
    dt_initial: float
    dt_cfl_factor: float = 0.8
    growth_factor: float = 1.2
    cut_factor: float = 0.5
    pid: bool = True
    kP: float = 0.075
    kI: float = 0.175
    kD: float = 0.01

    enable_failure_ceiling: bool = False
    failure_ceiling_factor: float = 0.7
    failure_hold_steps: int = 10
    ceiling_growth_factor: float = 1.05
    aggressive_cut_factor: float = 0.25

    enable_mttf_governor: bool = False
    mttf_x1: float = 0.60
    mttf_x2: float = 0.35
    mttf_x3: float = 0.15
    mttf_y1: float = 0.02
    mttf_y2: float = 0.05
    mttf_y3: float = 0.10

    steps_since_failure: int = 0
    _base_growth_factor: float = 1.2

    _dt: float = 0.0
    _err_prev: float | None = None
    _dt_prev: float = 0.0
    _dt_ceiling: float | None = None
    _failed_dt: float | None = None
    _steps_at_ceiling: int = 0

    def __post_init__(self) -> None:
        self._dt = max(self.dt_min, min(self.dt_initial, self.dt_max))
        self._dt_prev = self._dt
        self._base_growth_factor = self.growth_factor
        if self.enable_failure_ceiling:
            self._dt_ceiling = self.dt_max
            self._failed_dt = self.dt_max
            self._steps_at_ceiling = 0

    @property
    def dt(self) -> float:
        """Return current time step."""
        return self._dt

    def register_failure(self, failed_dt: float, cut_factor: float | None = None) -> float:
        """Record step failure and set a temporary ceiling or adapt MTTF growth."""
        factor = self.cut_factor if cut_factor is None else cut_factor
        if self.enable_mttf_governor:
            excess = max(self.growth_factor - 1.0, 0.0)
            if self.steps_since_failure < 5:
                excess *= (1.0 - self.mttf_x1)
                factor = min(factor, 0.70)
            elif self.steps_since_failure < 10:
                excess *= (1.0 - self.mttf_x2)
                factor = min(factor, 0.75)
            elif self.steps_since_failure < 20:
                excess *= (1.0 - self.mttf_x3)
                factor = min(factor, 0.80)
            self.growth_factor = 1.0 + max(excess, 0.02)
            self.steps_since_failure = 0

        self._dt = max(failed_dt * factor, self.dt_min)
        if self.enable_failure_ceiling:
            self._failed_dt = failed_dt
            ceiling_factor = min(self.failure_ceiling_factor, factor * 1.5) if cut_factor is not None else self.failure_ceiling_factor
            self._dt_ceiling = max(self.dt_min, failed_dt * ceiling_factor)
            self._steps_at_ceiling = 0
        return self._dt

    def record_success(self) -> None:
        """Record a successful step and adapt dynamic growth factor if MTTF governor is active."""
        self.steps_since_failure += 1
        if self.enable_mttf_governor:
            excess = max(self.growth_factor - 1.0, 0.0)
            base_excess = max(self._base_growth_factor - 1.0, 0.01)
            if self.steps_since_failure >= 40:
                excess = min(base_excess, excess + self.mttf_y3 * base_excess)
            elif self.steps_since_failure >= 30:
                excess = min(base_excess, excess + self.mttf_y2 * base_excess)
            elif self.steps_since_failure >= 20:
                excess = min(base_excess, excess + self.mttf_y1 * base_excess)
            self.growth_factor = 1.0 + excess

    def cfl_limit(self, dx: float, vel: float, D: float) -> float:
        """Compute global CFL estimate for advection-diffusion."""
        adv_limit = math.inf if vel == 0 else dx / abs(vel)
        dif_limit = math.inf if D == 0 else dx * dx / (2 * D)
        return self.dt_cfl_factor * min(adv_limit, dif_limit)

    def get_effective_max(self, _log: Optional[Callable[[str], None]] = None) -> float:
        """Return effective maximum dt considering the dynamic failure ceiling and hold period."""
        if self.enable_failure_ceiling and self._dt_ceiling is not None:
            if self._dt >= self._dt_ceiling * 0.98:
                self._steps_at_ceiling += 1

            if self._steps_at_ceiling >= self.failure_hold_steps:
                if self.ceiling_growth_factor > 1.0 and self._dt_ceiling < self.dt_max:
                    old_ceiling = self._dt_ceiling
                    self._dt_ceiling = min(
                        self._dt_ceiling * self.ceiling_growth_factor,
                        self.dt_max,
                    )
                    self._steps_at_ceiling = 0
                    if _log is not None:
                        _log(
                            f"  Ceiling hold of {self.failure_hold_steps} steps completed: raising dt ceiling from {get_time_units(old_ceiling):.2f~P} to {get_time_units(self._dt_ceiling):.2f~P}."
                        )
                return min(self._dt_ceiling, self.dt_max)
            else:
                return min(self._dt_ceiling, self.dt_max)
        return self.dt_max

    def update(
        self,
        error_metric: float,
        dt_cfl: float | None = None,
        step_success: bool = True,
        target_error: float = 1e-4,
        cut_factor: float | None = None,
    ) -> float:
        """Compute the next dt based on solver performance and change magnitude."""
        if not step_success:
            factor = self.cut_factor if cut_factor is None else cut_factor
            self._dt = max(self._dt * factor, self.dt_min)
            return self._dt

        effective_max = self.get_effective_max()

        if self.pid:
            if self._err_prev is None:
                self._err_prev = target_error

            err = max(error_metric, 1e-25)
            err_prev = max(self._err_prev, 1e-25)
            err_ref = target_error

            factor = (
                (err_prev / err) ** self.kP
                * (err_ref / err) ** self.kI
                * (err_ref / err_prev) ** self.kD
            )

            factor = max(self.cut_factor, min(self.growth_factor, factor))
            self._dt = max(self.dt_min, min(self._dt * factor, effective_max))
        else:
            if error_metric < target_error:
                self._dt = min(self._dt * self.growth_factor, effective_max)
            else:
                self._dt = max(self._dt * self.cut_factor, self.dt_min)

        if dt_cfl is not None:
            self._dt = min(self._dt, dt_cfl)

        self._err_prev = error_metric
        return self._dt


def build_native_1d_transport_stencil(
    z: np.ndarray,
    dx: np.ndarray,
    phi: Any,
    D_cell: np.ndarray,
    w: float,
    bc_props: Dict[str, Any],
    D_irr: Optional[np.ndarray] = None,
    solid_scheme: str = "powerlaw",
) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs the exact 1D tridiagonal finite-volume transport stencil in pure NumPy.

    Returns:
        ab: shape (3, N) banded matrix (row 0: upper, row 1: diag, row 2: lower)
        b_transport: shape (N,) boundary flux vector
    """
    N = len(z)
    is_dissolved = (bc_props.get("type", "dissolved") == "dissolved")
    eff_phi_val = np.asarray(phi.value if hasattr(phi, "value") else phi)
    eff_phi = eff_phi_val if is_dissolved else (1.0 - eff_phi_val)
    if eff_phi.ndim == 0:
        eff_phi_arr = np.full(N, float(eff_phi), dtype=np.float64)
    else:
        eff_phi_arr = np.asarray(eff_phi, dtype=np.float64)
    eff_phi_face = (eff_phi_arr[:-1] + eff_phi_arr[1:]) / 2.0

    d_centers = z[1:] - z[:-1]
    D_cell_arr = np.asarray(D_cell)
    if D_cell_arr.ndim == 0:
        D_cell_arr = np.full(N, float(D_cell_arr), dtype=np.float64)
    # Face diffusion: distance-weighted harmonic mean
    D_face = 2.0 / (1.0 / np.maximum(D_cell_arr[:-1], 1e-30) + 1.0 / np.maximum(D_cell_arr[1:], 1e-30))
    diff_cond = (eff_phi_face * D_face) / d_centers

    # Face convection
    F_face = eff_phi_face * w
    if solid_scheme == "upwind" and not is_dissolved:
        a_E = diff_cond + np.maximum(F_face, 0.0)
        a_W = diff_cond + np.maximum(-F_face, 0.0)
    else:
        Pe = F_face / np.maximum(diff_cond, 1e-30)
        A_pe = np.maximum(0.0, (1.0 - 0.1 * np.abs(Pe))**5)
        a_E = diff_cond * A_pe + np.maximum(F_face, 0.0)
        a_W = diff_cond * A_pe + np.maximum(-F_face, 0.0)

    ab = np.zeros((3, N), dtype=np.float64)
    b_transport = np.zeros(N, dtype=np.float64)

    # Upper diagonal: ab[0, 1:] -> coeff of C_{i+1} in equation i
    ab[0, 1:] = -a_W
    # Lower diagonal: ab[2, :-1] -> coeff of C_{i-1} in equation i
    ab[2, :-1] = -a_E

    # Main diagonal interior:
    ab[1, 0] = a_E[0]
    ab[1, 1:-1] = a_W[:-1] + a_E[1:]
    ab[1, -1] = a_W[-1]

    # Left boundary (z=0):
    top_val = bc_props.get("top", 0.0)
    d_top = z[0]
    phi_top = eff_phi_arr[0]

    if is_dissolved:
        D_top = (phi_top * D_cell_arr[0]) / d_top
        F_top = phi_top * w
        Pe_top = F_top / np.maximum(D_top, 1e-30)
        A_top = np.maximum(0.0, (1.0 - 0.1 * np.abs(Pe_top))**5)
        a_in_diag = D_top * A_top + np.maximum(-F_top, 0.0)
        a_in_rhs = D_top * A_top + np.maximum(F_top, 0.0)
        ab[1, 0] += a_in_diag
        b_transport[0] += a_in_rhs * top_val
    else:
        b_transport[0] += top_val

    # Right boundary (z=L): Neumann zero-gradient (advection carries mass out)
    ab[1, -1] += eff_phi_arr[-1] * w

    # Bio-irrigation (non-local exchange for dissolved species)
    if is_dissolved and D_irr is not None:
        irr_arr = np.asarray(D_irr)
        if np.any(irr_arr > 0):
            irr_coeff = eff_phi_arr * irr_arr * dx
            ab[1, :] += irr_coeff
            b_transport += irr_coeff * top_val

    return ab, b_transport


class GovernorStats:
    """Accumulates solver and governor events and reports summary breakdown."""

    def __init__(self):
        self.total_sweeps = 0
        self.total_steps = 0
        self.single_sweep_sweeps = 0
        self.multi_sweep_sweeps = 0
        self.near_converged_sweeps = 0
        self.graceful_sweeps = 0
        self.porewater_capped_sweeps = 0
        self.isotope_capped_sweeps = 0
        self.rejected_sweeps = 0
        self.porewater_details: Dict[str, int] = {}
        self.rejected_reasons: Dict[str, int] = {}

    def record_step(
        self,
        sweeps: int,
        outcome: str,
        porewater_capped: bool = False,
        limiting_species: Optional[str] = None,
        isotope_capped: bool = False,
    ) -> None:
        self.total_steps += 1
        self.total_sweeps += sweeps

        if porewater_capped:
            self.porewater_capped_sweeps += sweeps
            if limiting_species:
                self.porewater_details[limiting_species] = (
                    self.porewater_details.get(limiting_species, 0) + 1
                )
        elif isotope_capped:
            self.isotope_capped_sweeps += sweeps
        elif outcome == "single_sweep":
            self.single_sweep_sweeps += sweeps
        elif outcome == "graceful":
            self.graceful_sweeps += sweeps
        elif outcome == "near_converged":
            self.near_converged_sweeps += sweeps
        else:
            self.multi_sweep_sweeps += sweeps

    def record_rejection(self, sweeps: int, reason: str = "failed") -> None:
        self.total_sweeps += sweeps
        self.rejected_sweeps += sweeps
        short_reason = reason.split("\n")[0][:60]
        self.rejected_reasons[short_reason] = (
            self.rejected_reasons.get(short_reason, 0) + 1
        )

    def format_summary(self) -> str:
        tot = max(self.total_sweeps, 1)
        lines = [
            f"Governor & Sweep Summary (Total Sweeps: {self.total_sweeps}, Steps: {self.total_steps}):"
        ]
        if self.single_sweep_sweeps > 0 or self.total_sweeps == 0:
            lines.append(
                f"  {self.single_sweep_sweeps / tot * 100:5.1f}% ({self.single_sweep_sweeps:d} sweeps): accepted with a single sweep"
            )
        if self.multi_sweep_sweeps > 0:
            lines.append(
                f"  {self.multi_sweep_sweeps / tot * 100:5.1f}% ({self.multi_sweep_sweeps:d} sweeps): accepted with multi-sweeps"
            )
        if self.graceful_sweeps > 0 or self.near_converged_sweeps > 0:
            grace_tot = self.graceful_sweeps + self.near_converged_sweeps
            lines.append(
                f"  {grace_tot / tot * 100:5.1f}% ({grace_tot:d} sweeps): accepted near convergence / graceful tolerance"
            )
        if self.porewater_capped_sweeps > 0:
            lim_str = ""
            if self.porewater_details:
                top_sp = max(self.porewater_details.items(), key=lambda x: x[1])[0]
                lim_str = f" (limiting: {top_sp})"
            lines.append(
                f"  {self.porewater_capped_sweeps / tot * 100:5.1f}% ({self.porewater_capped_sweeps:d} sweeps): capped because porewater depleted too fast{lim_str}"
            )
        if self.isotope_capped_sweeps > 0:
            lines.append(
                f"  {self.isotope_capped_sweeps / tot * 100:5.1f}% ({self.isotope_capped_sweeps:d} sweeps): capped by isotope limiter"
            )
        if self.rejected_sweeps > 0:
            top_reason = ""
            if self.rejected_reasons:
                top_r = max(self.rejected_reasons.items(), key=lambda x: x[1])[0]
                top_reason = f" ({top_r})"
            lines.append(
                f"  {self.rejected_sweeps / tot * 100:5.1f}% ({self.rejected_sweeps:d} sweeps): rejected / Picard convergence failure{top_reason}"
            )
        return "\n".join(lines)


class MonolithicNewtonSystem:
    """Manages the 1D monolithic block-tridiagonal ADR system."""

    def __init__(
        self,
        species_struct: List[Dict[str, Any]],
        mesh: Any,
        mp: Any,
        bc_map: Optional[Dict[str, Any]] = None,
        D_mol: Optional[Any] = None,
        z: Optional[np.ndarray] = None,
    ):
        self.species_struct = species_struct
        self.species_names = [s["name"] for s in species_struct]
        self.num_species = len(self.species_names)
        self.num_cells = getattr(mesh, "numberOfCells", len(mesh.cellVolumes))
        self.vol = np.asarray(mesh.cellVolumes)
        self.z = z if z is not None else np.asarray(mesh.cellCenters[0])

        N = self.num_cells
        S = self.num_species

        # Block tridiagonal static transport stencils:
        # A: subdiagonal (coupling to cell i-1), shape (N, S)
        # B_trans: diagonal transport part, shape (N, S)
        # C: superdiagonal (coupling to cell i+1), shape (N, S)
        # b_transport: transport RHS from BCs/irrigation, shape (N, S)
        # eff_phi_vol: effective porosity * cell volume, shape (N, S)
        self.A = np.zeros((N, S), dtype=np.float64)
        self.B_trans = np.zeros((N, S), dtype=np.float64)
        self.C = np.zeros((N, S), dtype=np.float64)
        self.b_transport = np.zeros((N, S), dtype=np.float64)
        self.eff_phi_vol = np.zeros((N, S), dtype=np.float64)

        phi_val = np.asarray(mp.phi.value if hasattr(mp.phi, "value") else mp.phi)
        solid_scheme = getattr(mp, "solid_convection_term", "powerlaw")

        for s_idx, s_obj in enumerate(species_struct):
            name = s_obj["name"]
            props = bc_map.get(name, {}) if bc_map is not None else {}

            is_diss = props.get("type", "dissolved") == "dissolved"
            eff_phi_val = phi_val if is_diss else (1.0 - phi_val)
            self.eff_phi_vol[:, s_idx] = eff_phi_val * self.vol

            # Effective diffusion coefficient
            D_eff = s_obj.get("D_total", None)
            if D_eff is None:
                D_mol_val = getattr(D_mol, name, 0.0) if D_mol is not None else 0.0
                D_bio_val = getattr(D_mol, "D_bio", 0.0) if D_mol is not None else 0.0
                D_eff = np.maximum(D_mol_val + D_bio_val, 1e-20)

            # Advection velocity
            w_val = getattr(mp, "w", 0.0)
            if is_diss:
                w_val -= getattr(mp, "advection", 0.0)

            # Bio-irrigation (dissolved only)
            D_irr_val = (
                getattr(D_mol, "D_irr", None)
                if (is_diss and D_mol is not None)
                else None
            )

            ab, b_t = build_native_1d_transport_stencil(
                z=self.z,
                dx=self.vol,
                phi=phi_val,
                D_cell=D_eff,
                w=w_val,
                bc_props=props,
                D_irr=D_irr_val,
                solid_scheme=solid_scheme,
            )

            # ab has shape (3, N):
            # row 0: superdiagonal (ab[0, i+1] couples to cell i+1)
            # row 1: main diagonal (ab[1, i] couples to cell i)
            # row 2: subdiagonal (ab[2, i-1] couples to cell i-1)
            self.B_trans[:, s_idx] = ab[1, :]
            if N > 1:
                self.C[:-1, s_idx] = ab[0, 1:]
                self.A[1:, s_idx] = ab[2, :-1]
            self.b_transport[:, s_idx] = b_t

        # Preallocated scratch buffers for Block TDMA
        self.B_work = np.zeros((N, S, S), dtype=np.float64)
        self.C_prime = np.zeros((N, S, S), dtype=np.float64)
        self.D_prime = np.zeros((N, S), dtype=np.float64)
        self.res = np.zeros((N, S), dtype=np.float64)
        self.delta_u = np.zeros((N, S), dtype=np.float64)
        self.u_curr = np.zeros((N, S), dtype=np.float64)
        self.u_old = np.zeros((N, S), dtype=np.float64)
        self.J_chem = np.zeros((N, S, S), dtype=np.float64)

    def load_state(self) -> None:
        """Loads current concentration state into u_curr and u_old."""
        for s_idx, s_obj in enumerate(self.species_struct):
            var = s_obj["var"]
            self.u_curr[:, s_idx] = np.asarray(var.value)
            self.u_old[:, s_idx] = (
                np.asarray(var.old.value)
                if hasattr(var, "old")
                else np.asarray(var.value)
            )

    def sync_to_variables(self, u_target: np.ndarray) -> None:
        """Updates CellVariables in-place with u_target."""
        for s_idx, s_obj in enumerate(self.species_struct):
            s_obj["var"].setValue(u_target[:, s_idx])

    def evaluate_transport_operator(self, u: np.ndarray) -> np.ndarray:
        """Computes T * u for all species: A * u_{i-1} + B_trans * u_i + C * u_{i+1}."""
        N, S = u.shape
        Tu = self.B_trans * u
        if N > 1:
            Tu[1:, :] += self.A[1:, :] * u[:-1, :]
            Tu[:-1, :] += self.C[:-1, :] * u[1:, :]
        return Tu


def run_non_steady_state_solver_monolithic(
    mp: Any,
    c: Any,
    species_list_full: List[str],
    species_list_partial: List[str],
    k: Any,
    diagenetic_reactions: Any,
    equilibrium_reactions: Any,
    mesh: Any,
    D_mol: Any,
    bc_map: Dict[str, Any],
    z: np.ndarray,
    plot_queue: Optional[Any] = None,
) -> Tuple[int, float]:
    """
    Solves the non-steady state ADR reactive transport system using a Monolithic
    Block Newton-Raphson backend with Numba Block TDMA.
    """
    log_file = (
        open("solver_debug.log", "w", buffering=1)
        if getattr(mp, "verbose", False)
        else None
    )

    def _log(msg: str) -> None:
        if log_file:
            log_file.write(f"[{time.strftime('%X')}] {msg}\n")
            log_file.flush()

    dt_controller = AdaptiveDT(
        dt_min=getattr(mp, "dt_min", 1.0),
        dt_max=getattr(mp, "dt_max", 31536000.0),
        dt_initial=getattr(mp, "dt_init", getattr(mp, "dt_min", 1.0)),
        growth_factor=float(getattr(mp, "dt_growth_factor", 1.25)),
        cut_factor=float(getattr(mp, "dt_cut_factor", 0.5)),
    )

    msg_info = (
        "Starting Monolithic Block Newton-Raphson Solver. "
        f"dt_init: {get_time_units(dt_controller.dt):.2f~P}"
    )
    _log(msg_info)
    print(msg_info)

    species_struct = [{"name": s, "var": getattr(c, s)} for s in species_list_partial]
    mono_system = MonolithicNewtonSystem(
        species_struct=species_struct,
        mesh=mesh,
        mp=mp,
        bc_map=bc_map,
        D_mol=D_mol,
        z=z,
    )

    N = mono_system.num_cells
    S = mono_system.num_species

    # Newton convergence settings
    max_newton_iters = int(getattr(mp, "max_newton_iters", 8))
    newton_tol = float(getattr(mp, "newton_tol", 1e-2))
    newton_norm = getattr(mp, "newton_norm", "wrms")
    target_optimal_iters = int(getattr(mp, "newton_target_optimal", 4))
    max_acceptable_iters = int(getattr(mp, "newton_max_acceptable", 6))

    # Fast NumPy proxy containers
    c_numpy = data_container({s: ArrayProxy(val.value) for s, val in c.items()})
    mp_numpy = data_container(mp)
    phi_val = mp.phi.value if hasattr(mp.phi, "value") else mp.phi
    mp_numpy.phi = ArrayProxy(phi_val)
    mp_numpy.in_solver = True

    f_res_scratch = data_container()

    # Chemical Jacobian evaluation strategy (Analytical SymPy vs Vectorized Finite Difference)
    use_numerical_jacobian = getattr(mp, "use_numerical_jacobian", False)
    jacobian_fn = None if use_numerical_jacobian else getattr(mp, "jacobian_fn", None)
    if jacobian_fn is None and not use_numerical_jacobian:
        jacobian_fn = getattr(diagenetic_reactions, "compute_chemical_jacobian", None)
    if jacobian_fn is None and not use_numerical_jacobian:
        rxn_mod = getattr(diagenetic_reactions, "__module__", None)
        if rxn_mod and rxn_mod in sys.modules:
            jacobian_fn = getattr(sys.modules[rxn_mod], "compute_chemical_jacobian", None)

    if jacobian_fn is not None:
        _log("Monolithic solver: Analytical chemical Jacobian enabled.")
        print("Monolithic solver: Analytical chemical Jacobian enabled.")
    else:
        _log("Monolithic solver: Finite-difference chemical Jacobian enabled.")
        print("Monolithic solver: Finite-difference chemical Jacobian enabled.")

    def _eval_rates(u_eval: np.ndarray, dt_step: float) -> Dict[str, np.ndarray]:
        """Evaluates bulk reaction rates for a given concentration state."""
        for s_idx, s_name in enumerate(mono_system.species_names):
            c_numpy[s_name].value[:] = u_eval[:, s_idx]
        mp_numpy.current_dt = dt_step
        _, rates_out = diagenetic_reactions(mp_numpy, c_numpy, k, f=f_res_scratch)
        return {s: np.array(val, copy=True) for s, val in rates_out.items()}

    governor_stats = GovernorStats()
    video_dt_sec = parse_time_to_seconds(getattr(mp, "video_dt", None))
    last_video_time = -math.inf
    next_milestone_pct = 10
    t_end = getattr(mp, "t_end", math.inf)
    max_steps = getattr(mp, "max_steps", None)

    step = 0
    total_time = 0.0
    status = "Maximum steps or end time reached"
    current_dt = dt_controller.dt
    wall_start_time = time.perf_counter()
    total_newton_iters = 0

    try:
        while (max_steps is None or step < max_steps) and total_time < t_end:
            step += 1

            for s_obj in species_struct:
                s_obj["var"].updateOld()

            mono_system.load_state()
            u_old = mono_system.u_old.copy()

            step_converged = False
            step_first_attempt = True

            while not step_converged:
                try:
                    # Current iterate starting at previous time step solution
                    u_curr = u_old.copy() if step_first_attempt else mono_system.u_curr.copy()
                    prev_wrms_err = math.inf

                    for iter_idx in range(1, max_newton_iters + 1):
                        total_newton_iters += 1
                        governor_stats.total_sweeps += 1

                        # 1. Evaluate base reaction rates
                        rates_base = _eval_rates(u_curr, current_dt)

                        # 2. Form residual vector R = (eff_phi_vol / dt)*(u - u_old) + T*u - b_trans - vol*rates
                        Tu = mono_system.evaluate_transport_operator(u_curr)
                        R = (mono_system.eff_phi_vol / current_dt) * (u_curr - u_old) + Tu - mono_system.b_transport
                        for s_idx, s_name in enumerate(mono_system.species_names):
                            r_bulk = rates_base.get(s_name, 0.0)
                            if hasattr(r_bulk, "value"):
                                r_bulk = np.asarray(r_bulk.value)
                            R[:, s_idx] -= mono_system.vol * r_bulk

                        # 3. Form Chemical Jacobian J_chem[:, s, m] = dR_s / dC_m
                        if jacobian_fn is not None:
                            mono_system.J_chem[:] = jacobian_fn(
                                c_numpy, mp_numpy, k, mono_system.species_names
                            )
                        else:
                            mono_system.J_chem.fill(0.0)
                            eps_rel = 1e-7
                            for m_idx, m_name in enumerate(mono_system.species_names):
                                cm = u_curr[:, m_idx]
                                delta_m = eps_rel * np.maximum(np.abs(cm), 1e-6)

                                u_curr[:, m_idx] += delta_m
                                rates_pert = _eval_rates(u_curr, current_dt)
                                u_curr[:, m_idx] -= delta_m

                                for s_idx, s_name in enumerate(mono_system.species_names):
                                    r_p = rates_pert.get(s_name, 0.0)
                                    r_b = rates_base.get(s_name, 0.0)
                                    if hasattr(r_p, "value"):
                                        r_p = np.asarray(r_p.value)
                                    if hasattr(r_b, "value"):
                                        r_b = np.asarray(r_b.value)
                                    mono_system.J_chem[:, s_idx, m_idx] = (r_p - r_b) / delta_m

                        # 4. Assemble Block Tridiagonal Matrix B_i
                        # B_i[s, m] = delta_{sm} * (eff_phi_vol / dt + B_trans) - vol * J_chem[s, m]
                        mono_system.B_work.fill(0.0)
                        vol_col = mono_system.vol[:, np.newaxis, np.newaxis]
                        mono_system.B_work = -vol_col * mono_system.J_chem

                        inv_dt = mono_system.eff_phi_vol / current_dt + mono_system.B_trans
                        for s_idx in range(S):
                            mono_system.B_work[:, s_idx, s_idx] += inv_dt[:, s_idx]

                        # RHS: -R
                        mono_system.res[:] = -R

                        # 5. Solve block tridiagonal linear system for delta_u
                        solve_block_tridiagonal_thomas(
                            mono_system.A,
                            mono_system.B_work,
                            mono_system.C,
                            mono_system.res,
                            mono_system.delta_u,
                            mono_system.C_prime,
                            mono_system.D_prime,
                        )

                        # 6. Backtracking line search with non-negativity protection
                        res_norm_curr = np.sqrt(np.mean(R**2))
                        alpha = 1.0
                        u_best = u_curr.copy()
                        min_res_norm = res_norm_curr
                        best_alpha = 1.0

                        for line_step in range(4):
                            u_trial = u_curr + alpha * mono_system.delta_u
                            neg_mask = u_trial < 0.0
                            if np.any(neg_mask):
                                u_trial[neg_mask] = np.where(u_curr[neg_mask] < 1e-6, 0.0, 0.1 * u_curr[neg_mask])

                            rates_trial = _eval_rates(u_trial, current_dt)
                            Tu_trial = mono_system.evaluate_transport_operator(u_trial)
                            R_trial = (mono_system.eff_phi_vol / current_dt) * (u_trial - u_old) + Tu_trial - mono_system.b_transport
                            for s_idx, s_name in enumerate(mono_system.species_names):
                                r_bulk = rates_trial.get(s_name, 0.0)
                                if hasattr(r_bulk, "value"):
                                    r_bulk = np.asarray(r_bulk.value)
                                R_trial[:, s_idx] -= mono_system.vol * r_bulk

                            res_norm_trial = np.sqrt(np.mean(R_trial**2))
                            if res_norm_trial < min_res_norm or line_step == 3:
                                min_res_norm = res_norm_trial
                                u_best = u_trial
                                best_alpha = alpha
                                break
                            alpha *= 0.5
                            u_best = u_trial
                            best_alpha = alpha

                        # 7. Check convergence on actual projected step
                        raw_newton_err = 0.0
                        newton_atol = float(getattr(mp, "newton_atol", 1e-6))
                        is_wrms = (newton_norm == "wrms")
                        for s_idx in range(S):
                            if is_wrms:
                                err_s = compute_species_residual_wrms(
                                    u_best[:, s_idx],
                                    u_curr[:, s_idx],
                                    inner_tol=newton_tol,
                                    atol=newton_atol,
                                )
                            else:
                                err_s = compute_species_residual_linf(
                                    u_best[:, s_idx],
                                    u_curr[:, s_idx],
                                    inner_tol=newton_tol,
                                    atol=newton_atol,
                                )
                            raw_newton_err = max(raw_newton_err, err_s)

                        u_curr[:] = u_best

                        if getattr(mp, "verbose", False):
                            max_du_per_species = {
                                mono_system.species_names[s]: float(np.max(np.abs(mono_system.delta_u[:, s])))
                                for s in range(S)
                            }
                            top_sp = sorted(max_du_per_species.items(), key=lambda x: x[1], reverse=True)[:3]
                            print(
                                f"  [Iter {iter_idx}] max_R={np.max(np.abs(R)):.2e}, "
                                f"res_norm={min_res_norm:.2e}, "
                                f"max_du={np.max(np.abs(mono_system.delta_u)):.2e} ({top_sp}), "
                                f"alpha={best_alpha:.3f}, err={raw_newton_err:.2e} ({newton_norm})",
                                flush=True,
                            )

                        res_tol = float(getattr(mp, "newton_res_tol", 1e-12))
                        if raw_newton_err <= 1.0 or min_res_norm <= res_tol:
                            step_converged = True
                            last_iters = iter_idx
                            break

                    if not step_converged:
                        raise RuntimeError(
                            f"Newton iteration failed to converge in {max_newton_iters} iterations (err={raw_newton_err:.2e})"
                        )

                except Exception as ex:
                    # Step failed: cut dt and retry
                    step_first_attempt = False
                    current_dt = max(current_dt * dt_controller.cut_factor, dt_controller.dt_min)
                    dt_controller._dt = current_dt
                    if current_dt <= dt_controller.dt_min + 1e-15:
                        raise RuntimeError(f"Monolithic solver failed at dt_min: {ex}") from ex
                    continue

            # Step accepted
            governor_stats.total_steps += 1
            mono_system.sync_to_variables(u_curr)
            total_time += current_dt

            # Optional post-step instantaneous reactions
            if hasattr(mp, "instantenous_reactions") and mp.instantenous_reactions:
                try:
                    equilibrium_reactions(mp, c, k, None, rates_base, current_dt)
                except Exception:
                    pass

            # Adapt dt based on Newton iterations
            if last_iters <= target_optimal_iters:
                dt_controller._dt = min(current_dt * dt_controller.growth_factor, dt_controller.dt_max)
                governor_stats.multi_sweep_sweeps += 1
            elif last_iters <= max_acceptable_iters:
                dt_controller._dt = min(current_dt * 1.05, dt_controller.dt_max)
            else:
                dt_controller._dt = max(current_dt * 0.85, dt_controller.dt_min)
                governor_stats.graceful_sweeps += 1
            current_dt = dt_controller._dt

            # Milestone reporting
            pct = int((total_time / t_end) * 100) if t_end < math.inf else 0
            if step % 25 == 0 or pct >= next_milestone_pct:
                t_str = f"{get_time_units(total_time):.1f~P}"
                dt_str = f"{get_time_units(current_dt):.2f~P}"
                print(f"Step {step}, time={t_str} ({pct}%), dt={dt_str}, iters={last_iters}", flush=True)
                if pct >= next_milestone_pct:
                    next_milestone_pct += 10

            # Live plotting / Video frame dispatch
            if plot_queue is not None and video_dt_sec is not None:
                if (total_time - last_video_time) >= video_dt_sec or step == 1 or total_time >= t_end:
                    last_video_time = total_time
                    title_str = f"Time: {get_time_units(total_time):.2f~P}"
                    write_to_queue_async(
                        plot_queue,
                        mp,
                        c,
                        getattr(mp, "k", k),
                        species_list_full,
                        z,
                        D_mol,
                        diagenetic_reactions,
                        equilibrium_reactions,
                        current_dt,
                        title_str,
                    )

    except KeyboardInterrupt:
        status = "Interrupted by user"
        print(f"\nSimulation interrupted by user at step {step}.")
    except Exception as e:
        status = f"Solver crashed: {e}"
        print(f"\nSimulation failed: {e}")
        traceback.print_exc()

    wall_total = time.perf_counter() - wall_start_time
    swp_per_sec = (total_newton_iters / wall_total) if wall_total > 0 else 0.0

    print(
        f"\nFinal Report: {status} in {step} steps ({total_newton_iters} total Newton iterations, "
        f"{swp_per_sec:.2f} it/s). Total Wall Time: {wall_total:.1f}s",
        flush=True,
    )

    if plot_queue is not None:
        title_str = f"Final Time: {get_time_units(total_time):.2f~P}"
        write_to_queue_async(
            plot_queue,
            mp,
            c,
            getattr(mp, "k", k),
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
            current_dt,
            title_str,
        )

    # Save final results
    experiment = getattr(mp, "experiment", "model_output")
    state_out = f"{experiment}_state.npz"
    try:
        save_state(c, state_out)
        print(f"State saved to {state_out}")
    except Exception as e:
        print(f"Could not save state: {e}")

    # Save final PDF if plot_queue is None and layout_file is specified
    if hasattr(mp, "layout_file") and mp.layout_file and plot_queue is None:
        save_final_pdf(
            mp,
            c,
            k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
        )

    return step, 0.0
