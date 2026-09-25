"""Direct Assembled Coupled Solver for FiPyrite.

Bypasses FiPy's binary equation tree traversal and hundreds of per-sweep
sparse matrix additions by pre-extracting the static 1D tridiagonal transport
stencils (diffusion, advection, irrigation) and assembling the Patankar-linearized
reaction-transport system directly into LAPACK banded matrix buffers.
"""

from __future__ import annotations

import gc
import math
import os
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Callable

import numpy as np
from scipy.linalg import solve_banded

from fipy import CellVariable
from .diff_lib import (
    data_container,
    get_time_units,
    save_data,
    save_data_async,
    save_state,
)
from .live_plot_lib import write_to_queue_async

if TYPE_CHECKING:
    from fipy.meshes.mesh import Mesh


class ArrayProxy:
    """Lightweight NumPy array wrapper mimicking FiPy CellVariable for rate evaluations."""

    def __init__(self, val: Any):
        self.value = val

    def __getattr__(self, name: str) -> Any:
        return getattr(self.value, name)

    def __add__(self, other: Any) -> ArrayProxy:
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(self.value + val)

    def __radd__(self, other: Any) -> ArrayProxy:
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(val + self.value)

    def __sub__(self, other: Any) -> ArrayProxy:
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(self.value - val)

    def __rsub__(self, other: Any) -> ArrayProxy:
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(val - self.value)

    def __mul__(self, other: Any) -> ArrayProxy:
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(self.value * val)

    def __rmul__(self, other: Any) -> ArrayProxy:
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(val * self.value)

    def __truediv__(self, other: Any) -> ArrayProxy:
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(self.value / val)

    def __rtruediv__(self, other: Any) -> ArrayProxy:
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(val / self.value)

    def __pow__(self, power: Any) -> ArrayProxy:
        return ArrayProxy(self.value**power)

    def __neg__(self) -> ArrayProxy:
        return ArrayProxy(-self.value)

    def __pos__(self) -> ArrayProxy:
        return self

    def __abs__(self) -> ArrayProxy:
        return ArrayProxy(abs(self.value))

    def __getitem__(self, idx: Any) -> Any:
        return self.value[idx]


class DirectAssembledSystem:
    """Pre-extracts and manages the static tridiagonal transport stencils

    for all species, performing in-place assembly and LAPACK solve on every sweep.
    """

    def __init__(
        self,
        species_struct: List[Dict[str, Any]],
        passive_eqs: Dict[str, Any],
        mesh: Mesh,
        mp: Any,
        bc_map: Optional[Dict[str, Any]] = None,
    ):
        self.mesh = mesh
        self.mp = mp
        self.species_struct = species_struct
        self.num_cells = mesh.numberOfCells
        self.vol = np.asarray(mesh.cellVolumes)

        # Storage per species:
        # ab_transport: shape (3, N) -> row 0: upper diagonal, row 1: main diagonal, row 2: lower diagonal
        # b_transport: shape (N,)
        # eff_phi_vol: shape (N,) -> eff_phi * cellVolumes
        self.species_names = [s["name"] for s in species_struct]
        self.ab_transport: Dict[str, np.ndarray] = {}
        self.b_transport: Dict[str, np.ndarray] = {}
        self.eff_phi_vol: Dict[str, np.ndarray] = {}

        # Default matrix class from FiPy for extraction
        from fipy.solvers import DefaultSolver

        matrixClass = DefaultSolver()._matrixClass

        ref_dt = 1.0
        phi_val = np.asarray(mp.phi.value if hasattr(mp.phi, "value") else mp.phi)

        for s_obj in species_struct:
            name = s_obj["name"]
            var = s_obj["var"]
            eq = passive_eqs[name]
            props = bc_map.get(name, {}) if bc_map is not None else {}

            is_diss = (props.get("type", "dissolved") == "dissolved")
            eff_phi_val = phi_val if is_diss else (1.0 - phi_val)
            eff_phi_v = eff_phi_val * self.vol
            self.eff_phi_vol[name] = eff_phi_v
            transient_ref_diag = eff_phi_v / ref_dt

            # Save original variable state
            orig_val = np.asarray(var.value).copy()
            orig_old = (
                np.asarray(var.old.value).copy()
                if hasattr(var, "old")
                else orig_val.copy()
            )

            # 1. Build reference linear system at var = 0, old = 0
            # With old=0, the TransientTerm contribution is identically zero,
            # so rhs_np0 isolates the spatial boundary influx (Dirichlet or Robin J_in).
            var.setValue(0.0)
            if hasattr(var, "old"):
                var.old.setValue(0.0)
            _, mat_ref0, rhs_ref0 = eq._buildAndAddMatrices(
                var, matrixClass, dt=ref_dt
            )

            # 2. Robin boundary condition sensitivity check at cell 0:
            # For particulate species with a Robin flux BC (w*C_face - J_solid)/D,
            # FiPy evaluates the (1-phi)*w*C_face term in the RHS at each sweep.
            # Measuring d(rhs)/d(var[0]) allows moving this term directly to the matrix diagonal!
            v_test = np.zeros(self.num_cells)
            v_test[0] = 1.0
            var.setValue(v_test)
            _, _, rhs_ref1 = eq._buildAndAddMatrices(
                var, matrixClass, dt=ref_dt
            )

            # Restore original variable state
            var.setValue(orig_val)
            if hasattr(var, "old"):
                var.old.setValue(orig_old)

            mat_np = mat_ref0.numpyArray
            rhs_np0 = np.asarray(rhs_ref0)
            gamma_robin = float(rhs_np0[0] - np.asarray(rhs_ref1)[0])

            # Extract transport matrix A_transport and rhs vector b_transport
            diag_transport = np.diag(mat_np) - transient_ref_diag
            diag_transport[0] += gamma_robin
            upper_transport = np.diag(mat_np, k=1)
            lower_transport = np.diag(mat_np, k=-1)

            # Format for scipy.linalg.solve_banded: shape (3, N)
            ab = np.zeros((3, self.num_cells), dtype=np.float64)
            ab[0, 1:] = upper_transport
            ab[1, :] = diag_transport
            ab[2, :-1] = lower_transport
            self.ab_transport[name] = ab

            self.b_transport[name] = rhs_np0

    def sweep(
        self,
        dt: float,
        c: Any,
        f_res: Any,
        prev_iterate: Dict[str, np.ndarray],
    ) -> None:
        """Assembles and solves the 1D tridiagonal system for all species in-place."""
        vol = self.vol

        def _get_arr(val: Any) -> np.ndarray:
            if hasattr(val, "value"):
                return np.asarray(val.value)
            return np.asarray(val)

        for s_obj in self.species_struct:
            name = s_obj["name"]
            var = s_obj["var"]
            old_val = np.asarray(var.old.value)
            prev_val = prev_iterate[name]

            eff_phi_v = self.eff_phi_vol[name]
            inv_dt_factor = eff_phi_v / dt

            # Retrieve reaction components from f_res
            lhs_val = _get_arr(f_res.raw_LHS.get(name, 0.0))
            rhs_val = _get_arr(f_res.raw_RHS.get(name, 0.0))
            cross_list = f_res.raw_CROSS.get(name, [])

            # Prepare banded matrix (3, N)
            ab = self.ab_transport[name].copy()
            # Patankar linearization on diagonal:
            # -min(0, lhs_val) * vol increases diagonal dominance for negative sink terms
            ab[1, :] += inv_dt_factor - np.minimum(0.0, lhs_val) * vol

            # Prepare RHS vector (N,)
            b = (
                self.b_transport[name]
                + inv_dt_factor * old_val
                + rhs_val * vol
            )
            if np.any(lhs_val > 0.0):
                b += np.maximum(0.0, lhs_val) * vol * prev_val

            # Cross-species couplings: evaluated explicitly using prev_iterate
            for source_name, coeff in cross_list:
                coeff_val = _get_arr(coeff)
                source_prev = prev_iterate[source_name]
                b += coeff_val * vol * source_prev

            # Direct LAPACK banded solve: O(N) in C
            new_val = solve_banded((1, 1), ab, b)

            # Update solution variable
            var.setValue(new_val)


def run_non_steady_state_solver_direct(
    mp: Any,
    c: Any,
    species_list_full: List[str],
    species_list_partial: List[str],
    k: Any,
    diagenetic_reactions: Any,
    equilibrium_reactions: Any,
    mesh: Mesh,
    D_mol: Any,
    bc_map: Dict[str, Any],
    z: np.ndarray,
    plot_queue: Optional[Any] = None,
) -> Tuple[int, float]:
    """Clean, standalone direct-assembled driver for the coupled reactive-transport model."""
    from .solver_calls import (
        AdaptiveDT,
        _build_passive_eqs,
        _compute_inner_residual,
        _format_wall_time,
        _format_sim_speed,
        _report_step_status,
        _compress_log_file,
        _calculate_dt_max_isotope,
        apply_porewater_depletion_governor,
    )

    # Initialize logging
    log_path = f"{mp.plot_name}.log"
    _log_file = open(log_path, "w", buffering=1)

    def _log(msg: str) -> None:
        print(msg, flush=True)
        _log_file.write(msg + "\n")

    total_sweeps = 0
    start_wall = time.time()
    last_report_wall = start_wall
    last_report_time = 0.0

    # Setup time-stepper
    dt_controller = AdaptiveDT(
        dt_min=getattr(mp, "dt_min", 1.0),
        dt_max=getattr(mp, "dt_max", 31536000.0),
        dt_initial=getattr(mp, "dt_init", getattr(mp, "dt_min", 1.0)),
        growth_factor=float(getattr(mp, "dt_growth_factor", 1.2)),
        cut_factor=float(getattr(mp, "dt_cut_factor", 0.5)),
        enable_failure_ceiling=getattr(mp, "enable_failure_ceiling", False),
        failure_ceiling_factor=float(getattr(mp, "failure_ceiling_factor", 0.7)),
        failure_hold_steps=int(getattr(mp, "failure_hold_steps", 10)),
        ceiling_growth_factor=float(getattr(mp, "ceiling_growth_factor", 1.05)),
        aggressive_cut_factor=float(getattr(mp, "aggressive_cut_factor", 0.25)),
        enable_mttf_governor=getattr(mp, "enable_mttf_governor", False),
        mttf_x1=float(getattr(mp, "mttf_x1", 0.60)),
        mttf_x2=float(getattr(mp, "mttf_x2", 0.35)),
        mttf_x3=float(getattr(mp, "mttf_x3", 0.15)),
        mttf_y1=float(getattr(mp, "mttf_y1", 0.02)),
        mttf_y2=float(getattr(mp, "mttf_y2", 0.05)),
        mttf_y3=float(getattr(mp, "mttf_y3", 0.10)),
    )

    print(
        f"Starting Direct Assembled ADR Solver. dt_init: {get_time_units(dt_controller.dt):.2f~P}"
    )
    msg_info = "  [Solver] Direct In-Place Sparse Matrix Assembly active (bypassing FiPy AST)."
    _log(msg_info)
    print(msg_info)

    # Build the transport backbone once
    species_struct, passive_eqs = _build_passive_eqs(
        mp, c, mesh, D_mol, bc_map, species_list_partial
    )

    # Initialize Direct Assembled System
    assembled_system = DirectAssembledSystem(
        species_struct, passive_eqs, mesh, mp, bc_map=bc_map
    )

    # Inner sweeping parameters
    enable_inner_sweeping = getattr(mp, "enable_inner_sweeping", False)
    max_inner_sweeps = int(getattr(mp, "max_inner_sweeps", 15))
    inner_tol = float(getattr(mp, "inner_tol", 1e-3))
    near_convergence_tol = float(getattr(mp, "near_convergence_tol", 1.25))
    graceful_acceptance_tol = float(
        getattr(mp, "graceful_acceptance_tol", near_convergence_tol)
    )
    inner_norm = getattr(mp, "inner_norm", "wrms")
    inner_relaxation = float(getattr(mp, "inner_relaxation", 1.0))
    enable_adaptive_damping = getattr(mp, "enable_adaptive_damping", True)
    inner_sweep_equilibrium = getattr(mp, "inner_sweep_equilibrium", True)
    adaptive_sweeps_dt = getattr(mp, "adaptive_sweeps_dt", True)
    sweep_target_optimal = int(getattr(mp, "sweep_target_optimal", 4))
    sweep_max_acceptable = int(getattr(mp, "sweep_max_acceptable", 7))
    enable_single_sweep_exit = getattr(mp, "enable_single_sweep_exit", False)
    sweep1_exit_tol = float(getattr(mp, "sweep1_exit_tol", 3.0))

    # Porewater governor parameters
    enable_porewater_governor = getattr(mp, "enable_porewater_governor", False)
    max_rel_porewater_change = float(getattr(mp, "max_rel_porewater_change", 0.25))
    porewater_conc_scale = float(getattr(mp, "porewater_conc_scale", 1e-4))
    porewater_conc_presence_floor = float(
        getattr(mp, "porewater_conc_presence_floor", 1e-6)
    )

    # Dynamic isotope dt limiter
    enable_isotope_dt_limiter = getattr(mp, "enable_isotope_dt_limiter", False)
    isotope_limiter_species = getattr(mp, "isotope_limiter_species", "FeS")
    isotope_onset_threshold = float(getattr(mp, "isotope_onset_threshold", 1e-5))
    dt_max_isotope = (
        _calculate_dt_max_isotope(mp, k, D_mol)
        if (enable_isotope_dt_limiter and getattr(mp, "isotopes", False))
        else dt_controller.dt_max
    )

    step = 0
    total_time = 0.0
    status = "Maximum steps or end time reached"
    current_dt = dt_controller.dt
    prev_dt = current_dt
    last_inner_sweeps = 1
    rms_change = 1.0
    prev_was_near_converged = False

    def _eval_reactions(dt_step: float) -> Tuple[Any, Any]:
        c_numpy = data_container(
            {s: ArrayProxy(val.value) for s, val in c.items()}
        )
        mp_numpy = data_container(mp)
        phi_val = mp.phi.value if hasattr(mp.phi, "value") else mp.phi
        mp_numpy.phi = ArrayProxy(phi_val)
        mp_numpy.current_dt = dt_step
        mp_numpy.in_solver = True
        f_res = data_container()
        try:
            f_res, RATES = diagenetic_reactions(mp_numpy, c_numpy, k, f=f_res)
        finally:
            mp_numpy.in_solver = False
        return f_res, RATES

    try:
        while (
            step < getattr(mp, "max_steps", 1000)
            and total_time < getattr(mp, "t_end", math.inf)
        ):
            step += 1

            for s_obj in species_struct:
                s_obj["var"].updateOld()

            converged = False
            step_first_attempt = True
            near_converged_accepted = False
            graceful_accepted = False

            while not converged:
                try:
                    prev_iterate = {
                        s_obj["name"]: np.asarray(s_obj["var"].value).copy()
                        for s_obj in species_struct
                    }
                    prev_inner_err = None
                    current_theta = inner_relaxation

                    if enable_inner_sweeping:
                        inner_sweeps = 0
                        for inner_iter in range(1, max_inner_sweeps + 1):
                            inner_sweeps += 1
                            total_sweeps += 1

                            f_res, RATES = _eval_reactions(current_dt)
                            assembled_system.sweep(
                                current_dt, c, f_res, prev_iterate
                            )

                            raw_inner_err = _compute_inner_residual(
                                species_struct,
                                prev_iterate,
                                inner_tol=inner_tol,
                                inner_norm=inner_norm,
                            )

                            if (
                                enable_adaptive_damping
                                and inner_iter >= 2
                                and prev_inner_err is not None
                            ):
                                if raw_inner_err > prev_inner_err:
                                    current_theta = max(
                                        current_theta * 0.7, 0.4
                                    )
                                elif (
                                    raw_inner_err <= prev_inner_err * 0.8
                                    and current_theta < inner_relaxation
                                ):
                                    current_theta = min(
                                        current_theta * 1.15, inner_relaxation
                                    )

                            if current_theta < 1.0:
                                for s_obj in species_struct:
                                    s_name = s_obj["name"]
                                    s_obj["var"].setValue(
                                        (1.0 - current_theta)
                                        * prev_iterate[s_name]
                                        + current_theta * s_obj["var"].value
                                    )
                                last_inner_err = _compute_inner_residual(
                                    species_struct,
                                    prev_iterate,
                                    inner_tol=inner_tol,
                                    inner_norm=inner_norm,
                                    )
                            else:
                                last_inner_err = raw_inner_err

                            if inner_sweep_equilibrium:
                                mp.in_clip = True
                                try:
                                    equilibrium_reactions(
                                        mp, c, k, None, RATES, current_dt
                                    )
                                finally:
                                    mp.in_clip = False

                            if last_inner_err <= 1.0:
                                last_inner_sweeps = inner_iter
                                break

                            if (
                                inner_iter == 1
                                and enable_single_sweep_exit
                                and not prev_was_near_converged
                                and last_inner_err <= sweep1_exit_tol
                            ):
                                last_inner_sweeps = 1
                                break

                            if inner_iter == max_inner_sweeps:
                                if last_inner_err <= near_convergence_tol:
                                    near_converged_accepted = True
                                    last_inner_sweeps = max_inner_sweeps
                                elif (
                                    last_inner_err <= graceful_acceptance_tol
                                ):
                                    graceful_accepted = True
                                    last_inner_sweeps = max_inner_sweeps
                                else:
                                    raise RuntimeError(
                                        f"Picard sweep failed to converge in {max_inner_sweeps} "
                                        f"iterations (scaled_err={last_inner_err:.2e})"
                                    )

                            prev_inner_err = last_inner_err
                            prev_iterate = {
                                s_obj["name"]: np.asarray(
                                    s_obj["var"].value
                                ).copy()
                                for s_obj in species_struct
                            }

                        if not inner_sweep_equilibrium:
                            mp.in_clip = True
                            try:
                                equilibrium_reactions(
                                    mp, c, k, None, RATES, current_dt
                                )
                            finally:
                                mp.in_clip = False
                    else:
                        total_sweeps += 1
                        f_res, RATES = _eval_reactions(current_dt)
                        assembled_system.sweep(
                            current_dt, c, f_res, prev_iterate
                        )
                        mp.in_clip = True
                        try:
                            equilibrium_reactions(
                                mp, c, k, None, RATES, current_dt
                            )
                        finally:
                            mp.in_clip = False

                    converged = True

                except Exception as e:
                    _log(
                        f"[{_format_wall_time(time.time() - start_wall)}]   Step failed at dt={get_time_units(current_dt):.2f~P}: {e}\n  Cutting dt and retrying."
                    )
                    for s_obj in species_struct:
                        s_obj["var"].value[:] = s_obj["var"].old.value

                    prev_was_near_converged = True
                    if step_first_attempt:
                        current_dt = dt_controller.register_failure(current_dt)
                        step_first_attempt = False
                    else:
                        current_dt = dt_controller.update(
                            0.0, step_success=False
                        )
                    if current_dt <= mp.dt_min * 1.01:
                        raise RuntimeError(
                            "Direct solver failed and time step is already at minimum."
                        )

            # --- Convergence metrics ---
            rms_change = max(
                float(
                    np.sqrt(
                        np.mean(
                            (s_obj["var"].value - s_obj["var"].old.value) ** 2
                        )
                    )
                )
                for s_obj in species_struct
            )

            total_time += current_dt
            if not (near_converged_accepted or graceful_accepted):
                dt_controller.record_success()
            else:
                dt_controller.steps_since_failure = 0
            prev_was_near_converged = near_converged_accepted or graceful_accepted

            # --- Adapt time step for next iteration ---
            if enable_inner_sweeping and adaptive_sweeps_dt:
                effective_max = dt_controller.get_effective_max(_log=_log)
                if graceful_accepted:
                    dt_controller._dt = max(
                        dt_controller._dt * 0.85, dt_controller.dt_min
                    )
                elif near_converged_accepted:
                    dt_controller._dt = max(
                        dt_controller._dt * 0.90, dt_controller.dt_min
                    )
                elif last_inner_sweeps <= sweep_target_optimal:
                    dt_controller._dt = min(
                        dt_controller._dt * dt_controller.growth_factor,
                        effective_max,
                    )
                elif last_inner_sweeps <= sweep_max_acceptable:
                    dt_controller._dt = min(
                        dt_controller._dt * 1.02, effective_max
                    )
                else:
                    dt_controller._dt = max(
                        dt_controller._dt * 0.85, dt_controller.dt_min
                    )
                dt_controller._dt_prev = dt_controller._dt
            else:
                adaptive_target = getattr(mp, "dt_target_change", 1e-4)
                dt_controller.update(
                    error_metric=rms_change,
                    dt_cfl=None,
                    step_success=True,
                    target_error=adaptive_target,
                )

            # --- Apply Dynamic Relative Porewater Depletion Governor ---
            if enable_porewater_governor:
                wall_str = (
                    f"[{_format_wall_time(time.time() - start_wall)}] "
                )
                (
                    adapted_dt,
                    obs_rel,
                    lim_sp,
                    lim_cell,
                ) = apply_porewater_depletion_governor(
                    species_struct=species_struct,
                    bc_map=bc_map,
                    current_dt=current_dt,
                    proposed_dt=dt_controller._dt,
                    max_rel_change=max_rel_porewater_change,
                    conc_scale=porewater_conc_scale,
                    conc_presence_floor=porewater_conc_presence_floor,
                    dt_min=dt_controller.dt_min,
                    dt_max=dt_controller.dt_max,
                    _log=_log,
                    wall_time_str=wall_str,
                )
                dt_controller._dt = adapted_dt
                dt_controller._dt_prev = adapted_dt

            # --- Apply Dynamic Isotope dt Limiter ---
            if enable_isotope_dt_limiter and getattr(mp, "isotopes", False):
                if isotope_limiter_species in c:
                    max_conc = np.max(c[isotope_limiter_species].value)
                    if max_conc > isotope_onset_threshold:
                        dt_controller._dt = min(
                            dt_controller._dt, dt_max_isotope
                        )
                        dt_controller._dt_prev = min(
                            dt_controller._dt_prev, dt_max_isotope
                        )

            if step % getattr(mp, "backup_step", 1000) == 0:
                gc.collect()
                save_state(c, f"{mp.plot_name}_bak.npz")

            if step % getattr(mp, "report_step", 10) == 0:
                _report_step_status(
                    step,
                    total_time,
                    current_dt,
                    rms_change,
                    mp,
                    c,
                    z,
                    species_list_full,
                    D_mol,
                    diagenetic_reactions,
                    equilibrium_reactions,
                    plot_queue,
                    _log,
                    sweeps=last_inner_sweeps if enable_inner_sweeping else None,
                    total_sweeps=total_sweeps,
                    start_wall=start_wall,
                    last_report_wall=last_report_wall,
                    last_report_time=last_report_time,
                )
                last_report_wall = time.time()
                last_report_time = total_time

            # Steady State Check
            if rms_change < mp.dt_tolerance:
                _log(
                    f"Steady State Met: rms_change {rms_change:.2e} < tolerance {mp.dt_tolerance:.2e}"
                )
                status = "Steady State Converged"
                break

            current_dt = dt_controller.dt

    except KeyboardInterrupt:
        status = "Solver interrupted by user"
    except Exception as e:
        status = f"Solver crashed: {e}"
        print(traceback.format_exc())

    elapsed_total = time.time() - start_wall
    sweeps_rate_str = (
        f", {total_sweeps / max(elapsed_total, 1e-6):.2f} swp/s"
        if total_sweeps > 0
        else ""
    )
    _log(
        f"Final Report: {status} in {step} steps ({total_sweeps} total sweeps{sweeps_rate_str}). "
        f"Total Wall Time: {_format_wall_time(elapsed_total)} ({elapsed_total:.2f}s)"
    )

    try:
        csv_file = f"{mp.plot_name}.csv"
        state_file = f"{mp.plot_name}_state.npz"
        print(
            f"Saving final results to {csv_file} and state to {state_file} ...",
            flush=True,
        )
        save_data(
            mp,
            c,
            k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
        )
        save_state(c, state_file)
    except Exception as e:
        print(f"Error during final synchronous save: {e}", flush=True)

    if plot_queue is not None:
        title_str = f"Final Time: {get_time_units(total_time):.2f~P}"
        write_to_queue_async(
            plot_queue,
            mp,
            c,
            k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
            current_dt,
            title_str,
        )

    _log_file.close()
    _compress_log_file(log_path)

    return step, rms_change
