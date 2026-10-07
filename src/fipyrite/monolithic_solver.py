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
from .direct_assembled_solver import (
    GovernorStats,
    build_native_1d_transport_stencil,
)
from .live_plot_lib import parse_time_to_seconds, save_final_pdf, write_to_queue_async
from .solver_calls import AdaptiveDT
from .solvers_numba import (
    compute_species_residual_linf,
    compute_species_residual_wrms,
    solve_block_tridiagonal_thomas,
)


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
                        raw_wrms_err = 0.0
                        newton_atol = float(getattr(mp, "newton_atol", 1e-6))
                        for s_idx in range(S):
                            err_s = compute_species_residual_wrms(
                                u_best[:, s_idx],
                                u_curr[:, s_idx],
                                inner_tol=newton_tol,
                                atol=newton_atol,
                            )
                            raw_wrms_err = max(raw_wrms_err, err_s)

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
                                f"alpha={best_alpha:.3f}, wrms={raw_wrms_err:.2e}",
                                flush=True,
                            )

                        res_tol = float(getattr(mp, "newton_res_tol", 1e-12))
                        if raw_wrms_err <= 1.0 or min_res_norm <= res_tol:
                            step_converged = True
                            last_iters = iter_idx
                            break

                        if iter_idx == max_newton_iters:
                            step_converged = True
                            last_iters = iter_idx

                    if not step_converged:
                        raise RuntimeError(
                            f"Newton iteration failed to converge in {max_newton_iters} iterations (err={raw_wrms_err:.2e})"
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
