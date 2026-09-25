"""Build the equation matrix and call the respective solvers."""

from __future__ import annotations

import gc
import gzip
import math
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Callable


def _compress_log_file(log_path: str) -> Optional[str]:
    """
    Compresses the log file using gzip after closing the file handle and removes the original plain file.
    """
    if not os.path.exists(log_path):
        return None
    gz_path = f"{log_path}.gz"
    try:
        with open(log_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(log_path)
        return gz_path
    except Exception as e:
        print(f"Warning: Failed to compress log file {log_path}: {e}", flush=True)
        return None


import numpy as np

from .diff_lib import (
    get_time_units,
    save_data,
    save_data_async,
    save_state,
)
from .live_plot_lib import write_to_queue_async




@dataclass
class AdaptiveDT:
    """
    Advanced PID-controlled adaptive time stepping.

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
    _err_prev: float | None = None  # Delay initialization
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
            # Only count steps towards the hold period once dt has actually reached the ceiling
            if self._dt >= self._dt_ceiling * 0.98:
                self._steps_at_ceiling += 1

            if self._steps_at_ceiling >= self.failure_hold_steps:
                # Hold period at ceiling has elapsed; increase ceiling if ceiling_growth_factor > 1.0
                if self.ceiling_growth_factor > 1.0 and self._dt_ceiling < self.dt_max:
                    old_ceiling = self._dt_ceiling
                    self._dt_ceiling = min(
                        self._dt_ceiling * self.ceiling_growth_factor,
                        self.dt_max,
                    )
                    self._steps_at_ceiling = 0  # Reset for another hold period at the new ceiling
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
        """
        Compute the next dt based on solver performance and change magnitude.

        Parameters:
        -----------
        error_metric : float
            Magnitude of variable change (e.g., Max or RMS) used for control.
        dt_cfl : float, optional
            Hard upper bound based on stability limits.
        step_success : bool
            Whether the linear solver converged.
        target_error : float
            The desired change per step.
        cut_factor : float, optional
            Explicit factor by which to cut dt on failure.
        """
        # 1. Handle step failure
        if not step_success:
            factor = self.cut_factor if cut_factor is None else cut_factor
            self._dt = max(self._dt * factor, self.dt_min)
            return self._dt

        effective_max = self.get_effective_max()

        # 2. PID Control (H211B)
        if self.pid:
            # We want max_change to be around target_error
            # If this is the first step, assume we are at the target
            if self._err_prev is None:
                self._err_prev = target_error

            err = max(error_metric, 1e-25)
            err_prev = max(self._err_prev, 1e-25)
            err_ref = target_error

            # PID Factor calculation: (ref/err) makes it shrink if err > ref
            factor = (
                (err_prev / err) ** self.kP
                * (err_ref / err) ** self.kI
                * (err_ref / err_prev) ** self.kD
            )

            # Dampen and limit the factor
            factor = max(self.cut_factor, min(self.growth_factor, factor))
            self._dt = max(self.dt_min, min(self._dt * factor, effective_max))
        else:
            # Simplistic growth/cut logic
            if error_metric < target_error:
                self._dt = min(self._dt * self.growth_factor, effective_max)
            else:
                self._dt = max(self._dt * self.cut_factor, self.dt_min)

        # 3. Apply CFL cap if provided
        if dt_cfl is not None:
            self._dt = min(self._dt, dt_cfl)

        # 4. Store state
        self._err_prev = error_metric
        return self._dt








def _calculate_dt_max_isotope(mp: Any, k: Any, D_mol: Any) -> float:
    """Calculates the maximum timestep constraint for isotope coupling dynamically
    based on the pseudo-first-order kinetic relaxation rate k_eff.

    Formula:
        dt_max_isotope = min(dt_max_user, gamma / k_eff)
    where gamma = 200.0 is the dimensionless kinetic coupling factor.
    """
    dt_max_user = float(getattr(mp, "dt_max_isotope", getattr(mp, "dt_max", 604800.0)))
    
    isotope_limiter_species = getattr(mp, "isotope_limiter_species", "FeS")
    if isotope_limiter_species == "FeS" and hasattr(k, "FeS_isp"):
        k_prec = float(getattr(k, "FeS_isp", 0.0))
        Hplus = float(getattr(k, "Hplus", 3.162e-5))
        FeS_sp = float(getattr(k, "FeS_sp", 0.3162))
        hs_frac = float(getattr(mp, "hs_frac", 0.76))
        
        bc_Fe3 = float(getattr(mp, "bc_Fe3", 0.0))
        w = max(float(getattr(mp, "w", 0.0)), 1e-20)
        phi_val = getattr(mp, "phi", 0.8)
        if hasattr(phi_val, "value"):
            phi_val = phi_val.value[0] if hasattr(phi_val.value, "__getitem__") else float(phi_val.value)
        phi_val = float(phi_val)
        Fe2_diss = float(getattr(mp, "Fe2_diss", 1.0))
        
        if bc_Fe3 > 0.0:
            Fe2_pw = (bc_Fe3 * Fe2_diss) / (w * phi_val)
        else:
            Fe2_pw = 0.1
            
        omega_den = Hplus * FeS_sp + 1e-30
        dO_dTS2 = Fe2_pw * hs_frac / omega_den
        k_eff = k_prec * 2.0 * dO_dTS2
        
        if k_eff > 0:
            gamma = float(getattr(mp, "isotope_gamma", 200.0))
            dt_kinetic = gamma / k_eff
            return float(min(dt_max_user, dt_kinetic))

    return dt_max_user


def _find_consecutive_trues(mask: np.ndarray, min_consecutive: int) -> Tuple[bool, int, int]:
    """
    Check if a boolean mask has at least `min_consecutive` consecutive True values.
    Returns (found, start_idx, count).
    """
    if min_consecutive <= 1:
        if np.any(mask):
            idx = int(np.where(mask)[0][0])
            return True, idx, 1
        return False, -1, 0

    if len(mask) < min_consecutive:
        return False, -1, 0

    padded = np.empty(len(mask) + 2, dtype=bool)
    padded[0] = False
    padded[-1] = False
    padded[1:-1] = mask
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    valid = lengths >= min_consecutive
    if np.any(valid):
        first_valid = int(np.where(valid)[0][0])
        idx = int(starts[first_valid])
        consec_count = int(lengths[first_valid])
        return True, idx, consec_count
    return False, -1, 0


def _compute_inner_residual(
    species_struct: List[Dict[str, Any]],
    prev_iterate: Dict[str, np.ndarray],
    inner_tol: float = 1e-4,
    atol_default: float = 1e-6,
    inner_norm: str = "wrms",
) -> float:
    """Computes normalized relative error across all active species between successive inner sweeps:

    For inner_norm == 'wrms':
        err = max_{species} sqrt( (1/N) * sum_{cells} ( |c^{m+1} - c^m| / (inner_tol * |c^{m+1}| + atol) )^2 )
    For inner_norm == 'linf':
        err = max_{species, cells} ( |c^{m+1} - c^m| / (inner_tol * |c^{m+1}| + atol) )
    Returns value <= 1.0 when inner convergence criterion is met.
    """
    max_err_ratio = 0.0
    for s_obj in species_struct:
        name = s_obj["name"]
        curr_val = s_obj["var"].value
        prev_val = prev_iterate[name]
        diff = np.abs(curr_val - prev_val)
        scale = inner_tol * np.abs(curr_val) + atol_default
        ratio = diff / scale
        if np.any(np.isnan(ratio)) or np.any(np.isinf(ratio)):
            return float("inf")
        if inner_norm == "wrms":
            err_ratio = float(np.sqrt(np.mean(ratio**2)))
        else:
            err_ratio = float(np.max(ratio))
        if err_ratio > max_err_ratio:
            max_err_ratio = err_ratio
    return max_err_ratio


def _check_cross_coupling_dominance(
    CROSS_vars: Dict[str, List[Tuple[Any, str]]],
    c: Any,
    current_dt: float,
    species_struct: List[Dict[str, Any]],
    threshold: float = 0.05,
    atol: float = 1e-6,
) -> bool:
    """Check if off-diagonal cross-coupling fluxes are significant relative to species inventory.

    Returns True if cross-couplings are dominant (>= threshold), requiring at least 2 Picard sweeps.
    """
    for s_obj in species_struct:
        name = s_obj["name"]
        cross_list = CROSS_vars.get(name, [])
        if not cross_list:
            continue
        c_target = np.abs(s_obj["var"].value)
        total_cross_flux = np.zeros_like(c_target)
        for v_cross, source_name in cross_list:
            source_var = getattr(c, source_name)
            c_source = np.abs(source_var.value if hasattr(source_var, "value") else source_var)
            coeff_val = np.abs(v_cross.value if hasattr(v_cross, "value") else v_cross)
            total_cross_flux += coeff_val * c_source
        rel_change = (total_cross_flux * current_dt) / (c_target + atol)
        if np.max(rel_change) > threshold:
            return True
    return False


def _validate_rates(
    monitored_rate_species: List[str],
    RATES_tentative: Dict[str, np.ndarray],
    prev_rates: Dict[str, np.ndarray],
    prev_rates_2: Dict[str, np.ndarray],
    current_dt: float,
    prev_dt: float,
    prev_dt_2: float,
    rate_threshold: float,
    enable_rate_magnitude_check: bool,
    rate_sign_min_change: float = 2e-8,
    rate_sign_min_consecutive_cells: int = 1,
) -> Tuple[bool, str]:
    """Performs rate validation checks (consecutive sign changes and magnitude checks)."""
    violation = False
    violation_reason = ""
    for name in monitored_rate_species:
        if name not in RATES_tentative or name not in prev_rates:
            continue
        
        r_tentative = np.asarray(RATES_tentative[name])
        r_prev = np.asarray(prev_rates[name])

        c_change_tentative = r_tentative * current_dt
        c_change_prev = r_prev * prev_dt
        
        # 1. Sign change check
        mask_sign = (np.abs(c_change_prev) >= rate_threshold) & (np.abs(c_change_tentative) >= rate_threshold)
        if np.any(mask_sign):
            abs_change = np.abs(c_change_tentative - c_change_prev)
            abs_rate_change_tentative = np.abs(r_tentative - r_prev)
            flipped_tentative = (
                (c_change_tentative * c_change_prev < 0)
                & (abs_change >= 5.0 * rate_threshold)
                & (abs_rate_change_tentative >= rate_sign_min_change)
            )
            
            if name in prev_rates_2:
                r_prev_2 = np.asarray(prev_rates_2[name])
                c_change_prev_2 = r_prev_2 * prev_dt_2
                mask_sign_prev = (np.abs(c_change_prev_2) >= rate_threshold) & (np.abs(c_change_prev) >= rate_threshold)
                abs_change_prev = np.abs(c_change_prev - c_change_prev_2)
                abs_rate_change_prev = np.abs(r_prev - r_prev_2)
                flipped_prev = (
                    (c_change_prev * c_change_prev_2 < 0)
                    & (abs_change_prev >= 5.0 * rate_threshold)
                    & (abs_rate_change_prev >= rate_sign_min_change)
                )
                osc_mask = mask_sign & mask_sign_prev & flipped_tentative & flipped_prev
            else:
                osc_mask = np.zeros_like(mask_sign, dtype=bool)

            has_consec, idx, consec_count = _find_consecutive_trues(osc_mask, rate_sign_min_consecutive_cells)
            if has_consec:
                violation = True
                r_prev_2 = np.asarray(prev_rates_2[name])
                cell_desc = f"across {consec_count} consecutive cells starting at cell {idx}" if consec_count > 1 else f"at cell {idx}"
                violation_reason = (
                    f"Consecutive sign changes (oscillation) in {name} rate {cell_desc} "
                    f"(prev2 rate: {r_prev_2[idx]:.2e}, prev rate: {r_prev[idx]:.2e}, tentative rate: {r_tentative[idx]:.2e}, rate_change: {abs_rate_change_tentative[idx]:.2e} mol/(m^3*s), conc_change: {abs_change[idx]:.2e} mmol/L)"
                )
                break
        
        # 2. Order-of-magnitude check
        if enable_rate_magnitude_check:
            mask_magnitude = (np.abs(c_change_prev) >= rate_threshold) & (np.abs(r_prev) >= 1e-8)
            if np.any(mask_magnitude):
                ratio = np.abs(c_change_tentative[mask_magnitude]) / np.abs(c_change_prev[mask_magnitude])
                large_increase = ratio > 10.0
                if np.any(large_increase):
                    idx = np.where(mask_magnitude)[0][np.where(large_increase)[0][0]]
                    violation = True
                    violation_reason = (
                        f"Order-of-magnitude rate increase in {name} at cell {idx} "
                        f"(prev: {r_prev[idx]:.2e}, tentative: {r_tentative[idx]:.2e}, ratio: {ratio[large_increase][0]:.2f})"
                    )
                    break
                    
    return violation, violation_reason


def _format_wall_time(seconds: float) -> str:
    """Format wall time into human-friendly string."""
    if seconds < 60:
        return f"{seconds:5.1f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}m {s:02d}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h:02d}h {m:02d}m"


def _format_sim_speed(sim_seconds: float, wall_seconds: float) -> str:
    """Format simulated time speedup relative to wall time."""
    if wall_seconds <= 1e-6:
        return ""
    speedup = sim_seconds / wall_seconds
    if speedup >= 3.1536e7:  # >= 1 yr/sec
        return f"{speedup / 3.1536e7:.2f} yr/s"
    elif speedup >= 3.1536e7 / 60.0:  # >= 1 yr/min
        return f"{(speedup / 3.1536e7) * 60.0:.2f} yr/min"
    elif speedup >= 86400.0 / 60.0:  # >= 1 day/min
        return f"{(speedup / 86400.0) * 60.0:.1f} d/min"
    elif speedup >= 3600.0 / 60.0:  # >= 1 hr/min
        return f"{(speedup / 3600.0) * 60.0:.1f} h/min"
    else:
        return f"{speedup * 60.0:.1f} s/min"


def _report_step_status(
    step: int,
    total_time: float,
    current_dt: float,
    rms_change: float,
    mp: Any,
    c: Any,
    z: np.ndarray,
    species_list_full: List[str],
    D_mol: Any,
    diagenetic_reactions: Any,
    equilibrium_reactions: Any,
    plot_queue: Optional[Any],
    _log: Callable[[str], None],
    sweeps: Optional[int] = None,
    total_sweeps: Optional[int] = None,
    start_wall: Optional[float] = None,
    last_report_wall: Optional[float] = None,
    last_report_time: Optional[float] = None,
) -> None:
    """Logs current step status parameters and triggers async data/plot saving."""
    from .diff_lib import get_delta, get_total_delta
    
    phi = mp.phi
    dz = np.diff(z)
    fe_total_bulk = phi * c.Fe2_total.value + (1 - phi) * (c.Fe3.value + c.FeS.value + c.FeS2.value)
    fe_slice = fe_total_bulk if len(fe_total_bulk) == len(dz) else fe_total_bulk[:-1]
    m_fe = float(np.sum(dz * fe_slice))
    time_str = f"Time: {get_time_units(total_time):.2f~P}"

    # Sweeps string
    if sweeps is not None:
        sweeps_str = f", sweeps: {sweeps}"
        if total_sweeps is not None:
            sweeps_str += f" (tot: {total_sweeps})"
    elif total_sweeps is not None:
        sweeps_str = f", sweeps: 1 (tot: {total_sweeps})"
    else:
        sweeps_str = ""

    # Wall-clock timestamp and throughput metrics
    wall_prefix = ""
    perf_str = ""
    if start_wall is not None:
        elapsed = time.time() - start_wall
        wall_prefix = f"[{_format_wall_time(elapsed)}] "
        if last_report_wall is not None:
            interval_wall = max(time.time() - last_report_wall, 1e-6)
            interval_sim = total_time - last_report_time if last_report_time is not None else current_dt
            report_n = getattr(mp, "report_step", 10)
            pace = interval_wall / max(report_n, 1)
            sim_speed = _format_sim_speed(interval_sim, interval_wall)
            perf_str = f" | {pace:.2f} s/step, {sim_speed}" if sim_speed else f" | {pace:.2f} s/step"

    if mp.isotopes:
        d34s = get_total_delta(c, mp)
        fes_mask = c.FeS.value > 1e-3
        d_fes = get_delta(c.FeS.value[fes_mask], c.FeS_32.value[fes_mask], mp.VCDT) if np.any(fes_mask) else np.array([])
        v_fes = d_fes[~np.isnan(d_fes)]
        min_dFeS = float(np.min(v_fes)) if len(v_fes) > 0 else np.nan
        max_dFeS = float(np.max(v_fes)) if len(v_fes) > 0 else np.nan

        ts2_mask = c.TS2.value > 1e-3
        d_ts2 = get_delta(c.TS2.value[ts2_mask], c.TS2_32.value[ts2_mask], mp.VCDT) if np.any(ts2_mask) else np.array([])
        v_ts2 = d_ts2[~np.isnan(d_ts2)]
        min_dTS2 = float(np.min(v_ts2)) if len(v_ts2) > 0 else np.nan
        max_dTS2 = float(np.max(v_ts2)) if len(v_ts2) > 0 else np.nan

        _log(
            f"{wall_prefix}Step {step:4d},  {time_str}, "
            f"dt: {get_time_units(current_dt):.2f~P}{perf_str}{sweeps_str}, "
            f"RMS: {rms_change:.2e}, "
            f"d34S = {d34s:.2f}, "
            f"dTS2: [{min_dTS2:.1f}, {max_dTS2:.1f}]‰, "
            f"dFeS: [{min_dFeS:.1f}, {max_dFeS:.1f}]‰"
        )
        if mp.title is None:
            title_str = time_str + r", $\delta^{34}$S = " + f"{d34s:.1f} [mUr]"
        else:
            title_str = mp.title
    else:
        _log(
            f"{wall_prefix}Step {step:4d},  {time_str}, "
            f"dt: {get_time_units(current_dt):.2f~P}{perf_str}{sweeps_str}, "
            f"RMS Chg: {rms_change:.2e}, "
            f"Total Fe {m_fe:.2e}"
        )
        if mp.title is None:
            title_str = time_str
        else:
            title_str = mp.title

    if plot_queue is not None:
        write_to_queue_async(
            plot_queue,
            mp,
            c,
            mp.k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
            current_dt,
            title_str,
        )
    else:
        save_data_async(
            mp,
            c,
            mp.k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
            current_dt,
            title=title_str,
        )


def apply_porewater_depletion_governor(
    species_struct: List[Dict[str, Any]],
    bc_map: Dict[str, Any],
    current_dt: float,
    proposed_dt: float,
    max_rel_change: float = 0.25,
    conc_scale: float = 1e-4,
    conc_presence_floor: float = 1e-3,
    dt_min: float = 1.0,
    dt_max: float = 3.1536e7,
    _log: Optional[Callable[[str], None]] = None,
    wall_time_str: str = "",
) -> Tuple[float, float, str, int]:
    """
    Relative Porewater Depletion Governor.

    Dynamically bounds the time step based on the maximum fractional temporal
    change of any active dissolved species:
        rel_change = max_{s in dissolved} max_{j in active} (|C_s^{n+1} - C_s^n| / (C_s^{n+1} + conc_scale))

    Only cells where the species is physically present (C_new > conc_presence_floor
    or C_old > conc_presence_floor) are evaluated, avoiding false triggers from
    iterative linear solver residual noise in deep/exhausted zones (e.g. O2 ~ 1e-4 mmol/L).

    If rel_change exceeds max_rel_change (e.g. 25%), scales down dt to prevent
    over-depleting porewater intermediate pools and overshooting non-linear
    equilibrium thresholds (e.g. Omega = 1.0 for FeS precipitation/dissolution).

    If rel_change is well within max_rel_change, sets a soft headroom cap on dt growth
    to ensure dt does not abruptly jump into an overshooting regime.

    Returns:
        (adapted_dt, max_observed_rel, limiting_species, limiting_cell)
    """
    max_observed_rel = 0.0
    limiting_species = ""
    limiting_cell = -1

    for s_obj in species_struct:
        s_name = s_obj["name"]
        if bc_map.get(s_name, {}).get("type") != "dissolved":
            continue
        c_new = np.asarray(s_obj["var"].value)
        c_old = np.asarray(s_obj["var"].old.value)

        # Only evaluate cells where the substance is actually present in significant quantity
        active_mask = (c_new > conc_presence_floor) | (c_old > conc_presence_floor)
        if not np.any(active_mask):
            continue

        denom = np.maximum(c_new, 0.0) + conc_scale
        rel_arr = np.where(active_mask, np.abs(c_new - c_old) / denom, 0.0)
        loc_max_idx = int(np.argmax(rel_arr))
        loc_max = float(rel_arr[loc_max_idx])
        if loc_max > max_observed_rel:
            max_observed_rel = loc_max
            limiting_species = s_name
            limiting_cell = loc_max_idx

    adapted_dt = proposed_dt
    if max_observed_rel > max_rel_change:
        # Exceeded target fractional change: scale down dt
        factor = max(0.5, (max_rel_change / max_observed_rel) ** 0.5)
        governor_dt = max(dt_min, min(current_dt * factor, proposed_dt))
        adapted_dt = min(adapted_dt, governor_dt)
        if _log is not None and factor < 0.98:
            _log(
                f"{wall_time_str}  Porewater governor: {limiting_species} changed by "
                f"{max_observed_rel * 100:.1f}% at cell {limiting_cell} (target: {max_rel_change * 100:.1f}%). "
                f"dt capped at {get_time_units(adapted_dt):.2f~P}."
            )
    elif max_observed_rel > 0.0:
        # Within target: set soft headroom cap on growth to prevent jumping past threshold
        headroom = max_rel_change / max(max_observed_rel, 0.01)
        growth_cap = max(dt_min, min(dt_max, current_dt * min(2.0, headroom ** 0.5)))
        adapted_dt = min(adapted_dt, growth_cap)

    return adapted_dt, max_observed_rel, limiting_species, limiting_cell


def run_non_steady_state_solver_coupled(
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
    """Solves the non-steady state ADR coupled reactive transport system using direct LAPACK assembly."""
    from .direct_assembled_solver import run_non_steady_state_solver_direct

    return run_non_steady_state_solver_direct(
        mp=mp,
        c=c,
        species_list_full=species_list_full,
        species_list_partial=species_list_partial,
        k=k,
        diagenetic_reactions=diagenetic_reactions,
        equilibrium_reactions=equilibrium_reactions,
        mesh=mesh,
        D_mol=D_mol,
        bc_map=bc_map,
        z=z,
        plot_queue=plot_queue,
    )
