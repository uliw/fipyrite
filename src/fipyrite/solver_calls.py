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

if TYPE_CHECKING:
    from fipy.meshes.mesh import Mesh


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


def _get_solver(mp: Any) -> Any:
    """Initialize and return the FiPy solver based on configuration."""
    backend = mp.solver_backend
    tol = mp.tolerance
    if getattr(mp, "isotopes", False):
        tol = min(tol, 1e-10)

    if backend == "default":
        from fipy import DefaultSolver

        solver = DefaultSolver(tolerance=tol)
    elif backend == "LinearLUSolver":
        from fipy import LinearLUSolver

        solver = LinearLUSolver(tolerance=tol)
    else:
        from petsc4py import PETSc
        if getattr(mp, "solver_monitor", False):
            PETSc.Options().setValue("ksp_monitor", "")
            PETSc.Options().setValue("ksp_converged_reason", "")

        # PETSc version-specific fix for converged reason constants
        if not hasattr(PETSc.KSP.ConvergedReason, "CONVERGED_ATOL_NORMAL"):
            PETSc.KSP.ConvergedReason.CONVERGED_ATOL_NORMAL = (
                PETSc.KSP.ConvergedReason.CONVERGED_ATOL_NORMAL_EQUATIONS
            )
        if not hasattr(PETSc.KSP.ConvergedReason, "CONVERGED_RTOL_NORMAL"):
            PETSc.KSP.ConvergedReason.CONVERGED_RTOL_NORMAL = (
                PETSc.KSP.ConvergedReason.CONVERGED_RTOL_NORMAL_EQUATIONS
            )

        if backend == "LinearGMRESSolver":
            from fipy.solvers.petsc import LinearGMRESSolver

            precon = getattr(mp, "solver_precon", "hypre")
            solver_kwargs = {"precon": precon, "tolerance": tol}
            if hasattr(mp, "solver_atol") and mp.solver_atol is not None:
                solver_kwargs["absolute_tolerance"] = mp.solver_atol
            if hasattr(mp, "solver_max_iterations") and mp.solver_max_iterations is not None:
                solver_kwargs["iterations"] = mp.solver_max_iterations

            solver = LinearGMRESSolver(**solver_kwargs)
        elif backend in ("PETScLUSolver", "LinearLUSolver_petsc"):
            from fipy.solvers.petsc import LinearLUSolver

            solver = LinearLUSolver(tolerance=tol)
        elif backend == "petscSolver":
            # this is currently not working
            from fipy.solvers.petsc import petscSolver

            solver = petscSolver(tolerance=tol)

        elif backend == "PETScNewtonSolver":
            raise ValueError(
                "PETScNewtonSolver is not a valid FiPy solver backend. "
                "FiPy does not have a native PETSc Newton solver class. "
                "Please use 'LinearGMRESSolver' or 'LinearLUSolver' instead, "
                "and handle nonlinearities through sweeping."
            )

    return solver


def _build_passive_eqs(
    mp: Any,
    c: Any,
    mesh: Mesh,
    D_mol: Any,
    bc_map: Dict[str, Any],
    species_list_partial: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build the invariant parts of the species equations (transport and transient).

    Returns:
    --------
    species_struct : list of dicts containing variable references
    passive_eqs : dict mapping species names to their partial FiPy equations
    """
    species_struct = []
    passive_eqs = {}

    from fipy import CellVariable
    from fipy.terms.diffusionTerm import DiffusionTerm
    from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
    from fipy.terms.powerLawConvectionTerm import PowerLawConvectionTerm
    from fipy.terms.upwindConvectionTerm import UpwindConvectionTerm
    from fipy.terms.transientTerm import TransientTerm

    # Ensure mp.phi is a CellVariable even if provided as a float
    # We create it once outside the loop to avoid recreating it for each species
    if not isinstance(mp.phi, CellVariable):
        phi_var = CellVariable(mesh=mesh, value=mp.phi)
        mp.phi = phi_var
    else:
        phi_var = mp.phi

    solid_scheme = getattr(mp, "solid_convection_term", "powerlaw")
    enable_caching = getattr(mp, "enable_matrix_caching", False)

    for name in species_list_partial:
        var = getattr(c, name)
        props = bc_map[name]

        species_struct.append({"name": name, "var": var})

        # Effective porosity for conservative form
        eff_phi = phi_var if props["type"] == "dissolved" else (1.0 - phi_var)

        # Diffusion coefficient (Molecular + Bio-diffusion)
        D_total = np.maximum(getattr(D_mol, name) + D_mol.D_bio, 1e-20)

        # Advection velocity
        vel = getattr(mp, "w", 0.0) - getattr(mp, "advection", 0.0) if props["type"] == "dissolved" else getattr(mp, "w", 0.0)
        u_var = CellVariable(mesh=mesh, value=vel, rank=1)

        # Terms with conservative phi handling
        if solid_scheme == "upwind" and props["type"] != "dissolved":
            conv_term = UpwindConvectionTerm(coeff=eff_phi * u_var, var=var)
        else:
            conv_term = PowerLawConvectionTerm(coeff=eff_phi * u_var, var=var)

        diff_term = DiffusionTerm(
            coeff=eff_phi * CellVariable(mesh=mesh, value=D_total), var=var
        )

        # Irrigation (Sources/Sinks for dissolved species)
        irr_term = 0.0
        if props["type"] == "dissolved":
            irr_term = ImplicitSourceTerm(
                coeff=eff_phi * CellVariable(mesh=mesh, value=-D_mol.D_irr), var=var
            ) + eff_phi * CellVariable(mesh=mesh, value=D_mol.D_irr * props["top"])

        if enable_caching:
            # Pre-compute and freeze convection face weights to bypass redundant Peclet loops
            cached_weight = conv_term._getWeight(var, None, None)
            conv_term._getWeight = lambda v=var, tg=None, dg=None, w=cached_weight: w
            # Pre-compute diffusion coeffDict
            diff_term._calcCoeffDict(var)

        # Passive equation: Transient + Convection - Diffusion - Irrigation
        passive_eqs[name] = (
            TransientTerm(coeff=eff_phi, var=var) + conv_term - diff_term - irr_term
        )

    return species_struct, passive_eqs


def _setup_static_coupled_equation(
    mp: Any,
    c: Any,
    k: Any,
    mesh: Mesh,
    passive_eqs: Dict[str, Any],
    species_struct: List[Dict[str, Any]],
    diagenetic_reactions: Any,
    species_list_partial: List[str],
) -> Tuple[
    Any, Dict[str, Any], Dict[str, Any], Dict[str, List[Any]]
]:
    """
    Setup static coefficient variables and compile the coupled equation system once.
    """
    from fipy import CellVariable
    from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
    from .diff_lib import data_container

    f_dummy = data_container()

    # Discover the coupling structure using a dummy run
    mp.in_solver = True
    try:
        f_res, _ = diagenetic_reactions(mp, c, k, f=f_dummy)
    finally:
        mp.in_solver = False

    LHS_vars = {}
    RHS_vars = {}
    CROSS_vars = {}  # species -> list of CellVariable

    eqs = []
    for s_obj in species_struct:
        name = s_obj["name"]

        # Pre-allocate static variables
        LHS_vars[name] = CellVariable(mesh=mesh, value=0.0)
        RHS_vars[name] = CellVariable(mesh=mesh, value=0.0)
        CROSS_vars[name] = []

        # Build coupled off-diagonal terms
        cross_list = f_res.raw_CROSS.get(name, [])
        cross_term = 0.0
        for source_name, _ in cross_list:
            v_cross = CellVariable(mesh=mesh, value=0.0)
            CROSS_vars[name].append((v_cross, source_name))
            cross_term += ImplicitSourceTerm(coeff=v_cross, var=c[source_name])

        lhs_reaction = ImplicitSourceTerm(coeff=LHS_vars[name], var=s_obj["var"])
        eq = passive_eqs[name] == lhs_reaction + cross_term + RHS_vars[name]
        eqs.append(eq)

    coupled_eq = reduce(lambda a, b: a & b, eqs)
    return coupled_eq, LHS_vars, RHS_vars, CROSS_vars


def _update_static_coefficients(
    mp: Any,
    c: Any,
    k: Any,
    diagenetic_reactions: Any,
    LHS_vars: Dict[str, Any],
    RHS_vars: Dict[str, Any],
    CROSS_vars: Dict[str, List[Any]],
    species_list_partial: List[str],
) -> Dict[str, np.ndarray]:
    """
    Calculate new reaction rates and update the static coefficient variables in-place.
    """
    from .diff_lib import data_container

    class ArrayProxy:
        def __init__(self, val):
            self.value = val
        def __getattr__(self, name):
            return getattr(self.value, name)
        def __add__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(self.value + val)
        def __radd__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(val + self.value)
        def __sub__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(self.value - val)
        def __rsub__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(val - self.value)
        def __mul__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(self.value * val)
        def __rmul__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(val * self.value)
        def __truediv__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(self.value / val)
        def __rtruediv__(self, other):
            val = other.value if hasattr(other, 'value') else other
            return ArrayProxy(val / self.value)
        def __pow__(self, power):
            return ArrayProxy(self.value ** power)
        def __neg__(self):
            return ArrayProxy(-self.value)
        def __getitem__(self, idx):
            return self.value[idx]

    c_numpy = data_container({s: ArrayProxy(val.value) for s, val in c.items()})
    mp_numpy = data_container(mp)
    mp_numpy.phi = ArrayProxy(mp.phi.value)

    f_res = data_container()

    mp_numpy.in_solver = True
    try:
        f_res, RATES = diagenetic_reactions(mp_numpy, c_numpy, k, f=f_res)
    finally:
        mp_numpy.in_solver = False

    def get_val(val):
        if hasattr(val, "value"):
            return val.value
        return val

    for s in species_list_partial:
        # Update diagonal coefficient
        LHS_vars[s].setValue(get_val(f_res.raw_LHS[s]))
        # Update explicit source
        RHS_vars[s].setValue(get_val(f_res.raw_RHS[s]))
        # Update off-diagonal couplings
        cross_list = f_res.raw_CROSS[s]
        for (v_cross, _), (source_name, coeff) in zip(CROSS_vars[s], cross_list):
            v_cross.setValue(get_val(coeff))
    RATES_numpy = {key: get_val(val) for key, val in RATES.items()}
    return RATES_numpy


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
    mesh: Mesh,
    D_mol: Any,
    bc_map: Dict[str, Any],
    z: np.ndarray,
    plot_queue: Optional[Any] = None,
) -> Tuple[int, float]:
    """
    Solves the non-steady state ADR coupled system with advanced adaptive time stepping.

    This function splits the model into manageable steps:
    1. Pre-builds passive transport terms.
    2. Runs a time loop where reaction terms are updated and solved.
    3. Adapts the time step using a PID-controlled logic.
    """
    if getattr(mp, "use_direct_assembly", False):
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

    from .diff_lib import get_delta, get_total_delta, save_state

    start_wall = time.time()
    solver = _get_solver(mp)

    log_path = f"{mp.plot_name}.log"
    _log_file = open(log_path, "w", buffering=1)

    def _log(msg: str) -> None:
        print(msg, flush=True)
        _log_file.write(msg + "\n")

    # --- Initialize Adaptive Time Stepping ---
    dt_controller = AdaptiveDT(
        dt_min=mp.dt_min,
        dt_max=mp.dt_max,
        dt_initial=getattr(mp, "dt_init", mp.dt_min),
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

    # --- Initialize Rate-Change-Based Timestep Adaptation ---
    enable_rate_adaptation = getattr(mp, "enable_rate_adaptation", False)
    monitored_rate_species = getattr(mp, "monitored_rate_species", ["FeS", "TS2"])
    rate_threshold = getattr(mp, "rate_threshold", 1e-8)
    rate_adaptation_start_step = getattr(mp, "rate_adaptation_start_step", 3)
    enable_rate_magnitude_check = getattr(mp, "enable_rate_magnitude_check", False)
    rate_sign_min_change = float(getattr(mp, "rate_sign_min_change", 2e-8))
    rate_sign_min_consecutive_cells = int(getattr(mp, "rate_sign_min_consecutive_cells", 1))
    prev_rates = {}
    prev_rates_2 = {}
    prev_dt = getattr(mp, "dt_init", mp.dt_min)
    prev_dt_2 = prev_dt

    # --- Initialize Inner Sweeping Loop (Method 1: Picard / Newton Iteration) ---
    enable_inner_sweeping = getattr(mp, "enable_inner_sweeping", False)
    max_inner_sweeps = int(getattr(mp, "max_inner_sweeps", 15))
    inner_tol = float(getattr(mp, "inner_tol", 1e-3))
    near_convergence_tol = float(getattr(mp, "near_convergence_tol", 1.25))
    graceful_acceptance_tol = float(getattr(mp, "graceful_acceptance_tol", near_convergence_tol))
    inner_norm = getattr(mp, "inner_norm", "wrms")
    inner_relaxation = float(getattr(mp, "inner_relaxation", 1.0))
    enable_adaptive_damping = getattr(mp, "enable_adaptive_damping", True)
    inner_sweep_equilibrium = getattr(mp, "inner_sweep_equilibrium", True)
    adaptive_sweeps_dt = getattr(mp, "adaptive_sweeps_dt", True)
    sweep_target_optimal = int(getattr(mp, "sweep_target_optimal", 4))
    sweep_max_acceptable = int(getattr(mp, "sweep_max_acceptable", 7))
    prev_inner_sweeps_count = 1
    enable_velocity_governor = getattr(mp, "enable_velocity_governor", True)
    velocity_ema = None
    velocity_ema_alpha = float(getattr(mp, "velocity_ema_alpha", 0.2))
    enable_single_sweep_exit = getattr(mp, "enable_single_sweep_exit", False)
    sweep1_exit_tol = float(getattr(mp, "sweep1_exit_tol", 3.0))
    max_cross_coupling_ratio = float(getattr(mp, "max_cross_coupling_ratio", 0.05))

    # --- Near-Convergence Rate Oscillation Safeguard Settings ---
    near_conv_rate_species = getattr(mp, "near_conv_rate_species", ["FeS"])
    near_conv_rate_sign_min_change = float(getattr(mp, "near_conv_rate_sign_min_change", 1e-9))
    near_conv_rate_sign_min_cells = int(getattr(mp, "near_conv_rate_sign_min_cells", 10))

    # --- Early Bail-Out & Aggressive Non-Linear Cut Settings ---
    enable_early_bailout = getattr(mp, "enable_early_bailout", True)
    early_bailout_iter = int(getattr(mp, "early_bailout_iter", 5))
    early_bailout_err = float(getattr(mp, "early_bailout_err", 3.0))
    enable_aggressive_cut = getattr(mp, "enable_aggressive_cut", True)
    aggressive_cut_threshold = float(getattr(mp, "aggressive_cut_threshold", 3.0))
    aggressive_cut_factor = float(getattr(mp, "aggressive_cut_factor", 0.25))

    # --- Relative Porewater Depletion Governor Settings ---
    enable_porewater_governor = getattr(mp, "enable_porewater_governor", enable_inner_sweeping)
    max_rel_porewater_change = float(getattr(mp, "max_rel_porewater_change", 0.25))
    porewater_conc_scale = float(getattr(mp, "porewater_conc_scale", 1e-4))
    porewater_conc_presence_floor = float(getattr(mp, "porewater_conc_presence_floor", 1e-3))

    # --- Initialize Dynamic Isotope dt Limiter ---
    enable_isotope_dt_limiter = getattr(mp, "enable_isotope_dt_limiter", False)
    isotope_limiter_species = getattr(mp, "isotope_limiter_species", "FeS")
    isotope_onset_threshold = getattr(mp, "isotope_onset_threshold", 1e-5)

    if enable_isotope_dt_limiter and getattr(mp, "isotopes", False):
        dt_max_isotope = _calculate_dt_max_isotope(mp, k, D_mol)
        mp.dt_max_isotope = dt_max_isotope
        w = max(getattr(mp, "w", 0.0), 1e-20)
        dz = getattr(mp, "reaction_zone_spacing", 0.0001)
        msg = f"Isotope dt Limiter: Calculated dt_max_isotope = {get_time_units(dt_max_isotope):.4f~P} (dz = {dz*1000:.3f} mm, w = {w*3.1536e7*100:.3f} cm/yr)"
        _log(msg)
        print(msg)

    # Global CFL bound (optional safety)
    dx = mesh.cellVolumes.min() ** (1 / mesh.dim)
    v_max = max(abs(getattr(mp, "w", 0.0)), abs(getattr(mp, "advection", 0.0)))
    D_max = 0.0
    for s in species_list_partial:
        D_s = np.max(getattr(D_mol, s) + D_mol.D_bio)
        D_max = max(D_max, D_s)

    dt_cfl = dt_controller.cfl_limit(dx, v_max, D_max)

    print(
        f"Starting Adaptive ADR Solver. dt_init: {get_time_units(dt_controller.dt):.2f~P}"
    )

    if getattr(mp, "enable_matrix_caching", False):
        msg_cache = "  [Solver] Transport operator caching enabled (stationary D, phi, w)."
        _log(msg_cache)
        print(msg_cache)
    if getattr(mp, "solid_convection_term", "powerlaw") == "upwind":
        msg_conv = "  [Solver] Upwind convection enabled for solid species."
        _log(msg_conv)
        print(msg_conv)

    # Build the transport backbone
    species_struct, passive_eqs = _build_passive_eqs(
        mp, c, mesh, D_mol, bc_map, species_list_partial
    )

    # Setup static coupled equation and coefficient variables (Option C)
    coupled_eq, LHS_vars, RHS_vars, CROSS_vars = _setup_static_coupled_equation(
        mp,
        c,
        k,
        mesh,
        passive_eqs,
        species_struct,
        diagenetic_reactions,
        species_list_partial,
    )

    step = 0
    total_sweeps = 0
    total_time = mp.start_time
    last_report_wall = start_wall
    last_report_time = total_time
    status = "Maximum steps or simulation time reached"
    max_change = 0.0
    title_str = ""
    prev_was_near_converged = False

    try:
        while total_time < mp.t_end and step < mp.max_steps:
            step += 1
            step_start_wall = time.time()
            current_dt = dt_controller.dt
            step_first_attempt = True

            # updateOld() stores current -> var.old; used below for RMS and restore
            for s_obj in species_struct:
                s_obj["var"].updateOld()

            # --- Solve Step (with automatic retry on failure) ---
            converged = False
            near_converged_accepted = False
            graceful_accepted = False
            RATES_tentative = {}
            last_inner_sweeps = 1
            last_inner_err = 0.0
            while not converged:
                mp.current_dt = current_dt
                try:
                    if enable_inner_sweeping:
                        inner_converged = False
                        current_theta = inner_relaxation
                        prev_inner_err = None

                        for inner_iter in range(1, max_inner_sweeps + 1):
                            # Cache iterate value before updating
                            prev_iterate = {
                                s_obj["name"]: np.copy(s_obj["var"].value)
                                for s_obj in species_struct
                            }

                            # Update static coefficient variables in-place using current iterate
                            RATES = _update_static_coefficients(
                                mp,
                                c,
                                k,
                                diagenetic_reactions,
                                LHS_vars,
                                RHS_vars,
                                CROSS_vars,
                                species_list_partial,
                            )

                            coupled_eq.sweep(
                                dt=current_dt,
                                solver=solver,
                            )
                            total_sweeps += 1

                            # Compute raw error of this solve
                            raw_inner_err = _compute_inner_residual(
                                species_struct, prev_iterate, inner_tol=inner_tol, inner_norm=inner_norm
                            )

                            # Adaptive Damping: detect oscillation or error growth
                            if enable_adaptive_damping and inner_iter >= 2 and prev_inner_err is not None:
                                if raw_inner_err > prev_inner_err:
                                    # Residual grew: damp the update to arrest oscillation
                                    current_theta = max(current_theta * 0.7, 0.4)
                                elif raw_inner_err <= prev_inner_err * 0.8 and current_theta < inner_relaxation:
                                    # Monotonic contraction: relax theta back towards base relaxation
                                    current_theta = min(current_theta * 1.15, inner_relaxation)

                            # Apply under-relaxation if theta < 1.0
                            if current_theta < 1.0:
                                for s_obj in species_struct:
                                    s_name = s_obj["name"]
                                    s_obj["var"].setValue(
                                        (1.0 - current_theta) * prev_iterate[s_name]
                                        + current_theta * s_obj["var"].value
                                    )
                                last_inner_err = _compute_inner_residual(
                                    species_struct, prev_iterate, inner_tol=inner_tol, inner_norm=inner_norm
                                )
                            else:
                                last_inner_err = raw_inner_err

                            # Intermediate equilibrium projection
                            if inner_sweep_equilibrium:
                                mp.in_clip = True
                                try:
                                    equilibrium_reactions(mp, c, k, None, RATES, current_dt)
                                finally:
                                    mp.in_clip = False

                            # Check inner convergence
                            if last_inner_err <= 1.0:
                                cross_dominated = False
                                if inner_iter == 1 and max_cross_coupling_ratio > 0.0:
                                    cross_dominated = _check_cross_coupling_dominance(
                                        CROSS_vars,
                                        c,
                                        current_dt,
                                        species_struct,
                                        threshold=max_cross_coupling_ratio,
                                    )
                                if not cross_dominated:
                                    inner_converged = True
                                    last_inner_sweeps = inner_iter
                                    break

                            # Check 1-sweep early exit during smooth evolution
                            if (
                                inner_iter == 1
                                and enable_single_sweep_exit
                                and not prev_was_near_converged
                                and last_inner_err <= sweep1_exit_tol
                            ):
                                cross_dominated = False
                                if max_cross_coupling_ratio > 0.0:
                                    cross_dominated = _check_cross_coupling_dominance(
                                        CROSS_vars,
                                        c,
                                        current_dt,
                                        species_struct,
                                        threshold=max_cross_coupling_ratio,
                                    )
                                if not cross_dominated:
                                    inner_converged = True
                                    last_inner_sweeps = 1
                                    break

                            # Early bail-out check: detect hopeless or diverging iterations early
                            if enable_early_bailout and inner_iter >= early_bailout_iter:
                                is_diverging = (
                                    prev_inner_err is not None
                                    and last_inner_err > prev_inner_err * 1.2
                                    and last_inner_err > 1.5
                                )
                                is_hopeless = (
                                    last_inner_err > early_bailout_err * 3.0
                                    or (
                                        last_inner_err > early_bailout_err
                                        and (prev_inner_err is not None and last_inner_err >= prev_inner_err * 0.95)
                                    )
                                )
                                if is_diverging or is_hopeless:
                                    reason = "diverging residual" if is_diverging else "stagnant/hopeless residual"
                                    raise RuntimeError(
                                        f"Early bailout at sweep {inner_iter}/{max_inner_sweeps} due to {reason} "
                                        f"(scaled_err={last_inner_err:.2e} vs threshold={early_bailout_err})"
                                    )

                            prev_inner_err = last_inner_err

                        if not inner_converged:
                            # 1. Check for rate oscillation before accepting near-convergence / graceful exit
                            rate_oscillation_detected = False
                            osc_reason = ""
                            if prev_rates:
                                RATES_tentative = _update_static_coefficients(
                                    mp,
                                    c,
                                    k,
                                    diagenetic_reactions,
                                    LHS_vars,
                                    RHS_vars,
                                    CROSS_vars,
                                    species_list_partial,
                                )
                                for r_name in near_conv_rate_species:
                                    if r_name in RATES_tentative and r_name in prev_rates:
                                        r_tent = np.asarray(RATES_tentative[r_name])
                                        r_pr = np.asarray(prev_rates[r_name])
                                        rate_diff = np.abs(r_tent - r_pr)

                                        # 3-point temporal reversal check: only flag true oscillations
                                        # (r_pr_2 -> r_pr flipped AND r_pr -> r_tent flipped).
                                        # A monotonic one-way transition (e.g. advancing reaction/burial front)
                                        # has (r_pr * r_pr_2 >= 0) and is therefore not flagged.
                                        if r_name in prev_rates_2:
                                            r_pr_2 = np.asarray(prev_rates_2[r_name])
                                            sign_change_mask = (
                                                (r_tent * r_pr < 0)
                                                & (r_pr * r_pr_2 < 0)
                                                & (rate_diff >= near_conv_rate_sign_min_change)
                                                & (np.abs(r_tent) >= 1e-11)
                                                & (np.abs(r_pr) >= 1e-11)
                                            )
                                        else:
                                            sign_change_mask = np.zeros_like(r_tent, dtype=bool)
                                        has_consec, cell_idx, consec_len = _find_consecutive_trues(
                                            sign_change_mask, near_conv_rate_sign_min_cells
                                        )
                                        if has_consec:
                                            rate_oscillation_detected = True
                                            osc_reason = (
                                                f"Rate sign change (oscillation) in {r_name} across {consec_len} consecutive cells "
                                                f"starting at cell {cell_idx} (prev rate: {r_pr[cell_idx]:.2e}, "
                                                f"tentative rate: {r_tent[cell_idx]:.2e}, rate change: {rate_diff[cell_idx]:.2e} mol/(m^3*s))"
                                            )
                                            break

                            if rate_oscillation_detected:
                                raise RuntimeError(
                                    f"Near-convergence rejected at sweep {max_inner_sweeps}: {osc_reason}."
                                )

                            if last_inner_err <= near_convergence_tol:
                                inner_converged = True
                                near_converged_accepted = True
                                last_inner_sweeps = max_inner_sweeps
                                _log(
                                    f"[{_format_wall_time(time.time() - start_wall)}]   Near-convergence accepted at sweep {max_inner_sweeps} "
                                    f"(scaled_err={last_inner_err:.2f} <= near_tol={near_convergence_tol:.2f})."
                                )
                            elif last_inner_err <= graceful_acceptance_tol:
                                inner_converged = True
                                graceful_accepted = True
                                last_inner_sweeps = max_inner_sweeps
                                _log(
                                    f"[{_format_wall_time(time.time() - start_wall)}]   Graceful acceptance at sweep {max_inner_sweeps} "
                                    f"(scaled_err={last_inner_err:.2f} <= graceful_tol={graceful_acceptance_tol:.2f}). Bypassing step failure."
                                )
                            else:
                                raise RuntimeError(
                                    f"Inner Picard sweep failed to converge in {max_inner_sweeps} iterations (scaled_err={last_inner_err:.2e})"
                                )

                        if not inner_sweep_equilibrium:
                            mp.in_clip = True
                            try:
                                equilibrium_reactions(mp, c, k, None, RATES, current_dt)
                            finally:
                                mp.in_clip = False
                    else:
                        # Legacy single sweep
                        RATES = _update_static_coefficients(
                            mp,
                            c,
                            k,
                            diagenetic_reactions,
                            LHS_vars,
                            RHS_vars,
                            CROSS_vars,
                            species_list_partial,
                        )

                        coupled_eq.sweep(
                            dt=current_dt,
                            solver=solver,
                        )
                        total_sweeps += 1

                        mp.in_clip = True
                        try:
                            equilibrium_reactions(mp, c, k, None, RATES, current_dt)
                        finally:
                            mp.in_clip = False

                    converged = True

                    if (enable_rate_adaptation or enable_inner_sweeping) and not RATES_tentative:
                        RATES_tentative = _update_static_coefficients(
                            mp,
                            c,
                            k,
                            diagenetic_reactions,
                            LHS_vars,
                            RHS_vars,
                            CROSS_vars,
                            species_list_partial,
                        )

                except Exception as e:
                    tb_str = "".join(
                        traceback.format_exception(type(e), e, e.__traceback__)
                    )
                    _log(
                        f"[{_format_wall_time(time.time() - start_wall)}]   Step failed at dt={get_time_units(current_dt):.2f~P}: {e}\n  Cutting dt and retrying."
                    )
                    # Restore state from FiPy's built-in old-value store
                    for s_obj in species_struct:
                        s_obj["var"].value[:] = s_obj["var"].old.value

                    # Determine cut factor: aggressive cut on non-linear divergence
                    use_aggressive_cut = (
                        enable_inner_sweeping
                        and enable_aggressive_cut
                        and (last_inner_err > aggressive_cut_threshold)
                    )
                    effective_cut = (
                        aggressive_cut_factor
                        if use_aggressive_cut
                        else dt_controller.cut_factor
                    )
                    if use_aggressive_cut:
                        _log(
                            f"[{_format_wall_time(time.time() - start_wall)}]   Aggressive dt cut ({effective_cut:.2f}x) triggered: "
                            f"scaled error {last_inner_err:.2e} exceeds threshold {aggressive_cut_threshold:.2e}."
                        )

                    # Cut time step and retry
                    prev_was_near_converged = True
                    if step_first_attempt:
                        current_dt = dt_controller.register_failure(current_dt, cut_factor=effective_cut)
                        step_first_attempt = False
                        if dt_controller.enable_failure_ceiling and dt_controller._dt_ceiling is not None:
                            _log(
                                f"[{_format_wall_time(time.time() - start_wall)}]   Failure ceiling activated: dt capped at {get_time_units(dt_controller._dt_ceiling):.2f~P} for at least {dt_controller.failure_hold_steps} steps."
                            )
                    else:
                        current_dt = dt_controller.update(0.0, step_success=False, cut_factor=effective_cut)
                    if current_dt <= mp.dt_min * 1.01:
                        raise RuntimeError(
                            "Solver failed and time step is already at minimum."
                        )

                # --- Rate Validation Check (A Posteriori) ---
                if converged and enable_rate_adaptation and step >= rate_adaptation_start_step and prev_rates:
                    violation, violation_reason = _validate_rates(
                        monitored_rate_species,
                        RATES_tentative,
                        prev_rates,
                        prev_rates_2,
                        current_dt,
                        prev_dt,
                        prev_dt_2,
                        rate_threshold,
                        enable_rate_magnitude_check,
                        rate_sign_min_change=rate_sign_min_change,
                        rate_sign_min_consecutive_cells=rate_sign_min_consecutive_cells,
                    )
                    if violation:
                        _log(f"[{_format_wall_time(time.time() - start_wall)}]   Step rejected at dt={get_time_units(current_dt):.4f~P}: {violation_reason}. Rollback.")
                        for s_obj in species_struct:
                            s_obj["var"].value[:] = s_obj["var"].old.value
                        
                        converged = False
                        if step_first_attempt:
                            current_dt = dt_controller.register_failure(current_dt)
                            step_first_attempt = False
                            if dt_controller.enable_failure_ceiling and dt_controller._dt_ceiling is not None:
                                _log(
                                    f"[{_format_wall_time(time.time() - start_wall)}]   Failure ceiling activated: dt capped at {get_time_units(dt_controller._dt_ceiling):.2f~P} for at least {dt_controller.failure_hold_steps} steps."
                                )
                        else:
                            current_dt = dt_controller.update(0.0, step_success=False)
                        if current_dt <= mp.dt_min * 1.01:
                            raise RuntimeError(
                                "Rate validation failed and time step is already at minimum."
                            )

            # --- Calculate Convergence Metrics ---
            rms_change = max(
                float(
                    np.sqrt(np.mean((s_obj["var"].value - s_obj["var"].old.value) ** 2))
                )
                for s_obj in species_struct
            )

            # --- Dynamically adapt solver tolerance to prevent false steady-state convergence ---
            if getattr(mp, "adaptive_solver_tolerance", False):
                new_tol = max(mp.tolerance, min(1e-4, rms_change * 0.1))
                solver.tolerance = new_tol

            total_time += current_dt
            if not (near_converged_accepted or graceful_accepted):
                dt_controller.record_success()
            else:
                # Near-converged or gracefully accepted steps are at the edge of stability:
                # Reset consecutive failure-free streak so MTTF does not boost growth_factor
                dt_controller.steps_since_failure = 0
            prev_was_near_converged = near_converged_accepted or graceful_accepted

            # --- Update Rate History for Next Step ---
            if enable_rate_adaptation or enable_inner_sweeping:
                prev_dt_2 = prev_dt
                prev_dt = current_dt
                for name in monitored_rate_species:
                    if name in RATES_tentative:
                        if name in prev_rates:
                            prev_rates_2[name] = np.copy(prev_rates[name])
                        prev_rates[name] = np.copy(RATES_tentative[name])

            # --- Adapt Time Step for Next Iteration ---
            if enable_inner_sweeping and adaptive_sweeps_dt:
                effective_max = dt_controller.get_effective_max(_log=_log)
                if graceful_accepted:
                    # Gracefully accepted at sweep ceiling: damp dt (0.85x) to guide solver back to lower sweeps
                    dt_controller._dt = max(dt_controller._dt * 0.85, dt_controller.dt_min)
                elif near_converged_accepted:
                    # Near-convergence accepted: gently damp dt (0.90x) to guide solver back to optimal sweeps
                    dt_controller._dt = max(dt_controller._dt * 0.90, dt_controller.dt_min)
                elif last_inner_sweeps <= sweep_target_optimal:
                    # Healthy, fast convergence in optimal sweeps -> grow dt
                    if prev_inner_sweeps_count > sweep_max_acceptable:
                        growth = 1.02
                    else:
                        growth = dt_controller.growth_factor
                    dt_controller._dt = min(dt_controller._dt * growth, effective_max)
                elif last_inner_sweeps <= sweep_max_acceptable:
                    # Moderate sweeps -> hold steady or mild growth (1.02x)
                    dt_controller._dt = min(dt_controller._dt * 1.02, effective_max)
                else:
                    # Approaching sweep limit -> damp dt to stay in comfortable zone
                    dt_controller._dt = max(dt_controller._dt * 0.85, dt_controller.dt_min)

                # Simulation Velocity Governor: detect throughput degradation
                if enable_velocity_governor:
                    step_wall = max(time.time() - step_start_wall, 1e-4)
                    v_step = current_dt / step_wall
                    if velocity_ema is None:
                        velocity_ema = v_step
                    else:
                        # Option B: Only damp dt if throughput dropped due to retried step failures
                        # or exceeding acceptable sweeps (do not penalize normal multi-sweep convergence)
                        if v_step < 0.75 * velocity_ema and (not step_first_attempt or last_inner_sweeps > sweep_max_acceptable):
                            dt_controller._dt = max(dt_controller._dt * 0.90, dt_controller.dt_min)
                        velocity_ema = (1.0 - velocity_ema_alpha) * velocity_ema + velocity_ema_alpha * v_step

                dt_controller._dt_prev = dt_controller._dt
                prev_inner_sweeps_count = last_inner_sweeps
            elif enable_rate_adaptation:
                effective_max = dt_controller.get_effective_max(_log=_log)
                dt_controller._dt = min(dt_controller._dt * dt_controller.growth_factor, effective_max)
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
                wall_str = f"[{_format_wall_time(time.time() - start_wall)}] "
                adapted_dt, obs_rel, lim_sp, lim_cell = apply_porewater_depletion_governor(
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
                        dt_controller._dt = min(dt_controller._dt, dt_max_isotope)
                        dt_controller._dt_prev = min(dt_controller._dt_prev, dt_max_isotope)

            if step % mp.backup_step == 0:
                gc.collect()
                save_state(c, f"{mp.plot_name}_bak.npz")

            # Reporting
            if step % mp.report_step == 0:
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
                dt = get_time_units(mp.dt_max)
                if hasattr(c, "Fe3") and hasattr(c, "Fe2_total"):
                    Fe3_lost = c.Fe3.value[0] - c.Fe3.value[-1]
                    Fe2_gained = c.Fe2_total.value[-1] - c.Fe2_total.value[0]
                    _log(
                        f"dt = {dt:~P.2f}, Fe3_lost = {Fe3_lost:.2f}, Fe2_gained = {Fe2_gained:.2f}"
                    )
                else:
                    _log(f"dt = {dt:~P.2f}")
                break

    except KeyboardInterrupt:
        status = "Solver interrupted by user"
    except Exception as e:
        status = f"Solver crashed: {e}"
        print(traceback.format_exc())

    # Final Save
    elapsed_total = time.time() - start_wall
    sweeps_rate_str = f", {total_sweeps / max(elapsed_total, 1e-6):.2f} swp/s" if total_sweeps > 0 else ""
    _log(
        f"Final Report: {status} in {step} steps ({total_sweeps} total sweeps{sweeps_rate_str}). Total Wall Time: {_format_wall_time(elapsed_total)} ({elapsed_total:.2f}s)"
    )
    _log_file.close()
    _compress_log_file(log_path)

    # Always write the final data and state synchronously to prevent data loss on termination/interrupt
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

    return step, rms_change
