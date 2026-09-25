"""
Utility library for the fipyrite package.

This package provides a small collection of functions and a lightweight container class
used throughout the Pyrite Burial model.

The module contains:

    - :class:=data_container= – a simple container that can be initialised from a
      space‑separated string of attribute names with optional default values, or from a
      dictionary mapping attribute names to values.

    - :func:=diff_coeff= – computes the diffusion coefficient (m² s⁻¹) for a given
      temperature (°C), porosity (percent) and the linear parameters /m0/ and /m1/ from
      Boudreau (1996).

    - :func:=get_delta= – calculates the isotopic delta value (‰) from the total
      concentration of an isotope pair and a reference ratio.

    - :func:=get_l_mass= – derives the concentration of the light isotope from a
      measured total concentration, a delta value and the reference ratio.

    - :func:=relax_solution= – blends a current solution vector with a previous one,
      limiting the change to a specified fraction and enforcing non‑negative values.

These helpers are primarily intended for modelling isotope diffusion and fractionation
processes in geological simulations.
"""

import numpy as np


class data_container(dict):
    """A dictionary-based container with attribute access.

    Supports initialization from a space-separated string or a dictionary.
    """

    def __init__(self, names=None, defaults=None):
        super().__init__()
        if isinstance(names, str):
            names = names.split(" ")
            if isinstance(defaults, list):
                for i, name in enumerate(names):
                    if name != "":
                        self[name] = defaults[i]
            else:
                for name in names:
                    if name != "":
                        self[name] = defaults
        elif isinstance(names, dict):
            self.update(names)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )


class FaceSelector:
    """Mock for mesh.facesLeft and mesh.facesRight."""

    def __init__(self, index: int, size: int):
        self.value = np.zeros(size, dtype=bool)
        self.value[index] = True


class Mesh1D:
    """Lightweight 1D non-uniform mesh replacing fipy.Grid1D."""

    def __init__(self, dx):
        self.dx = np.asarray(dx, dtype=np.float64)
        self.cellVolumes = self.dx
        self.numberOfCells = len(self.dx)
        self.faceCoordinates = np.concatenate(([0.0], np.cumsum(self.dx)))
        self.cellCenters = np.array([(self.faceCoordinates[:-1] + self.faceCoordinates[1:]) / 2.0])
        self.faceCenters = np.array([self.faceCoordinates])
        self._cellDistances = np.concatenate((
            [self.dx[0] / 2.0],
            (self.dx[:-1] + self.dx[1:]) / 2.0,
            [self.dx[-1] / 2.0],
        ))
        self.facesLeft = FaceSelector(0, self.numberOfCells + 1)
        self.facesRight = FaceSelector(self.numberOfCells, self.numberOfCells + 1)


class ArrayProxy:
    """Lightweight NumPy array wrapper mimicking FiPy CellVariable for rate evaluations."""

    def __init__(self, val):
        self.value = np.asarray(val, dtype=np.float64)

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __add__(self, other):
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(self.value + val)

    def __radd__(self, other):
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(val + self.value)

    def __sub__(self, other):
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(self.value - val)

    def __rsub__(self, other):
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(val - self.value)

    def __mul__(self, other):
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(self.value * val)

    def __rmul__(self, other):
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(val * self.value)

    def __truediv__(self, other):
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(self.value / val)

    def __rtruediv__(self, other):
        val = other.value if hasattr(other, "value") else other
        return ArrayProxy(val / self.value)

    def __pow__(self, power):
        return ArrayProxy(self.value**power)

    def __neg__(self):
        return ArrayProxy(-self.value)

    def __pos__(self):
        return self

    def __abs__(self):
        return ArrayProxy(abs(self.value))

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def __getitem__(self, idx):
        return self.value[idx]

    def __setitem__(self, idx, val):
        self.value[idx] = val

    def __len__(self):
        return len(self.value)


class FaceValueProxy(ArrayProxy):
    """Array proxy for face-centered fields supporting FaceSelector indexing."""

    def __getitem__(self, idx):
        if hasattr(idx, "value"):
            idx = idx.value
        return super().__getitem__(idx)


class VariableArray(ArrayProxy):
    """Lightweight NumPy array wrapper mimicking FiPy CellVariable for state & rates."""

    def __init__(self, value=0.0, name="", mesh=None, hasOld=True, **kwargs):
        val = kwargs.get("val", value)
        val_arr = val.value if hasattr(val, "value") else val
        if np.ndim(val_arr) == 0:
            if mesh is not None:
                n_cells = getattr(mesh, "numberOfCells", len(getattr(mesh, "dx", [])))
                val_arr = np.full(n_cells, float(val_arr), dtype=np.float64)
            else:
                val_arr = np.array([float(val_arr)], dtype=np.float64)
        super().__init__(val_arr)
        self.name = name
        self.mesh = mesh
        self.hasOld = hasOld
        self.old = ArrayProxy(self.value.copy())

    def setValue(self, val, where=None):
        if where is not None:
            mask = where.value if hasattr(where, "value") else where
            if isinstance(mask, np.ndarray) and mask.dtype == bool and len(mask) == len(self.value) + 1:
                if mask[0]:
                    self.value[0] = np.asarray(val)
                if mask[-1]:
                    self.value[-1] = np.asarray(val)
            else:
                self.value[mask] = np.asarray(val)
        else:
            self.value[:] = np.asarray(val)

    def updateOld(self):
        self.old.value[:] = self.value

    @property
    def faceValue(self):
        if len(self.value) == 0:
            return FaceValueProxy(self.value)
        faces = np.empty(len(self.value) + 1, dtype=np.float64)
        faces[0] = self.value[0]
        faces[1:-1] = 0.5 * (self.value[:-1] + self.value[1:])
        faces[-1] = self.value[-1]
        return FaceValueProxy(faces)

    @property
    def faceGrad(self):
        return self

    def constrain(self, *args, **kwargs):
        pass


def diff_coeff(T, m0, m1, phi):
    """Calculate the diffusion coeefficien in m^2/s.

    T: temperature in C
    phi: porosity in percent
    m0, m1: parameter as from table X in Boudreau 1996
    """
    # If phi is a FiPy CellVariable, use its .value for numpy math
    phi_val = getattr(phi, "value", phi)
    return (m0 + m1 * T) * 1e-10 / (1 - np.log(phi_val**2))


def get_delta(c, li, r):
    """Calculate the delta from the mass of light and heavy isotope.

    :param li: light isotope mass/concentration
    :param h: heavy isotope mass/concentration
    :param r: reference ratio

    :return : delta

    """
    #    import numpy

    with np.errstate(divide="ignore", invalid="ignore"):
        # 1. Numerical Safeguards
        # Ensure total mass 'c' is at least 'li' (light isotope) to avoid negative heavy mass
        # and prevent division by very near zero or zero.
        li_safe = np.maximum(li, 1e-30)
        c_safe = np.maximum(c, li_safe)

        h = c_safe - li_safe
        ratio = h / li_safe

        # 2. Thresholding for NaN
        # If light isotope concentration is below 1 nmol/L (1e-6 mmol/L), delta is undefined (NaN)
        d = np.where(li < 1e-6, np.nan, 1000 * (ratio - r) / r)

        # 3. Clipping Extreme Values
        d = np.clip(d, -1000.0, 1000.0)

    return d


def get_delta_from_concentration(c, li, r):
    """Calculate the delta from the mass of light and heavy isotope.

    :param c: total mass/concentration
    :param li: light isotope mass/concentration
    :param r: reference ratio

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        li_safe = np.maximum(li, 1e-30)
        c_safe = np.maximum(c, li_safe)
        h = c_safe - li_safe
        ratio = h / li_safe
        d = np.where(li < 1e-6, np.nan, 1000 * (ratio - r) / r)
        d = np.clip(d, -1000.0, 1000.0)

    return d


def get_l_mass(m, d, r):
    """Calculate the light isotope mass from the total mass and delta.

    :param m: total mass/concentration
    :param d: delta
    :param r: reference ratio

    :return: light isotope mass
    """
    return (1000.0 * m) / ((d + 1000.0) * r + 1000.0)


def get_total_s_export(df, VCDT=0.044162589):
    """Calculate the total sulfur and delta34S at the bottom of the column.

    Uses the logic as defined in run_fipy.py.
    """
    phi = df.phi.iloc[-1]
    s = phi * (df.c_SO4.iloc[-1] + df.c_h2s.iloc[-1]) + (1 - phi) * (
        df.c_S0.iloc[-1] + df.c_FeS.iloc[-1] + 2 * df.c_FeS2.iloc[-1]
    )
    s32 = phi * (df.c_SO4_32.iloc[-1] + df.c_h2s_32.iloc[-1]) + (1 - phi) * (
        df.c_S0_32.iloc[-1] + df.c_FeS_32.iloc[-1] + df.c_FeS2_32.iloc[-1]
    )
    h = s - s32
    d34s = np.where(s32 < 0.001, np.nan, 1000 * (h / s32 - VCDT) / VCDT)
    return float(s), float(d34s)


def relax_solution(curr_sol, last_sol, fraction):
    """Blend two solution vectors.

    In such away that they only chance by a given fraction
    """
    sol = last_sol * (1 - fraction) + curr_sol * fraction
    return sol * (sol >= 0)  # exclude negative solutions


def bioturbation_profile(z, D_max, cutoff_depth, threshold=1e-12):
    """
    Calculate the necessary steepness.

    To fit a mixing profile into 'cutoff_depth', ensuring D_max at surface and
    'threshold' at cutoff.
    """
    # 1. Safety Checks
    if cutoff_depth <= 0 or D_max <= threshold:
        return np.zeros_like(z)

    # 2. Calculate the Magnitude of the Drop
    # We need to drop from D_max down to threshold.
    # ratio represents the magnitude of this drop (e.g., 10,000,000x)
    ratio = (D_max / threshold) - 1
    log_ratio = np.log(ratio)

    # 3. Dynamic Steepness Calculation
    # We want the "slide" to start at z=0 and finish exactly at z=cutoff_depth.
    # To fit the full drop (log_ratio) into the distance (cutoff_depth):
    # k = total_drop_in_log_units / distance
    # We add a safety factor (e.g., 1.1) to ensure the 'shoulder' is slightly below surface
    calculated_steepness = (log_ratio / cutoff_depth) * 1.05

    # 4. Calculate Inflection Point
    # The inflection point is where the value is half of D_max.
    # Based on the calculated steepness, we shift the curve so the tail hits
    # 'threshold' exactly at 'cutoff_depth'.
    shift = log_ratio / calculated_steepness
    inflection_point = cutoff_depth - shift

    # 5. Generate Sigmoid
    sigmoid = D_max / (1 + np.exp(calculated_steepness * (z - inflection_point)))

    # 6. Hard Clamp
    sigmoid[z > cutoff_depth] = 0.0

    return sigmoid


def bioturbation_profile_2(z, D_max, cutoff_depth, threshold=1e-12):
    """
    Generate a sigmoid mixing profile.

    That fits EXACTLY within 'cutoff_depth'.  Calculates dynamic steepness so shallow
    depths get sharper curves automatically.
    """
    # Safety: If depth is zero or D_max is negligible, return zeros
    if cutoff_depth <= 0 or D_max <= threshold:
        return np.zeros_like(z)

    # 1. Calculate how "steep" we need to be to drop from D_max to threshold
    #    within the allocated depth.
    #    formula: D_max / (1 + exp(k * z)) = threshold
    ratio = (D_max / threshold) - 1
    if ratio <= 0:
        return np.zeros_like(z)

    total_drop_log = np.log(ratio)

    # We want the drop to finish slightly before the cutoff (95% of depth)
    # to ensure the clamp is clean.
    effective_depth = cutoff_depth * 0.95
    calculated_steepness = total_drop_log / effective_depth

    # 2. Calculate the "Shift" (Inflection Point)
    #    We shift the curve so it starts flat at surface and drops at the end.
    #    Shifting by log(ratio)/k moves the tail to the cutoff.
    #    We shift slightly less to keep the 'shoulder' near the surface.
    shift = total_drop_log / calculated_steepness
    inflection_point = cutoff_depth - shift

    # 3. Generate Profile
    sigmoid = D_max / (1 + np.exp(calculated_steepness * (z - inflection_point)))

    # 4. Hard Clamp to ensure true zero below the cutoff
    sigmoid[z > cutoff_depth] = 0.0

    return sigmoid


def compute_sigmoidal_db(z, Db0, xL, xbm):
    """
    Compute the bio-diffusivity (Db) at a specific depth (z).

    Using Equation 4 from van de Velde and Meysman (2016).

    Parameters
    ----------
    z   : float or np.ndarray
          Depth into the sediment in m
    Db0 : float
          Bio-diffusivity coefficient m^2/s
    xL  : float
          Depth of the mixed layer in m
    xbm : float
          Attenuation coefficient determining the width of the transition zone [m]

    Returns
    -------
    float or np.ndarray: The bio-diffusivity at depth z.
    """
    z_cm = z * 100
    xL_cm = xL * 100
    xbm_cm = xbm * 100

    # Define the term used in the exponents
    exponent_term = -(z_cm - xL_cm) / (0.25 * xbm_cm)

    # Use a robust sigmoid formula to avoid overflow/NaN warnings
    # We clip the exponent to a range that won't overflow double precision floats (~700)
    # This still results in 0.0 or Db0 as expected at the limits.
    Db_z = Db0 / (1.0 + np.exp(np.clip(-exponent_term, -700, 700)))

    return Db_z


def compute_bio_irrigation_alpha(z, alpha0, x_irr):
    """
    Compute the bio-irrigation coefficient (alpha) at a specific depth (z).

    Using Equation 6 from van de Velde and Meysman (2016).

    Parameters
    ----------
    z     : float or np.ndarray
            Depth into the sediment (m).
    alpha0: float
            Bio-irrigation coefficient at the sediment-water interface(SWI).
    x_irr : float
            Attenuation coefficient determining the depth of irrigation (m).

    Returns
    -------
    float or np.ndarray
        The irrigation intensity at depth z.
    """
    z = z * 100
    x_irr = x_irr * 100
    # Equation 6: alpha(z) = alpha0 * exp(-z / x_irr)
    alpha_z = alpha0 * np.exp(-z / x_irr)

    return alpha_z


def make_grid(L, initial_spacing, max_spacing, r=1.05):
    """
    Construct a 1D grid driven by spacing constraints.

    Growth is geometric (at rate r) until max_spacing is reached.

    Args:
    -----
        L (float): Total length of the domain (max depth).
        initial_spacing (float): The size of the first cell (at z=0).
        max_spacing (float): Maximum cell size allowed.
        r (float): Geometric growth rate (default 1.05).

    Returns
    -------
        tuple (mesh, z_centers)
            mesh: A fipy.Grid1D object.
            z_centers: A numpy array of cell center coordinates.
    """
    if initial_spacing >= max_spacing:
        initial_spacing = max_spacing

    dx_list = []
    current_z = 0
    current_dx = initial_spacing

    # 1. Geometric Section
    while current_z + current_dx < L and current_dx < max_spacing:
        dx_list.append(current_dx)
        current_z += current_dx
        current_dx *= r

    # 2. Uniform Section
    if current_z < L:
        remaining_L = L - current_z
        n_uniform = int(np.ceil(remaining_L / max_spacing))
        if n_uniform > 0:
            actual_uniform_dx = remaining_L / n_uniform
            dx_list.extend([actual_uniform_dx] * n_uniform)

    dx_array = np.array(dx_list)
    N = len(dx_array)
    print(f"Grid generated with {N} points.")

    mesh = Mesh1D(dx=dx_array)
    z_centers = mesh.cellCenters[0]

    return mesh, z_centers


def save_data(mp, c, k, species_list, z, D_mol, diagenetic_reactions, equilibrium_reactions):
    """
    Save the model results to a CSV file (Synchronous).
    """
    c_numpy = data_container({s: ArrayProxy(val.value) for s, val in c.items()})
    mp_numpy = data_container(mp)
    mp_numpy.phi = ArrayProxy(mp.phi.value)
    f_final, RATES = diagenetic_reactions(mp_numpy, c_numpy, k, data_container())
    f_final, RATES = equilibrium_reactions(
        mp, c, k, f_final, RATES, getattr(mp, "current_dt", 0.0)
    )
    return _save_data_to_disk(mp, c, k, species_list, z, D_mol, f_final)


def _save_data_to_disk(mp, c, k, species_list, z, D_mol, f_final):
    """Internal helper to write data to disk."""
    import pandas as pd
    import pathlib as pl

    data = {"z": z}

    def get_v(obj):
        """Helper to extract value from FiPy variables or return as-is."""
        if hasattr(obj, "value"):
            return obj.value
        return obj

    # 1. Collect all basic concentrations and rates
    for species_name in species_list:
        data[f"c_{species_name}"] = get_v(getattr(c, species_name))
        res_tuple = getattr(f_final, species_name)
        data[f"f_{species_name}"] = get_v(res_tuple[2])

    # Export process-specific rates dynamically
    for key in f_final.keys():
        if key not in species_list:
            res_tuple = getattr(f_final, key)
            data[f"f_{key}"] = get_v(res_tuple[2])

    # 2. Save all items in D_mol (diffusion coefficients)
    for d_name, d_val in D_mol.items():
        key = f"D_{d_name}" if d_name in species_list else d_name
        data[key] = get_v(d_val)

    # 3. Calculate delta values for specific sulfur species
    # We prioritize common names (SO4, h2s, FeS, S0, FeS2)
    isotope_map = {
        "SO4": "SO4_32",
        "h2s": "h2s_32",
        "hs": "hs_32",
        "TS2": "TS2_32",
        "FeS": "FeS_32",
        "S0": "S0_32",
        "FeS2": "FeS2_32",
    }

    for base, iso in isotope_map.items():
        if f"c_{base}" in data and f"c_{iso}" in data:
            s_total = data[f"c_{base}"]
            if base == "FeS2":
                s_total = 2.0 * s_total
            s32 = data[f"c_{iso}"]
            data[f"d_{base}"] = get_delta(s_total, s32, mp.VCDT)

    data["w"] = np.ones(len(z)) * mp.w
    data["phi"] = np.ones(len(z)) * mp.phi

    df = pd.DataFrame(data)
    # Ensure no NaN from misalignment by strictly using the same structure
    fqfn = pl.Path.cwd() / f"{mp.plot_name}.csv"
    df.to_csv(fqfn, index=False)

    return df, fqfn


def save_state(c, filename):
    """
    Save the current concentration profiles (state) to a compressed numpy file.

    :param c: data_container containing species as CellVariables
    :param filename: Path to save the NPZ file
    """
    from pathlib import Path

    path = Path(filename)
    if path.exists():
        path.rename(f"{filename}-old")

    state_dict = {}
    for key, var in c.items():
        if hasattr(var, "value"):
            state_dict[key] = var.value

    np.savez_compressed(filename, **state_dict)
    print(f"State saved to {filename}")


def read_state(c, filename):
    """
    Read concentration profiles (state) from a compressed numpy file and update c.

    :param c: data_container containing species as CellVariables
    :param filename: Path to the NPZ file
    """
    import pathlib as pl

    if not pl.Path(filename).exists():
        print(f"Warning: State file {filename} not found. Skipping initialization.")
        return False

    with np.load(filename) as data:
        for key in data.files:
            if hasattr(c, key):
                var = getattr(c, key)
                if hasattr(var, "setValue"):
                    var.setValue(data[key])
                else:
                    c[key] = data[key]
            else:
                print(
                    f"Warning: Species {key} found in state file but not in model container."
                )

    print(f"State loaded from {filename}")
    return True


def safe_ratio(
    num: np.ndarray,
    den: np.ndarray,
    fill: Union[float, int],
) -> np.ndarray:
    """
    Return ``num / den`` element‑wise while protecting against division‑by‑zero.

    Parameters
    ----------
    num : np.ndarray
        Numerator array (any shape that broadcasts with ``den``).
    den : np.ndarray
        Denominator array. Zeros are handled gracefully.
    fill : float or int, optional
        Value to place where ``den == 0``.  The default is ``np.nan``.
        Use ``0`` if you prefer a zero‑filled result.

    Returns
    -------
    np.ndarray
        Array of the same shape as the broadcasted inputs containing the
        element‑wise ratios. Positions where ``den`` is zero contain ``fill``.

    Notes
    -----
    * ``np.divide`` is used with the ``where`` argument – this avoids the
      creation of intermediate infinities and suppresses the runtime warning.
    * The output array is allocated with ``np.empty_like(num, dtype=float)`` so
      the result is always a floating‑point array, even if the inputs are integer.
    """
    # Ensure float output – division of ints would truncate otherwise
    out = np.empty_like(num, dtype=float)

    # Perform division only where denominator is non‑zero
    np.divide(num, den, out=out, where=den != 0)

    # Fill the “bad” positions
    out[den == 0] = fill
    return out


def calculate_k_iron_reduction(Fe3, h2s):
    """
    Calculates the rate constant k_FeOx-SII for an array of ratios.
    Fe3+/H2S

    Based on Equation 46 and 47 from Halevy et al. (2023).
    """
    # 1. Define the piecewise conditions for half-life (tau_1/2) in hours
    # Condition 1: Ratio < 1 -> tau = 1.5h [cite: 1462]
    # Condition 2: 1 <= Ratio <= 2 -> Linear transition [cite: 1463]
    # Condition 3: Ratio > 2 -> tau = 0.5h [cite: 1464]

    ratios = safe_ratio(Fe3, h2s, 10.0)

    tau_half = np.piecewise(
        ratios,
        [ratios < 1, (ratios >= 1) & (ratios <= 2), ratios > 2],
        [1.5, lambda r: 0.5 + 1.0 * (2.0 - r), 0.5],
    )

    # 2. Calculate the rate constant k = 0.693 / tau_1/2
    k_values = 0.693 / tau_half

    return k_values / (60 * 60 * 24 * 1e3)


def wt_percent_to_solid_conc(wp, mw, d, phi):
    """
    Convert a weight‑percent (dry mass) of a solid component to its
    concentration in bulk solution (mmol L⁻¹, i.e. mol m⁻³).

    Parameters
    ----------
    wp : float or array‑like
        Weight percentage of the component (0–100 %).
    mw : float
        Molecular weight of the component (g mol⁻¹).
    d  : float
        Grain density of the pure solid (g cm⁻³).
    phi : float
        Porosity of the bulk material (fraction, 0–1).

    Returns
    -------
    C_bulk : float or np.ndarray
        Concentration in mol m⁻³ (identical numerically to mmol L⁻¹).

    Notes
    -----
    The conversion assumes a representative bulk volume of 1 m³.
    1 m³ = 1 000 000 cm³, hence the factor 1e6 in the formula.
    """
    # Ensure inputs are NumPy arrays for broadcasting (works also for scalars)
    wp = np.asarray(wp, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # 1. Mass fraction (dimensionless)
    w_frac = wp / 100.0

    phi_val = getattr(phi, "value", phi)

    # 2. Bulk density of the pure component (g cm⁻³)
    rho_bulk = d * (1.0 - phi_val)

    # 3. Mass of the component per 1 m³ bulk volume (g)
    #    1 m³ = 1 000 000 cm³
    mass_per_m3 = rho_bulk * w_frac * 1.0e6  # g m⁻³

    # 4. Convert mass to moles (mol m⁻³)
    C_bulk = mass_per_m3 / mw  # mol m⁻³

    return C_bulk


def solid_conc_to_umol_per_g(C_solid, d):
    """
    Convert a concentration that is expressed per unit solid volume
    (mmol L⁻¹ solid) to mol/dry weight.

    Parameters
    ----------
    C_solid : float or array‑like
        Concentration per unit solid volume (mmol L⁻¹ of solid).
       umol/cm^3 or mol/m^3
    d : float
        Grain (particle) density of the pure solid (g cm⁻³).


    Returns
    -------
    umol_gram : float or np.ndarray
        umol/g *dry* sediment
    """

    return C_solid / d


def solid_conc_to_wt_percent(C_solid, mw, d, phi):
    """
    Convert a concentration that is expressed per unit solid volume
    (mmol L⁻¹ solid) to a weight‑percent of the dry sediment matrix.

    Parameters
    ----------
    C_solid : float or array‑like
        Concentration per unit solid volume (mmol L⁻¹ of solid).
        Numerically this is equivalent to mol m⁻³ of solid because
        1 mmol L⁻¹ = 1 mol m⁻³.
    mw : float
        Molecular weight of the component (g mol⁻¹).
    d : float
        Grain (particle) density of the pure solid (g cm⁻³).
    phi : float
        Porosity of the bulk sediment (fraction, 0 ≤ phi ≤ 1).

    Returns
    -------
    wt_percent : float or np.ndarray
        Weight percentage of the component in the *dry* sediment
        (0 – 100 %).  If `phi == 1` (no solid) the function returns
        `np.nan` because a weight‑percent is undefined.

    Notes
    -----
    The conversion follows the steps:

    1. Convert solid‑phase concentration to bulk‑phase concentration:
       C_bulk = C_solid * (1‑phi)   [mol m⁻³ bulk].

    2. Mass of solute per bulk volume:
       m_sol = C_bulk * mw          [g m⁻³].

    3. Dry bulk density of the sediment:
       rho_dry = d * (1‑phi) * 1e6  [g m⁻³].

    4. Weight fraction = m_sol / rho_dry, then *100 for wt %.

    Because the factor (1‑phi) appears in both numerator and denominator,
    it cancels analytically, leaving a result that is independent of porosity.
    The implementation keeps the full expression for clarity and to
    guard against accidental misuse.
    """
    # -----------------------------------------------------------------
    # 1. Input handling / validation
    # -----------------------------------------------------------------
    C_solid = np.asarray(C_solid, dtype=float)

    # -----------------------------------------------------------------
    # 2. Core calculation
    # -----------------------------------------------------------------
    # (a) Bulk concentration (mol m⁻³)
    C_bulk = C_solid * (1.0 - phi)

    # (b) Mass of solute per bulk volume (g m⁻³)
    m_sol = C_bulk * mw  # because 1 mmol L⁻¹ = 1 mol m⁻³

    # (c) Dry bulk density (g m⁻³)
    rho_dry = d * (1.0 - phi) * 1e6  # 1 g cm⁻³ = 1 × 10⁶ g m⁻³

    # (d) Weight percent
    wt_percent = (m_sol / rho_dry) * 100.0

    return wt_percent


def liquid_conc_to_wt_percent(C_pw, mw, d, phi):
    """
    Convert a dissolved‑species concentration expressed as
    mmol L⁻¹ of pore‑water to a weight‑percentage of the dry sediment
    (wt % dry‑mass).

    Parameters
    ----------
    C_pw : float or array‑like
        Pore‑water concentration (mmol L⁻¹).  Numerically this is the
        same as mol m⁻³ because 1 mmol L⁻¹ = 1 mol m⁻³.
    mw : float
        Molecular weight of the dissolved component (g mol⁻¹).
    d : float
        Grain (particle) density of the pure solid (g cm⁻³).  Typical
        values for quartz‑rich sediments are ≈2.65 g cm⁻³.
    phi : float
        Porosity of the bulk material (fraction, 0 ≤ phi ≤ 1).

    Returns
    -------
    wt_percent : float or np.ndarray
        Weight percentage of the component in the *dry* sediment
        (0 – 100 %).  If phi == 0 the function returns 0 because no
        pore‑water (and thus no dissolved iron) can exist.

    Notes
    -----
    The conversion proceeds as

    1. **Moles of solute per bulk volume**
       `n_bulk = C_pw * phi`   (mol m⁻³)

       (`C_pw` is per pore‑water volume, so we multiply by the
       fraction of the bulk that is water.)

    2. **Mass of solute per bulk volume**
       `m_solute = n_bulk * mw`   (g m⁻³)

    3. **Mass of dry solids per bulk volume**
       The dry‑bulk density ρ_bulk = d · (1 – phi)   (g cm⁻³)
       Convert to g m⁻³: `rho_bulk = d * (1 - phi) * 1e6`

    4. **Weight percent**
       `wt% = (m_solute / rho_bulk) * 100`

    The function is fully vectorised (accepts NumPy arrays) and
    performs basic input validation.
    """
    # -----------------------------------------------------------------
    # 1. Input validation
    # -----------------------------------------------------------------
    C_pw = np.asarray(C_pw, dtype=float)

    # -----------------------------------------------------------------
    # 2. Core calculation
    # -----------------------------------------------------------------
    # (a) moles of solute per bulk volume (mol m⁻³)
    n_bulk = C_pw * phi  # because 1 mmol L⁻¹ = 1 mol m⁻³

    # (b) grams of solute per bulk volume (g m⁻³)
    m_solute = n_bulk * mw  # g m⁻³

    # (c) dry‑bulk density (g m⁻³)
    rho_bulk = d * (1.0 - phi) * 1e6  # 1 g cm⁻³ = 1 × 10⁶ g m⁻³

    # (d) weight percent
    wt_percent = (m_solute / rho_bulk) * 100.0

    return wt_percent


def solid_conc_to_wt_percent_old(C_bulk, mw, d, phi):
    """
    Convert a bulk concentration (mmol L⁻¹ ≡ mol m⁻³) back to weight
    percentage of the component in the dry solid matrix.

    Parameters
    ----------
    C_bulk : float or array‑like
        Bulk concentration (mol m⁻³; same numerically as mmol L⁻¹).
    mw : float
        Molecular weight of the component (g mol⁻¹).
    d  : float
        Grain density of the pure solid (g cm⁻³).
    phi : float
        Porosity of the bulk material (fraction, 0–1).

    Returns
    -------
    wp : float or np.ndarray
        Weight percentage (0–100 %) of the component in the dry mass.
    """
    C_bulk = np.asarray(C_bulk, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # 1. Bulk density of the pure solid (g cm⁻³)
    phi_val = getattr(phi, "value", phi)
    rho_bulk = d * (1.0 - phi_val)

    # 2. Mass of the component per 1 m³ bulk volume (g m⁻³)
    #    mass = concentration × molecular weight
    mass_per_m3 = C_bulk * mw  # g m⁻³

    # 3. Convert mass per m³ to a mass fraction of the dry solid.
    #    mass_per_m3 = rho_bulk * w_frac * 1e6   →  w_frac = mass / (rho_bulk·1e6)
    w_frac = mass_per_m3 / (rho_bulk * 1.0e6)  # dimensionless

    # 4. Finally express as weight percent (0–100 %)
    wp = w_frac * 100.0
    return wp


_executor = None


def _get_executor():
    """Create a module‑wide ThreadPoolExecutor on first use."""
    global _executor
    if _executor is None:
        from concurrent.futures import ThreadPoolExecutor
        _executor = ThreadPoolExecutor(max_workers=1)
    return _executor


def check_peclet_numbers(mesh, mp, D_mol, species_list, bc_map):
    # Get cell sizes (dx)
    # For a 1D mesh, mesh.dx is an array of cell lengths
    dx = mesh.dx

    for species in species_list:
        props = bc_map[species]

        # 1. Determine Velocity
        vel = mp.w
        if props["type"] == "dissolved":
            vel = mp.w - mp.advection

            # 2. Determine Diffusion
            # For solids, D_mol is usually 0, so D is just D_bio
            d_val = getattr(D_mol, species) + D_mol.D_bio

            # 3. Calculate Pe for every cell
            # We use a small epsilon to avoid division by zero
            pe_cells = (np.abs(vel) * dx) / (d_val + 1e-20)

            max_pe = np.max(pe_cells)

            if max_pe > 2:
                print(f"Species: {species:10} | Max Pe: {max_pe:.2f}")
                print(
                    f"  --> WARNING: Pe > 2. Numerical dispersion/oscillations likely."
                )


def save_data_async(
    mp,
    c,
    k,
    species_list,
    z,
    D_mol,
    diagenetic_reactions,
    equilibrium_reactions,
    current_dt,
    title=None,
) -> None:
    """
    Schedule a background write of model results to CSV and return immediately.
    """
    # 1. Capture current rates in the main thread (FiPy objects aren't thread-safe)
    c_numpy = data_container({s: ArrayProxy(val.value) for s, val in c.items()})
    mp_numpy = data_container(mp)
    mp_numpy.phi = ArrayProxy(mp.phi.value)
    f_final, RATES = diagenetic_reactions(mp_numpy, c_numpy, k, data_container())
    f_final, RATES = equilibrium_reactions(mp, c, k, f_final, RATES, current_dt)

    # 2. Snapshot values (numpy arrays) to avoid race conditions during solver update
    def snap(obj):
        if hasattr(obj, "value"):
            return obj.value.copy()
        if hasattr(obj, "copy"):
            return obj.copy()
        return obj

    c_snap = data_container({s: snap(getattr(c, s)) for s in species_list})
    f_snap = data_container(
        {key: (None, None, snap(getattr(f_final, key)[2])) for key in f_final.keys()}
    )
    D_mol_snap = data_container({key: snap(val) for key, val in D_mol.items()})

    # Copy metadata
    mp_snap = data_container(mp)
    k_snap = data_container(k)
    z_snap = z.copy()

    # 3. Submit to background worker
    _get_executor().submit(
        _save_data_to_disk,
        mp_snap,
        c_snap,
        k_snap,
        species_list,
        z_snap,
        D_mol_snap,
        f_snap,
    )

    return None, None


_ureg = None

def get_time_units(t):
    """Adjust time units between second to ky."""
    global _ureg
    if _ureg is None:
        import pint
        _ureg = pint.UnitRegistry()
    Q_ = _ureg.Quantity

    seconds = 60
    minutes = seconds * 60
    hours = minutes * 60
    days = hours * 24
    weeks = days * 7
    months = weeks * 4.5
    years = months * 12
    kyears = years * 1000
    Myears = kyears * 1000

    if t < seconds:
        t = Q_(f"{t} seconds")
    elif t < minutes:
        t = Q_(f"{t} seconds").to("minutes")
    elif t < hours:
        t = Q_(f"{t} seconds").to("hours")
    elif t < days:
        t = Q_(f"{t} seconds").to("days")
    elif t < weeks:
        t = Q_(f"{t} seconds").to("week")
    elif t < months:
        t = Q_(f"{t} seconds").to("month")
    elif t < years:
        t = Q_(f"{t} seconds").to("year")
    elif t < kyears:
        t = Q_(f"{t} seconds").to("kyear")
    else:
        t = Q_(f"{t} seconds").to("kyear")

    return t


def make_grid2(
    L: float,
    initial_spacing: float,
    reaction_zone_spacing: float,
    max_spacing: float,
    reaction_zone: tuple,
    r=1.05,
):
    """Build a variable distance mesh.

    The mesh will decrease from initial_spacing to reaction_zone_spacing, and then increases to max_spacing below the reaction zone. The reaction zone depth interval is given by the reaction_zone tuple e.g. (0.5, 1)

    Args:
    -----
        L (float): Total length of the domain (max depth).
        initial_spacing (float): The size of the first cell (at z=0).
        max_spacing (float): Maximum cell size allowed.
        r (float): Geometric growth rate (default 1.05).

    Returns:
    -------
        tuple (mesh, z_centers)
            mesh: A fipy.Grid1D object.
            z_centers: A numpy array of cell center coordinates.
    """
    rz_start, rz_end = reaction_zone
    dx_list = []
    current_z = 0

    # 1. Refinement phase: from surface to reaction zone
    current_dx = initial_spacing
    while current_z + current_dx < rz_start:
        dx_list.append(current_dx)
        current_z += current_dx
        current_dx = max(current_dx / r, reaction_zone_spacing)

    if current_z < rz_start:
        dx_list.append(rz_start - current_z)
        current_z = rz_start

    # 2. Reaction zone phase: uniform fine spacing
    n_rz = int(np.ceil((rz_end - rz_start) / reaction_zone_spacing))
    actual_rz_dx = (rz_end - rz_start) / n_rz
    dx_list.extend([actual_rz_dx] * n_rz)
    current_z = rz_end

    # 3. Coarsening phase: from reaction zone to L
    current_dx = actual_rz_dx * r
    while current_z + current_dx < L and current_dx < max_spacing:
        dx_list.append(current_dx)
        current_z += current_dx
        current_dx *= r

    # 4. Uniform Section
    if current_z < L:
        remaining_L = L - current_z
        n_uniform = int(np.ceil(remaining_L / max_spacing))
        if n_uniform > 0:
            actual_uniform_dx = remaining_L / n_uniform
            dx_list.extend([actual_uniform_dx] * n_uniform)

    dx_array = np.array(dx_list)
    N = len(dx_array)
    print(f"Grid generated with {N} points.")

    mesh = Mesh1D(dx=dx_array)
    z_centers = mesh.cellCenters[0]

    return mesh, z_centers


def get_total_delta(c, mp, index=-1):
    """Get total delta that is being buried.

    Calculate the total amount of S and S32 that leaves the system through the lower
    boundary. Note that we have to count the mass of FeS2 since it has 2 S, however,
    FeS2_32 is already corrected, so we do not mutiply it.
    """
    from .diff_lib import get_delta

    phi_val = getattr(mp.phi, "value", mp.phi)
    phi = phi_val[index] if hasattr(phi_val, "__getitem__") else phi_val
    f_s = 1.0 - phi

    # Liquid species are scaled by porosity (phi)
    # Solid species are scaled by solid fraction (1-phi)
    s = phi * (c.SO4.value[index] + c.TS2.value[index]) + f_s * (
        c.S0.value[index] + c.FeS.value[index] + 2 * c.FeS2.value[index]
    )
    s32 = phi * (c.SO4_32.value[index] + c.TS2_32.value[index]) + f_s * (
        c.S0_32.value[index] + c.FeS_32.value[index] + c.FeS2_32.value[index]
    )

    return get_delta(s, s32, mp.VCDT)


# =============================================================================
# HELPER FUNCTIONS (Matrix Math Abstraction)
# =============================================================================
def add_implicit_sink(
    LHS,
    RATES,
    species,
    coeff,
    rate,
    mp,
    has_solid: bool,
    c=None,
    reaction=None,
):
    """Add an implicit consumption term to the LHS matrix."""
    phi = mp.phi
    fac = (1.0 - phi) if has_solid else phi

    bulk_rate = rate * fac
    LHS[species] = LHS[species] - coeff * fac
    RATES[species] -= bulk_rate
    if reaction is not None:
        key = f"r_{reaction}_{species}"
        if key not in RATES:
            RATES[key] = np.zeros_like(bulk_rate)
        RATES[key] -= bulk_rate


def add_explicit_source(
    RHS,
    RATES,
    species,
    rate,
    mp,
    has_solid: bool,
    update_rates=True,
    c=None,
    reaction=None,
):
    """Add a production term to the RHS vector."""
    phi = mp.phi
    fac = (1.0 - phi) if has_solid else phi

    scaled_rate = rate * fac
    RHS[species] = RHS[species] + scaled_rate
    if update_rates:
        RATES[species] += getattr(scaled_rate, "value", scaled_rate)
    if reaction is not None:
        key = f"r_{reaction}_{species}"
        if key not in RATES:
            RATES[key] = np.zeros_like(scaled_rate)
        RATES[key] += getattr(scaled_rate, "value", scaled_rate)


def add_monod_sink(
    LHS,
    RHS,
    RATES,
    species: str,
    conc: Any,
    K_m: Any,
    R_max: Any,
    mp: Any,
    has_solid: bool,
    scheme: str = "hybrid",
    c: Any = None,
    reaction: str | None = None,
    multiplier: float = 1.0,
):
    """Add a Monod consumption term: R = multiplier * R_max * C / (C + K_m).

    Supports three linearization schemes:
    - 'picard': Secant Picard linearization (unconditional positivity, matches legacy).
    - 'newton': Tangent Newton-Raphson linearization (quadratic convergence).
    - 'hybrid': Safeguarded Newton: Newton for C > K_m (zero-order regime),
                smoothly reverting to Picard for C <= K_m (near-zero first-order regime).
    """
    phi = mp.phi
    fac = (1.0 - phi) if has_solid else phi

    # Extract concentration values if CellVariable or ArrayProxy
    C_val = getattr(conc, "value", conc)
    C_val = np.maximum(C_val, 1e-30)

    # Scale R_max by stoichiometric multiplier if provided
    R_max_eff = R_max * multiplier if multiplier != 1.0 else R_max

    denom = C_val + K_m
    monod_frac = C_val / denom
    physical_rate = R_max_eff * monod_frac
    bulk_rate = physical_rate * fac

    scheme_lower = scheme.lower() if isinstance(scheme, str) else "hybrid"

    if scheme_lower == "picard":
        coeff = R_max_eff / denom
        LHS[species] = LHS[species] - coeff * fac
        RATES[species] -= bulk_rate
    elif scheme_lower == "newton":
        J = R_max_eff * (K_m / (denom * denom))
        R_rem = -R_max_eff * (monod_frac * monod_frac)
        scaled_rem = R_rem * fac

        LHS[species] = LHS[species] - J * fac
        RHS[species] = RHS[species] + scaled_rem
        RATES[species] -= bulk_rate
    elif scheme_lower == "hybrid":
        # Newton where C > K_m, Picard where C <= K_m
        is_newton = np.where(C_val > K_m, 1.0, 0.0)
        is_picard = 1.0 - is_newton

        J = R_max_eff * (K_m / (denom * denom))
        R_rem = -R_max_eff * (monod_frac * monod_frac)
        coeff_picard = R_max_eff / denom

        coeff_eff = is_newton * J + is_picard * coeff_picard
        rem_eff = is_newton * R_rem * fac

        LHS[species] = LHS[species] - coeff_eff * fac
        RHS[species] = RHS[species] + rem_eff
        RATES[species] -= bulk_rate
    else:
        raise ValueError(f"Unknown Monod scheme '{scheme}'. Choose from 'picard', 'newton', or 'hybrid'.")

    if reaction is not None:
        key = f"r_{reaction}_{species}"
        if key not in RATES:
            RATES[key] = np.zeros_like(bulk_rate)
        RATES[key] -= bulk_rate


def smooth_ramp(x, eps=0.02):
    """C1 continuous smooth ramp function:
    Approximates max(x, 0) with continuous value and continuous first derivative:
        S_eps(x) = 0                       for x <= -eps
                 = (x + eps)^2 / (4 * eps) for -eps < x < eps
                 = x                       for x >= eps
    Eliminates discontinuous operator jumps across phase change boundaries (e.g. Omega = 1).
    """
    quad = (x + eps) * (x + eps) / (4.0 * eps)
    return np.where(x <= -eps, 0.0, np.where(x >= eps, x, quad))


def add_implicit_coupling_new(
    CROSS,
    RATES,
    LHS,
    target_species,
    source_species,
    coeff,
    rate,
    mp,
    has_solid: bool,
    c=None,
    add_lhs_sink=True,
    stoich_ratio=1.0,
    reaction=None,
):
    """Add a coupled implicit source term with optional stoichiometry."""
    phi = mp.phi
    fac = (1.0 - phi) if has_solid else phi

    cross_coeff = coeff * stoich_ratio * fac
    CROSS[target_species].append((source_species, cross_coeff))

    bulk_rate = rate * fac
    if add_lhs_sink:
        LHS[source_species] = LHS[source_species] - coeff * fac
        RATES[source_species] -= bulk_rate
        if reaction is not None:
            key_src = f"r_{reaction}_{source_species}"
            if key_src not in RATES:
                RATES[key_src] = np.zeros_like(bulk_rate)
            RATES[key_src] -= bulk_rate

    scaled_target_rate = bulk_rate * stoich_ratio
    RATES[target_species] += scaled_target_rate
    if reaction is not None:
        key_tgt = f"r_{reaction}_{target_species}"
        if key_tgt not in RATES:
            RATES[key_tgt] = np.zeros_like(scaled_target_rate)
        RATES[key_tgt] += scaled_target_rate


def _get_base_species(name: str) -> str:
    if not name:
        return ""
    return name.split("_")[0].upper()


def add_coupled_reaction(
    CROSS,
    LHS,
    RATES,
    mp,
    master_species,
    reactants,
    products,
    coeff_master,
    rate_master,
    has_solid: bool,
    reaction_name: str,
    ref_species: str = None,
    stoich_ref: float = None,
):
    """Couple multiple reactants and products to a single master reactant."""
    phi = mp.phi
    fac = (1.0 - phi) if has_solid else phi

    # 1. Parse master_species and master_stoich
    if isinstance(master_species, dict):
        # Expecting a single-key dictionary like {"Fe3": 4}
        master_species_name = list(master_species.keys())[0]
        master_stoich = float(list(master_species.values())[0])
    else:
        master_species_name = str(master_species)
        master_stoich = 1.0

    # 2. Automatically compute scaling factor based on ref_species stoichiometry in the unnormalized reaction
    scaling_factor = 1.0
    if ref_species:
        if stoich_ref is not None:
            scaling_factor = master_stoich / float(stoich_ref)
        else:
            ref_base = _get_base_species(ref_species)
            master_base = _get_base_species(master_species_name)
            
            if ref_base == master_base:
                stoich_ref_local = master_stoich
            else:
                # Check reactants
                for spec, stoich in reactants.items():
                    if _get_base_species(spec) == ref_base:
                        stoich_ref_local = float(stoich)
                        break
                else:
                    # Check products
                    for spec, stoich in products.items():
                        if _get_base_species(spec) == ref_base:
                            stoich_ref_local = float(stoich)
                            break
                    else:
                        stoich_ref_local = 1.0
            
            scaling_factor = master_stoich / stoich_ref_local

    coeff_master_scaled = coeff_master * scaling_factor
    rate_master_scaled = rate_master * scaling_factor

    # 3. Normalize reactants and products to 1 unit of master species for matrix coupling
    reactants_norm = {k: v / master_stoich for k, v in reactants.items()}
    products_norm = {k: v / master_stoich for k, v in products.items()}

    # 4. Apply self-implicit sink to the master reactant
    LHS[master_species_name] = LHS[master_species_name] - coeff_master_scaled * fac
    bulk_rate_master = rate_master_scaled * fac
    if master_species_name in RATES:
        RATES[master_species_name] -= bulk_rate_master

    key_master = f"r_{reaction_name}_{master_species_name}"
    if key_master not in RATES:
        RATES[key_master] = np.zeros_like(bulk_rate_master)
    RATES[key_master] -= bulk_rate_master

    # Helper for cross-coupling other species
    def couple_species(species, stoich_norm, sign):
        if species not in CROSS:
            return

        cross_coeff = coeff_master_scaled * stoich_norm * fac
        CROSS[species].append((master_species_name, sign * cross_coeff))

        bulk_rate = bulk_rate_master * stoich_norm
        if species in RATES:
            RATES[species] += sign * bulk_rate

        key = f"r_{reaction_name}_{species}"
        if key not in RATES:
            RATES[key] = np.zeros_like(bulk_rate)
        RATES[key] += sign * bulk_rate

    # 5. Couple other reactants (consumed -> sign = -1.0)
    for spec, stoich_norm in reactants_norm.items():
        if spec != master_species_name:
            couple_species(spec, stoich_norm, sign=-1.0)

    # 6. Couple products (produced -> sign = 1.0)
    for spec, stoich_norm in products_norm.items():
        couple_species(spec, stoich_norm, sign=1.0)


def calculate_fractionated_coeff_32(coeff_total, c_total, c_32, alpha, eps=1e-20):
    """
    Calculate the fractionated isotope rate coefficient for 32S.
    """
    c_tot_np = np.asarray(c_total)
    c_32_np = np.asarray(c_32)
    valid = c_tot_np > 1e-12
    alpha_eff = np.where(valid, alpha, 1.0)
    safe_tot = np.where(valid, c_tot_np, 1.0)
    safe_32 = np.where(valid, c_32_np, 1.0)
    ratio_32 = np.where(valid, safe_32 / safe_tot, 1.0)
    ratio_32 = np.clip(ratio_32, 0.5, 1.5)
    denom_ratio = 1.0 + (alpha_eff - 1.0) * ratio_32
    return coeff_total * alpha_eff / denom_ratio


def partition_equilibrium_isotope_32(c_32, frac_target, frac_other, alpha_eq_32, eps=1e-30):
    """
    Calculate the concentration of the 32S isotope in a partitioned species 
    under equilibrium fractionation.
    """
    denom = frac_target + alpha_eq_32 * frac_other + eps
    return c_32 * frac_target / denom
