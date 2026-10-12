# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-10-12

### Added

- **File Path Resolution & Diagnostic Error Handling**:
  - Enhanced file path resolution in `DiageneticModel` (`_resolve`), searching the current working directory, the model script directory (`base_dir`), and fallback naming patterns (such as `<experiment>_reactions.py`).
  - Implemented early file validation (`validate_simulation_files()`) during `DiageneticModel` initialization and setup to catch missing simulation files before simulation execution.
  - Added informative error diagnostics (`[MANTIS ERROR]`) detailing exact search locations and resolution guidance when species, reactions, constants, limiters, or plot layout files are missing.
  - Improved plot layout file resolution in `plot_data_new.load_layout_from_file` and `plot_summary` with clear error reporting and search path feedback.
  - Added unit test suite in `tests/test_diagenetic_model.py` testing missing file scenarios, exit behavior, and directory resolution from both CWD and model base directories.

## [0.1.1] - 2026-10-12

### Added

- **Steady-State Detection & Convergence**:
  - Implemented a grid-independent spatial $L_2$ steady-state convergence check in `run_non_steady_state_solver_monolithic`. Computes a volume-weighted root-mean-square (RMS) change across grid cells per species and checks against `steady_state_tolerance` (or falls back to `dt_tolerance` if set).
  - Added a minimum step requirement (step > 5) to prevent premature steady-state termination.
- **Model Parameters & Unit Conversions**:
  - Added support for grid spacing parameters with unit conversion in `DiageneticModel`: `initial_spacing`, `reaction_zone_spacing`, and `max_spacing`.
  - Added support for the `reaction_zone` tuple/list interval parameter with unit conversion.
  - Added support for `steady_state_tolerance` and `dt_tolerance` solver tolerance parameters.
  - Added `check_reactant_coexistence` boolean option and enabled the post-simulation reactant coexistence diagnostic by default.
- **Examples**:
  - Added fast and slow simulation setups for the Velde et al. (2016) sediment diagenesis benchmark:
    - `examples/velde_et_al_2016/run_velde_fast.py` & `kinetic_constants_fast.py`: fast configuration with pyrite precipitation.
    - `examples/velde_et_al_2016/run_velde_slow.py` & `kinetic_constants_slow.py`: slow configuration tracking sulfur isotopes.
- **CI/CD & Automation**:
  - Added GitHub Actions workflow `.github/workflows/tests.yml` to run automated test suites across Python 3.12, 3.13, and 3.14.
  - Added GitHub Actions workflow `.github/workflows/publish-pypi.yml` for automated PyPI package publishing upon GitHub releases.
- **Documentation**:
  - Added comprehensive documentation on physical units (`pint` integration) and non-uniform grid generation to `docs/tutorial.md` and `docs/tutorial.org`.

### Changed

- **Rate Laws & Equation System Generation**:
  - Simplified particulate organic carbon (POC) respiration rate equations by removing redundant coefficient calculations.
  - Simplified FeS precipitation and dissolution rate formulation by removing unnecessary Michaelis-Menten terms.
  - Adjusted default smoothing parameter `fes_smooth_epsilon` from `0.05` to `0.01` for sharper, smoother reaction transitions.
  - Dynamically extract and handle active POC species and kinetic parameter pairs instead of using hardcoded POC fast/slow loops.
- **Differential Operators Library (`diff_lib`)**:
  - Added type annotations across numerical and differential operator functions.
  - Improved formatting, line wrapping, indentation, and visual structure.
- **Project Packaging & Metadata**:
  - Updated package distribution name to `mantis-rtm` and enabled dynamic versioning using `setuptools_scm`.
  - Added `pint` as an explicit core dependency in `pyproject.toml` and `requirements.txt`.
  - Added AGPL-3.0-or-later license configuration and classifiers.
  - Configured project URLs (Homepage, Repository, Issues) and updated build ignore patterns in `.gitignore`.
- **Examples**:
  - Corrected FeS dissolution rate constant in `examples/velde_et_al_2016/kinetic_constants_slow.py`.
  - Adjusted broken axis limits in `examples/velde_et_al_2016/plot_layout.py` for clearer concentration profile visualization.

### Removed

- Removed obsolete `examples/velde_et_al_2016/run_velde_model.py` script.
- Removed legacy table of contents from tutorial documentation.

## [0.1.0] - 2026-10-11

- Initial tagged release with core MANTIS reactive transport solver and tutorial documentation.
