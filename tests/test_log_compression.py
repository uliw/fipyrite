import gzip
import os
import tempfile
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fipy import CellVariable, Grid1D
from fipyrite.diff_lib import data_container
from fipyrite.solver_calls import _compress_log_file, run_non_steady_state_solver_coupled


def test_compress_log_file():
    """Test that _compress_log_file compresses the text log and removes the uncompressed original."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test_run.log")
        gz_path = f"{log_path}.gz"
        
        # Write dummy log data
        with open(log_path, "w") as f:
            f.write("Line 1: Simulation started\n")
            f.write("Line 2: Step 1 dt=1.0\n")
            f.write("Line 3: Final Report: Completed\n")
        
        # Compress
        res = _compress_log_file(log_path)
        assert res == gz_path
        assert os.path.exists(gz_path)
        assert not os.path.exists(log_path)
        
        # Verify content
        with gzip.open(gz_path, "rt") as f:
            content = f.read()
        assert "Simulation started" in content
        assert "Final Report: Completed" in content


class MockMP:
    def __init__(self, plot_name):
        self.plot_name = plot_name
        self.max_steps = 1
        self.start_time = 0.0
        self.t_end = 100.0
        self.dt_min = 1.0
        self.dt_init = 5.0
        self.dt_max = 50.0
        self.backup_step = 100
        self.report_step = 100
        self.isotopes = False
        self.title = "test"
        self.enable_rate_adaptation = False
        self.phi = 0.8
        self.solver_backend = "default"
        self.tolerance = 1e-12
        self.dt_tolerance = -1.0
        self.adaptive_solver_tolerance = False


class MockD:
    def __init__(self):
        self.FeS = 0.0
        self.TS2 = 0.0
        self.D_bio = 0.0
        self.D_irr = 0.0


@patch("fipyrite.solver_calls._setup_static_coupled_equation")
@patch("fipyrite.solver_calls._update_static_coefficients")
@patch("fipyrite.solver_calls.save_data")
@patch("fipyrite.solver_calls.save_state")
def test_solver_compresses_log_on_completion(mock_save_state, mock_save_data, mock_update_coeffs, mock_setup_eq):
    """Test that solver run automatically closes and compresses its log file to .log.gz."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_name = os.path.join(tmpdir, "my_sim")
        log_path = f"{plot_name}.log"
        gz_path = f"{plot_name}.log.gz"
        
        mp = MockMP(plot_name)
        mesh = Grid1D(nx=3)
        c = data_container({
            "FeS": CellVariable(mesh=mesh, value=1.0, hasOld=True),
        })
        k = data_container()
        D_mol = MockD()
        bc_map = {"FeS": {"type": "solid", "top": 0.0}}
        z = np.array([0.0, 1.0, 2.0, 3.0])
        
        mock_coupled_eq = MagicMock()
        mock_setup_eq.return_value = (mock_coupled_eq, {}, {}, {})
        mock_update_coeffs.return_value = {}
        
        run_non_steady_state_solver_coupled(
            mp, c, ["FeS"], ["FeS"], k, MagicMock(), MagicMock(), mesh, D_mol, bc_map, z
        )
        
        assert not os.path.exists(log_path)
        assert os.path.exists(gz_path)
        
        with gzip.open(gz_path, "rt") as f:
            lines = f.readlines()
            assert any("Final Report:" in line for line in lines)
            assert any("total sweeps" in line for line in lines)


@patch("fipyrite.solver_calls.save_data_async")
def test_report_step_status_enhanced_logging(mock_save_async):
    """Test that _report_step_status includes wall time, pace, sim speed, and sweeps."""
    from fipyrite.solver_calls import _report_step_status

    logged = []
    def mock_log(msg):
        logged.append(msg)

    class DummyMP:
        phi = 0.8
        report_step = 10
        isotopes = False
        title = None
        k = data_container()

    mesh = Grid1D(nx=3)
    c = data_container({
        "Fe2_total": CellVariable(mesh=mesh, value=1.0),
        "Fe3": CellVariable(mesh=mesh, value=1.0),
        "FeS": CellVariable(mesh=mesh, value=1.0),
        "FeS2": CellVariable(mesh=mesh, value=1.0),
    })

    z = np.array([0.0, 1.0, 2.0, 3.0])

    _report_step_status(
        step=10,
        total_time=86400.0 * 10,
        current_dt=3600.0,
        rms_change=1e-3,
        mp=DummyMP(),
        c=c,
        z=z,
        species_list_full=[],
        D_mol=None,
        diagenetic_reactions=None,
        equilibrium_reactions=None,
        plot_queue=None,
        _log=mock_log,
        sweeps=4,
        total_sweeps=42,
        start_wall=time.time() - 120.0,
        last_report_wall=time.time() - 5.0,
        last_report_time=86400.0 * 5,
    )

    assert len(logged) == 1
    line = logged[0]
    # Check for wall timestamp, pace, sim speed, sweeps, and total sweeps
    assert "Step   10" in line
    assert "s/step" in line
    assert "d/min" in line or "yr/min" in line
    assert "sweeps: 4" in line
    assert "tot: 42" in line

