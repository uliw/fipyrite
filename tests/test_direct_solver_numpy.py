"""Test the decoupled direct assembled solver and VariableArray without FiPy."""

import sys
import numpy as np
import pytest

from fipyrite.diff_lib import Mesh1D, VariableArray, data_container
from fipyrite.direct_assembled_solver import (
    DirectAssembledSystem,
    build_native_1d_transport_stencil,
)


def test_no_fipy_import():
    """Verify that importing direct assembled solver components does not load fipy."""
    # Note: fipy may be imported by other test files if run in the same process,
    # so we verify that direct_assembled_solver and diff_lib modules themselves
    # do not have fipy in their globals.
    import fipyrite.diff_lib as dlib
    import fipyrite.direct_assembled_solver as das

    assert "fipy" not in dlib.__dict__
    assert "fipy" not in das.__dict__


def test_mesh1d_and_variable_array():
    """Test Mesh1D and VariableArray interface compatibility."""
    dx = np.array([0.01, 0.02, 0.03, 0.04])
    mesh = Mesh1D(dx)
    assert mesh.numberOfCells == 4
    assert len(mesh.cellVolumes) == 4
    assert len(mesh.faceCoordinates) == 5

    # VariableArray initialization
    var = VariableArray(value=10.0, name="test_var", mesh=mesh)
    assert len(var) == 4
    assert np.allclose(var.value, 10.0)

    # Face value interpolation
    fval = var.faceValue
    assert len(fval) == 5
    assert np.allclose(fval.value, 10.0)

    # Indexing with FaceSelector
    assert fval[mesh.facesLeft] == 10.0
    assert fval[mesh.facesRight] == 10.0

    # Constrain and setValue
    var.constrain(5.0, mesh.facesLeft)
    var.faceGrad.constrain([0.0], mesh.facesRight)
    var.setValue(12.0)
    assert np.allclose(var.value, 12.0)


def test_build_native_1d_transport_stencil():
    """Test native 1D tridiagonal transport stencil builder."""
    z = np.linspace(0.01, 0.1, 10)
    dx = np.full(10, 0.01)
    phi = 0.8
    D = np.full(10, 1e-9)
    w = 1e-10

    # Test Dirichlet dissolved species
    bc_dirichlet = {"type": "dissolved", "top": 28.0}
    ab, b = build_native_1d_transport_stencil(
        z=z, dx=dx, phi=phi, D_cell=D, w=w, bc_props=bc_dirichlet
    )
    assert ab.shape == (3, 10)
    assert b.shape == (10,)
    assert b[0] > 0.0  # Dirichlet boundary flux on RHS
    assert np.all(b[1:] == 0.0)

    # Test Robin particulate species
    bc_robin = {"type": "particulate", "top": 1e-6}
    ab_r, b_r = build_native_1d_transport_stencil(
        z=z, dx=dx, phi=phi, D_cell=D, w=w, bc_props=bc_robin
    )
    assert ab_r.shape == (3, 10)
    assert b_r[0] == 1e-6  # Bulk influx placed directly in b[0]


def test_direct_assembled_sweep():
    """Test DirectAssembledSystem sweep on a simple 1D diffusion problem."""
    dx = np.full(20, 0.01)
    mesh = Mesh1D(dx)
    z = mesh.cellCenters[0]

    mp = data_container({"phi": 0.8, "w": 0.0, "advection": 0.0})
    D_mol = data_container({"C": np.full(20, 1e-6), "D_bio": 0.0})
    bc_map = {"C": {"type": "dissolved", "top": 100.0}}

    var = VariableArray(value=0.0, name="C", mesh=mesh)
    c = data_container({"C": var})
    species_struct = [{"name": "C", "var": var}]

    system = DirectAssembledSystem(
        species_struct=species_struct,
        mesh=mesh,
        mp=mp,
        bc_map=bc_map,
        D_mol=D_mol,
        z=z,
    )

    f_res = data_container({"raw_LHS": {}, "raw_RHS": {}, "raw_CROSS": {}})
    prev_iterate = {"C": var.value.copy()}

    # Perform 5 time steps
    dt = 10.0
    for _ in range(5):
        var.updateOld()
        prev_iterate["C"][:] = var.value
        system.sweep(dt=dt, c=c, f_res=f_res, prev_iterate=prev_iterate)

    # After diffusion from top boundary (top=100), concentration should decrease monotonically with depth
    assert var.value[0] > var.value[1] > var.value[2]
    assert 0.0 < var.value[0] < 100.0
    assert np.all(var.value >= 0.0)
