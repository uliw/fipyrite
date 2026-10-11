"""test_diagenetic_model.py

Unit and regression tests for DiageneticModel interface in fipyrite.model.
"""

from pathlib import Path
import pytest
from fipyrite import DiageneticModel


def test_diagenetic_model_experiment_name_derivation():
    model = DiageneticModel(
        species="nbk/experiments_new/species.py",
        reactions="nbk/experiments_new/fes_reactions.py",
        constants="nbk/experiments_new/kinetic_constants_slow.py",
    )
    # Since caller frame is test_diagenetic_model.py, stem should be test_diagenetic_model
    assert model.experiment == "test_diagenetic_model"


def test_diagenetic_model_cli_empty_arg_query(capsys):
    model = DiageneticModel(
        species="nbk/experiments_new/species.py",
        reactions="nbk/experiments_new/fes_reactions.py",
        constants="nbk/experiments_new/kinetic_constants_slow.py",
        experiment="test_exp",
    )
    with pytest.raises(SystemExit) as exc_info:
        model.parse_cli_args(["--solver"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--solver" in captured.out
    assert "coupled" in captured.out or "monolithic" in captured.out


def test_diagenetic_model_setup(tmp_path):
    eq_file = tmp_path / "test_setup_exp_equations.py"
    model = DiageneticModel(
        species="nbk/experiments_new/species.py",
        reactions="nbk/experiments_new/fes_reactions.py",
        constants="nbk/experiments_new/kinetic_constants_slow.py",
        equations=eq_file,
        layout="nbk/experiments_new/plot_layout.py",
        experiment="test_setup_exp",
    )
    model.configure(
        max_depth="0.1 m",
        display_length="0.1 m",
        t_end="1 day",
        phi=0.8,
        bioirrigation={"BI0": 0.0, "depth": "0 cm"},
    )
    mp, c, k, species_transport, z, D_mol, rxn_fn = model.setup()
    assert len(species_transport) > 0
    assert hasattr(c, "SO4")
    assert hasattr(c, "Fe3")
    assert hasattr(c, "TS2")
    assert hasattr(mp, "get_limiters")
    assert callable(mp.get_limiters)
