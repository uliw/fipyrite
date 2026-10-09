import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fipyrite.plot_data_new import (
    BrokenLocator,
    BrokenYScale,
    _apply_matplotlib_options,
    exclude_y,
    plot,
)


def test_options_left_string_and_list():
    fig, ax = plt.subplots()

    # String format
    _apply_matplotlib_options(ax, "set_ylim(0, 50), set_title('String Title')")
    assert ax.get_ylim() == (0.0, 50.0)
    assert ax.get_title() == "String Title"

    # List format
    _apply_matplotlib_options(ax, ["set_ylim(-10, 40)", "set_title('List Title')"])
    assert ax.get_ylim() == (-10.0, 40.0)
    assert ax.get_title() == "List Title"
    plt.close(fig)


def test_exclude_y_in_options(tmp_path):
    df = pd.DataFrame({
        "z": np.linspace(0, 1, 50),
        "c_SO4": np.linspace(20, 25, 50),
        "d_SO4": np.linspace(21, 24, 50),
        "d_TS2": np.linspace(-45, -42, 50),
    })

    plot_description = {
        "panel1": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.d_SO4, "SO4", {"color": "C0"}],
                [df.d_TS2, "TS2", {"color": "C1"}],
            ],
            "options-left": ["set_ylim(-55, 30)", "exclude_y(-4, 20)"],
        }
    }

    outfile = tmp_path / "test_broken.pdf"
    fig, axes = plot(
        df,
        display_length=1.0,
        outfile=outfile,
        show=False,
        plot_description=plot_description,
    )

    ax = axes[0]
    assert hasattr(ax, "_broken_y_info")
    assert ax._broken_y_info["y1"] == -4.0
    assert ax._broken_y_info["y2"] == 20.0
    assert outfile.exists()
    plt.close(fig)


def test_exclude_y_direct_key(tmp_path):
    df = pd.DataFrame({
        "z": np.linspace(0, 1, 50),
        "val": np.linspace(-50, 50, 50),
    })

    plot_description = {
        "panel1": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [[df.val, "val", {}]],
            "ylim": (-60, 60),
            "exclude_y": (-10, 15),
        }
    }

    outfile = tmp_path / "test_direct_key.pdf"
    fig, axes = plot(
        df,
        display_length=1.0,
        outfile=outfile,
        show=False,
        plot_description=plot_description,
    )

    ax = axes[0]
    assert hasattr(ax, "_broken_y_info")
    assert ax._broken_y_info["y1"] == -10.0
    assert ax._broken_y_info["y2"] == 15.0
    plt.close(fig)


def test_broken_locator_ticks():
    fig, ax = plt.subplots()
    ax.set_ylim(-55, 30)
    exclude_y(ax, -4, 20)

    ticks = ax.yaxis.get_major_locator()()
    # Ticks should be in [-55, -4] and [20, 30], and NOT in (-4, 20)
    assert all(t <= -4 or t >= 20 for t in ticks)
    # Check that there are ticks on both sides of the gap
    assert any(t <= -4 for t in ticks)
    assert any(t >= 20 for t in ticks)
    plt.close(fig)


def test_broken_axis_options(tmp_path):
    df = pd.DataFrame({
        "z": np.linspace(0, 1, 50),
        "d_SO4": np.linspace(21, 24, 50),
        "d_TS2": np.linspace(-45, -42, 50),
    })

    # Exact syntax requested by user: ["broken_axis(-45, -35, 20, 30)"]
    plot_description = {
        "panel1": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.d_SO4, "SO4", {}],
                [df.d_TS2, "TS2", {}],
            ],
            "options-left": ["broken_axis(-45, -35, 20, 30)"],
        }
    }

    outfile = tmp_path / "test_broken_axis.pdf"
    fig, axes = plot(
        df,
        display_length=1.0,
        outfile=outfile,
        show=False,
        plot_description=plot_description,
    )

    ax = axes[0]
    assert hasattr(ax, "_broken_y_info")
    # Overall range: (-45, 30)
    assert ax.get_ylim() == (-45.0, 30.0)
    # Excluded range: (-35, 20)
    assert ax._broken_y_info["y1"] == -35.0
    assert ax._broken_y_info["y2"] == 20.0

    ticks = ax.yaxis.get_major_locator()()
    assert all(t <= -35 or t >= 20 for t in ticks)
    assert any(t <= -35 for t in ticks)
    assert any(t >= 20 for t in ticks)
    assert outfile.exists()
    plt.close(fig)


def test_broken_axis_direct_key_and_tuples(tmp_path):
    df = pd.DataFrame({
        "z": np.linspace(0, 1, 50),
        "val": np.linspace(-45, 30, 50),
    })

    plot_description = {
        "panel1": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [[df.val, "val", {}]],
            "broken_axis": ((-45, -35), (20, 30)),
        }
    }

    outfile = tmp_path / "test_broken_axis_tuples.pdf"
    fig, axes = plot(
        df,
        display_length=1.0,
        outfile=outfile,
        show=False,
        plot_description=plot_description,
    )

    ax = axes[0]
    assert ax.get_ylim() == (-45.0, 30.0)
    assert ax._broken_y_info["y1"] == -35.0
    assert ax._broken_y_info["y2"] == 20.0
    plt.close(fig)


def test_broken_axis_method():
    fig, ax = plt.subplots()
    ax.broken_axis(-45, -35, 20, 30)
    assert ax.get_ylim() == (-45.0, 30.0)
    assert ax._broken_y_info["y1"] == -35.0
    assert ax._broken_y_info["y2"] == 20.0
    plt.close(fig)


def test_species_delta_thresholding():
    from fipyrite.diff_lib import get_delta, get_species_delta_threshold, is_solid_species

    class DummyMP:
        phi = 0.8
        VCDT = 0.045
        bc_map = {
            "FeS": {"type": "particulate"},
            "SO4": {"type": "dissolved"},
        }

    mp = DummyMP()

    # Verify solid vs liquid classification
    assert is_solid_species("FeS", mp) is True
    assert is_solid_species("SO4", mp) is False
    assert is_solid_species("FeS2", mp) is True
    assert is_solid_species("TS2", mp) is False

    # Verify default threshold values
    # For liquids: 0.001 mmol/L
    assert get_species_delta_threshold("SO4", mp) == 0.001
    # For solids: 0.001 * phi / (1 - phi) = 0.001 * 0.8 / 0.2 = 0.004 mmol/L
    np.testing.assert_allclose(get_species_delta_threshold("FeS", mp), 0.004)

    # Test species-specific override: e.g. TS2_delta_threshold and FeS_delta_threshold
    mp.TS2_delta_threshold = 0.0005
    assert get_species_delta_threshold("TS2", mp) == 0.0005
    assert get_species_delta_threshold("TS2_32", mp) == 0.0005

    mp.FeS_delta_threshold = 0.05
    assert get_species_delta_threshold("FeS", mp) == 0.05
    assert get_species_delta_threshold("FeS_32", mp) == 0.05

    # Verify get_delta threshold masking
    c_total = np.array([0.0001, 0.002, 0.05])
    c_32 = c_total / (1.0 + DummyMP.VCDT)  # zero delta

    # Using liquid threshold 0.001: only index 0 is below threshold
    d_liq = get_delta(c_total, c_32, DummyMP.VCDT, threshold=0.001)
    assert np.isnan(d_liq[0])
    assert not np.isnan(d_liq[1])
    assert not np.isnan(d_liq[2])

    # Using solid threshold 0.004: indices 0 and 1 are below threshold
    d_sol = get_delta(c_total, c_32, DummyMP.VCDT, threshold=0.004)
    assert np.isnan(d_sol[0])
    assert np.isnan(d_sol[1])
    assert not np.isnan(d_sol[2])


