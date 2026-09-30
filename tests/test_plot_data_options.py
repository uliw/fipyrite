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
