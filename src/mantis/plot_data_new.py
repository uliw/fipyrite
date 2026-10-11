#!/usr/bin/env python
"""Plot diagenetic modeling data.

This module provides flexible plotting functionality for diagenetic modeling data.
The main `plot()` function can be used with default settings or with a custom
plot_description dictionary for complete control over plot structure.

It also supports overlaying measured data from a CSV file and loading plot
layouts dynamically from external Python files.

Basic Usage:
-----------
    import plot_data_new
    import pandas as pd

    df = pd.read_csv("model_output.csv")
    plot_data_new.plot(df, display_length=4, outfile="output.pdf")

Overlaying Measured Data:
------------------------
    plot_data_new.plot(df, 4, "output.pdf", measured_data_path="measured.csv")

Custom Plot Description:
-----------------------
    plot_description = {
        "first_subplot": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.c_SO4, "SO4 [mmol]", {"color": "blue"}],
                [df.c_h2s, "H2S [mmol]", {"color": "red", "linestyle": "--"}],
            ],
            "left_ylabel": "Concentration [mmol/l]",
            "right": [[df.c_o2, "O2 [μmol]", {"color": "green"}]],
            "right_ylabel": "O2 [μmol/l]",
            "options-left": ["set_ylim(-55, 30)", "exclude_y(-4, 20)"],  # Broken y-axis & methods
        },
    }
    plot_data_new.plot(df, 10, "output.pdf", plot_description=plot_description)
"""

import argparse
import importlib.util
import pathlib as pl
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use("TkAgg")

# Maximum number of right y-axes supported per subplot
MAX_RIGHT_AXES = 10


def plot(
    df,
    display_length,
    outfile,
    show=True,
    plot_description=None,
    measured_data_path=None,
    fig_handle=None,
    keep_open=False,
    title=None,
):
    """Plot data dynamically based on plot_description.

    Args
    ----
        df: DataFrame with data to plot
        display_length: Length of x-axis to display
        outfile: Output file path
        show: Whether to show the plot
        plot_description: Dictionary describing plot structure. If None, uses default structure.
        measured_data_path: Path to CSV containing measured data to overlay as scatter plots.
        fig_handle: Optional existing figure handle to reuse.
    """
    # Use default plot structure if none provided
    if plot_description is None:
        plot_description = _get_default_plot_description(df)

    # Filter for valid subplot configurations (must be dictionaries)
    valid_subplots = {k: v for k, v in plot_description.items() if isinstance(v, dict)}
    n_subplots = len(valid_subplots)

    if n_subplots == 0:
        raise ValueError("No valid subplots to create")

    # Load measured data if path provided
    df2 = _load_measured_data(measured_data_path)

    # Create figure and subplots with constrained layout
    if fig_handle is None:
        try:
            fig, axes = plt.subplots(n_subplots, 1, layout="constrained")
        except TypeError:
            fig, axes = plt.subplots(n_subplots, 1, constrained_layout=True)
        if n_subplots == 1:
            axes = [axes]
    else:
        fig = fig_handle
        for ax in fig.axes:
            try:
                ax.set_xscale("linear")
                ax.set_yscale("linear")
            except Exception:
                pass
        fig.clear()
        if hasattr(fig, "set_layout_engine"):
            fig.set_layout_engine("constrained")
        else:
            try:
                fig.set_constrained_layout(True)
            except Exception:
                pass
        axes = fig.subplots(n_subplots, 1)
        if n_subplots == 1:
            axes = [axes]

    # Get figure width from top-level or first subplot
    # Support both 'fig_width' and 'plot_width'
    fig_width = plot_description.get("fig_width", plot_description.get("plot_width"))
    if fig_width is None and valid_subplots:
        first_config = list(valid_subplots.values())[0]
        fig_width = first_config.get("fig_width", first_config.get("plot_width", 12))
    elif fig_width is None:
        fig_width = 12

    # Calculate maximum number of right axes across all subplots
    max_right_axes = 0
    # Potential keys: "right", "right1", "right2", ...
    keys_to_check = ["right"] + [f"right{i}" for i in range(1, MAX_RIGHT_AXES + 1)]
    for subplot_config in valid_subplots.values():
        n_right = sum(
            1 for key in keys_to_check if key in subplot_config and subplot_config[key]
        )
        max_right_axes = max(max_right_axes, n_right)

    # Grow figure width only for extra right axes beyond the first one
    # Each extra right axis is offset by 60 points (~0.83 inches)
    extra_right_axes = max(0, max_right_axes - 1)
    fig_width += extra_right_axes * 0.83

    if fig_handle is None:
        fig.set_size_inches(fig_width, 2 + 2 * n_subplots)

    if title is not None:
        fig.suptitle(f"{title}", fontsize=16)
    # Track all axes for xlim adjustment
    all_axes = []
    ax_objects = []

    # Process each subplot
    for idx, (subplot_key, subplot_config) in enumerate(valid_subplots.items()):
        # Setup axes for this subplot
        ax_main, right_axes = _setup_subplot_axes(axes[idx], subplot_config)
        all_axes.append(ax_main)
        ax_objects.append(ax_main)
        for rax, _, _, _ in right_axes:
            all_axes.append(rax)

        # Get x-axis data
        x_data, x_label = _get_xaxis_data(df, subplot_config)

        # Plot on left axis
        left_config = subplot_config.get("left", [])
        if left_config:
            left_lines, left_labels = _draw_series(ax_main, x_data, left_config, df2)

            # Set left axis properties
            if left_lines:
                ylabel = subplot_config.get(
                    "left_ylabel", left_labels[0] if left_labels else ""
                )
                ax_main.set_ylabel(ylabel)
                if len(left_lines) == 1:
                    ax_main.yaxis.label.set_color(left_lines[0].get_color())
                    ax_main.tick_params(axis="y", colors=left_lines[0].get_color())

        # Plot on right axes
        for twin_ax, config_key, series_idx, series in right_axes:
            lines, labels = _draw_series(twin_ax, x_data, [series], df2)
            if not lines:
                continue

            # Only set axis properties for the first series in a group
            if series_idx == 0:
                line = lines[0]
                label = labels[0]

                ylabel_key = f"{config_key}_ylabel"
                ylabel = subplot_config.get(ylabel_key, label)
                twin_ax.set_ylabel(ylabel)
                twin_ax.yaxis.label.set_color(line.get_color())
                twin_ax.tick_params(axis="y", colors=line.get_color())
                twin_ax.spines["right"].set_color(line.get_color())

        # Set x-label (usually only on bottom plot)
        if idx == n_subplots - 1:
            ax_main.set_xlabel(x_label)

        # Apply scale and any special axis properties
        if "yscale" in subplot_config:
            ax_main.set_yscale(subplot_config["yscale"])
        if "xscale" in subplot_config:
            ax_main.set_xscale(subplot_config["xscale"])
        if "xlim" in subplot_config:
            ax_main.set_xlim(subplot_config["xlim"])
        if "ylim" in subplot_config:
            ax_main.set_ylim(subplot_config["ylim"])
        if "broken_axis" in subplot_config:
            b_val = subplot_config["broken_axis"]
            if isinstance(b_val, (tuple, list)):
                broken_axis(ax_main, *b_val)
            elif isinstance(b_val, str):
                _apply_matplotlib_options(ax_main, f"broken_axis({b_val})" if "(" not in b_val else b_val)
        elif "broken_y" in subplot_config:
            b_val = subplot_config["broken_y"]
            if isinstance(b_val, (tuple, list)):
                broken_axis(ax_main, *b_val)
            elif isinstance(b_val, str):
                _apply_matplotlib_options(ax_main, f"broken_axis({b_val})" if "(" not in b_val else b_val)
        elif "exclude_y" in subplot_config:
            ex = subplot_config["exclude_y"]
            if isinstance(ex, (tuple, list)) and len(ex) >= 2:
                exclude_y(ax_main, ex[0], ex[1])
            elif isinstance(ex, str):
                _apply_matplotlib_options(ax_main, f"exclude_y({ex})" if "(" not in ex else ex)
        if "grid" in subplot_config:
            grid_config = subplot_config["grid"]
            if isinstance(grid_config, dict):
                ax_main.grid(**grid_config)
            else:
                ax_main.grid(grid_config)

        # Handle show_grid_options
        if "show_grid_options" in subplot_config:
            opts = subplot_config["show_grid_options"]
            if isinstance(opts, dict):
                g_kwargs = opts.copy()
                grid_data = g_kwargs.pop("grid", df.z if "z" in df.columns else None)
                if grid_data is not None:
                    show_grid(ax_main, grid_data, **g_kwargs)

        # Apply properties to right axes if specified (e.g., "right_yscale", "right2_ylim")
        seen_right_axes = set()
        for twin_ax, key, _, _ in right_axes:
            if twin_ax not in seen_right_axes:
                for prop in ["yscale", "xscale", "xlim", "ylim", "grid"]:
                    key_prop = f"{key}_{prop}"
                    if key_prop in subplot_config:
                        val = subplot_config[key_prop]
                        if prop == "grid":
                            if isinstance(val, dict):
                                twin_ax.grid(**val)
                            else:
                                twin_ax.grid(val)
                        else:
                            getattr(twin_ax, f"set_{prop}")(val)
                seen_right_axes.add(twin_ax)

        # Handle legend display
        _add_unified_legend(ax_main, right_axes, subplot_config)

        # Apply arbitrary matplotlib options
        _apply_all_options(ax_main, right_axes, subplot_config)

    # Adjust x-axis length for all plots that don't have an explicit xlim
    for idx, (subplot_key, subplot_config) in enumerate(valid_subplots.items()):
        if "xlim" not in subplot_config:
            ax_main = ax_objects[idx]
            if ax_main.get_xscale() == "log":
                ax_main.set_xlim(1e-4, display_length)
            else:
                ax_main.set_xlim(0, display_length)

    # Determine if constrained layout is active
    is_constrained = False
    if hasattr(fig, "get_layout_engine"):
        is_constrained = fig.get_layout_engine() is not None
    elif hasattr(fig, "get_constrained_layout"):
        is_constrained = fig.get_constrained_layout()

    if not is_constrained:
        fig.tight_layout()
    _finalize_broken_axes(fig)
    if outfile:
        # Save current size to restore it later (preserves GUI window state)
        original_size = fig.get_size_inches()

        # Set figure size strictly for PDF output to ensure independence from GUI/handle state
        fig.set_size_inches(fig_width, 2 + 2 * n_subplots)
        if not is_constrained:
            fig.tight_layout()
        _finalize_broken_axes(fig)
        fig.savefig(outfile, bbox_inches="tight")

        # Restore original size if the figure is meant to stay open or be shown
        if show or keep_open:
            fig.set_size_inches(*original_size)
            if not is_constrained:
                fig.tight_layout()
            _finalize_broken_axes(fig)

    if show:
        _finalize_broken_axes(fig)
        plt.show()
    elif not keep_open:
        plt.close(fig)

    return fig, ax_objects


def _load_measured_data(measured_data_path):
    """
    Load measured data from a CSV file.

    Args
    ----
    measured_data_path: Path to the CSV file.

    Returns
    -------
    pd.DataFrame or None: Loaded data or None if path not provided/invalid.
    """
    if not measured_data_path:
        return None

    mpath = pl.Path(measured_data_path)
    if mpath.exists():
        return pd.read_csv(mpath)

    warnings.warn(f"Measured data file not found: {measured_data_path}")
    return None


class DataFrameWrapper:
    """Wraps a DataFrame to dynamically delegate m_ attributes to a second DataFrame."""
    def __init__(self, df, df2):
        self._df = df
        self._df2 = df2

    def __getattr__(self, name):
        if name.startswith("m_") and self._df2 is not None:
            real_name = name[2:]
            if hasattr(self._df2, real_name):
                return getattr(self._df2, real_name)
            for col in self._df2.columns:
                if col.lower() == real_name.lower():
                    return self._df2[col]
        return getattr(self._df, name)

    def __getitem__(self, key):
        if isinstance(key, str) and key.startswith("m_") and self._df2 is not None:
            real_name = key[2:]
            if real_name in self._df2.columns:
                return self._df2[real_name]
            for col in self._df2.columns:
                if col.lower() == real_name.lower():
                    return self._df2[col]
        return self._df[key]

    def __len__(self):
        return len(self._df)

    @property
    def columns(self):
        return self._df.columns


def _setup_subplot_axes(ax_main, subplot_config):
    """
    Configure twin axes for a subplot based on configuration.
    Series under the same key share a twin axis.

    Args
    ----
    ax_main: The primary matplotlib axis.
    subplot_config: Configuration dictionary for this subplot.

    Returns
    -------
    tuple: (ax_main, right_axes_list) where right_axes_list is [(ax, config_key, series_idx, series), ...]
    """
    right_axes = []
    current_axis_idx = 0

    # Potential keys: "right", "right1", "right2", ...
    keys_to_check = ["right"] + [f"right{i}" for i in range(1, MAX_RIGHT_AXES + 1)]

    for key in keys_to_check:
        if key in subplot_config and subplot_config[key] is not None:
            series_list = subplot_config[key]
            if not series_list:
                continue

            # Create ONE twin axis for this key
            twin_ax = ax_main.twinx()
            # Position the spine outward based on the axis index (60 points ~ 0.83 inches)
            # The first axis (idx 0) is at the default position ("outward", 0)
            twin_ax.spines.right.set_position(("outward", 60 * current_axis_idx))

            for series_idx, series in enumerate(series_list):
                right_axes.append((twin_ax, key, series_idx, series))

            current_axis_idx += 1

    for twin_ax, _, _, _ in right_axes:
        twin_ax._is_twin_right = True
    ax_main._has_right_axes = len(right_axes) > 0

    return ax_main, right_axes


def _get_xaxis_data(df, subplot_config):
    """
    Get x-axis data and label from subplot configuration.

    Args
    ----
    df: Data source.
    subplot_config: Subplot configuration.

    Returns
    -------
    tuple: (x_data, x_label)
    """
    xaxis_config = subplot_config.get("xaxis")
    if xaxis_config is None:
        if "z" in df.columns:
            return df.z, "Depth [m]"
        return df.index, "Index"

    x_data = xaxis_config[0]
    x_label = xaxis_config[1] if len(xaxis_config) > 1 else ""
    return x_data, x_label


def _draw_series(ax, x_data, series_list, df2):
    """
    Draw lines and measured scatter points on an axis.

    Args
    ----
    ax: Matplotlib axis.
    x_data: default x-axis data.
    series_list: List of series configurations.
    df2: Measured data DataFrame.

    Returns
    -------
    tuple: (lines, labels)
    """
    lines = []
    labels = []
    for series in series_list:
        if len(series) < 2:
            continue

        # Determine if custom x-data is provided:
        # Standard: [y_data, label, kwargs] where series[1] is a string.
        # Custom:   [x_data, y_data, label, kwargs] where series[1] is an array/Series (not a string).
        has_custom_x = False
        if len(series) >= 3 and not isinstance(series[1], (str, bytes)):
            if hasattr(series[1], "__len__") or hasattr(series[1], "__array__"):
                has_custom_x = True

        if has_custom_x:
            series_x_data = series[0]
            y_data = series[1]
            label = series[2]
            kwargs = dict(series[3]) if len(series) > 3 else {}
        else:
            y_data = series[0]
            label = series[1]
            kwargs = dict(series[2]) if len(series) > 2 else {}
            
            # Legacy check: if y_data is actually from df2 (different length than default x_data),
            # use df2's z as the x-axis to prevent dimension mismatches.
            if df2 is not None and "z" in df2.columns and hasattr(y_data, "__len__") and len(y_data) != len(x_data) and len(y_data) == len(df2):
                series_x_data = df2.z
            else:
                series_x_data = x_data

        fill_only = kwargs.pop("fill_only", False)
        fill = kwargs.pop("fill", False) or fill_only
        fill_alpha = kwargs.pop("alpha", 0.3) if fill else None
        if fill_only:
            kwargs["lw"] = 0

        # When fill_only: hide line from legend, label the fill patch instead
        plot_label = "_nolegend_" if fill_only else label
        fill_label = label if fill_only else "_nolegend_"

        (line,) = ax.plot(series_x_data, y_data, label=plot_label, **kwargs)

        if fill:
            ax.fill_between(
                series_x_data, y_data,
                alpha=fill_alpha, color=line.get_color(), label=fill_label,
            )
        lines.append(line)
        labels.append(label)

    return lines, labels


def _add_unified_legend(ax_main, right_axes, subplot_config):
    """
    Collect all labels from axes and add a unified legend.

    Args
    ----
    ax_main: Primary axis.
    right_axes: List of twin axes.
    subplot_config: Subplot configuration.
    """
    if not subplot_config.get("legend", True):
        return

    raw_lines = []
    raw_labels = []

    # Main axis
    l_lines, l_labels = ax_main.get_legend_handles_labels()
    raw_lines.extend(l_lines)
    raw_labels.extend(l_labels)

    # Right axes - only collect from each unique twin axis once
    seen_axes = {ax_main}
    for twin_ax, _, _, _ in right_axes:
        if twin_ax not in seen_axes:
            r_lines, r_labels = twin_ax.get_legend_handles_labels()
            raw_lines.extend(r_lines)
            raw_labels.extend(r_labels)
            seen_axes.add(twin_ax)

    # Deduplicate by label while preserving order
    final_lines = []
    final_labels = []
    seen_labels = set()
    for line, label in zip(raw_lines, raw_labels):
        if label not in seen_labels:
            final_lines.append(line)
            final_labels.append(label)
            seen_labels.add(label)

    if final_lines:
        default_fontsize = plt.rcParams.get("legend.fontsize", 10)
        fontsize = (
            "small" if isinstance(default_fontsize, str) else default_fontsize * 0.8
        )
        target_ax = right_axes[-1][0] if right_axes else ax_main

        leg = target_ax.legend(
            final_lines,
            final_labels,
            loc="upper left",
            frameon=True,
            framealpha=0.7,
            facecolor="white",
            edgecolor="none",
            prop={"size": fontsize},
        )
        leg.set_zorder(100)


def _apply_all_options(ax_main, right_axes, subplot_config):
    """
    Apply matplotlib options to all axes in a subplot.

    Args
    ----
    ax_main: Primary axis.
    right_axes: List of twin axes.
    subplot_config: Subplot configuration.
    """
    # Left axis
    if "options-left" in subplot_config:
        _apply_matplotlib_options(ax_main, subplot_config["options-left"])
    elif "options" in subplot_config:
        _apply_matplotlib_options(ax_main, subplot_config["options"])

    # Right axes
    right_axis_map = {}
    current_idx = 0
    keys_to_check = ["right"] + [f"right{i}" for i in range(1, MAX_RIGHT_AXES + 1)]

    # Map each active key to the twin axis it was assigned (0, 1, 2...)
    for key in keys_to_check:
        if key in subplot_config and subplot_config[key]:
            right_axis_map[key] = current_idx
            current_idx += 1

    # Get distinct twin axes from the right_axes list (which is [(ax, key, s_idx, s), ...])
    # The axes were created in order of 'keys_to_check' in _setup_subplot_axes.
    unique_twin_axes = []
    seen_axes = set()
    for ax, key, s_idx, s in right_axes:
        if ax not in seen_axes:
            unique_twin_axes.append(ax)
            seen_axes.add(ax)

    # Apply options to the correct twin axis
    for key, axis_idx in right_axis_map.items():
        opt_key = f"options-{key}"
        if opt_key in subplot_config and axis_idx < len(unique_twin_axes):
            _apply_matplotlib_options(
                unique_twin_axes[axis_idx], subplot_config[opt_key]
            )


def load_layout_from_file(df, layout_path, measured_data_path=None):
    """
    Load a plot layout from a Python file.

    Args
    ----
    df: DataFrame containing the data to plot.
    layout_path: Path to the Python file containing the layout.
    measured_data_path: Optional path to the measured data CSV.

    Returns
    -------
    dict: The plot description dictionary.
    """
    df2 = _load_measured_data(measured_data_path)
    df_wrapped = DataFrameWrapper(df, df2)

    path = pl.Path(layout_path)
    if not path.exists():
        import sys
        argv0_dir = pl.Path(sys.argv[0]).parent
        fallback_path = argv0_dir / path.name
        if fallback_path.exists():
            path = fallback_path
        else:
            fallback_path2 = pl.Path("experiments") / path.name
            if fallback_path2.exists():
                path = fallback_path2
            else:
                raise FileNotFoundError(
                    f"Layout file not found: {layout_path} "
                    f"(also searched at: {fallback_path} and {fallback_path2})"
                )

    # Use importlib to load the module from a file path
    spec = importlib.util.spec_from_file_location("dynamic_layout", path)
    module = importlib.util.module_from_spec(spec)
    
    # Inject df2 and df_wrapped into module globals to support both direct access to df2
    # and legacy calls that query df.m_c_O2
    module.df = df_wrapped
    module.df2 = df2

    spec.loader.exec_module(module)

    if not hasattr(module, "get_layout"):
        raise AttributeError(
            f"Layout file {layout_path} must contain a 'get_layout(df)' function."
        )

    import inspect
    sig = inspect.signature(module.get_layout)
    has_df2_param = False
    params = list(sig.parameters.values())
    if len(params) >= 2:
        if params[1].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            has_df2_param = True
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        has_df2_param = True

    if has_df2_param:
        return module.get_layout(df_wrapped, df2)
    else:
        return module.get_layout(df_wrapped)


class BrokenYTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, axis, y1, y2, gap_ratio=0.03, split_ratio=0.5):
        super().__init__()
        self.axis = axis
        self.y1 = float(min(y1, y2))
        self.y2 = float(max(y1, y2))
        self.gap_ratio = float(gap_ratio)
        self.split_ratio = split_ratio

    def _get_params(self):
        if self.axis is not None:
            ymin, ymax = self.axis.get_view_interval()
        else:
            ymin, ymax = -55.0, 30.0

        y_low = self.y1
        y_high = self.y2

        if ymin > ymax:
            ymin, ymax = ymax, ymin

        if self.split_ratio == "proportional" or self.split_ratio is None:
            range1 = max(y_low - ymin, 1e-9)
            range2 = max(ymax - y_high, 1e-9)
            total = range1 + range2
            f1 = range1 / total
            f1 = min(max(f1, 0.2), 0.8)
        else:
            f1 = float(self.split_ratio)

        p1 = (1.0 - self.gap_ratio) * f1
        p2 = p1 + self.gap_ratio
        return ymin, ymax, y_low, y_high, p1, p2

    def transform_non_affine(self, a):
        ymin, ymax, y_low, y_high, p1, p2 = self._get_params()
        vals = np.asarray(a, dtype=float)
        res = np.zeros_like(vals)
        mask1 = vals <= y_low
        mask2 = vals >= y_high
        mask_mid = ~mask1 & ~mask2

        if y_low > ymin:
            res[mask1] = (vals[mask1] - ymin) / (y_low - ymin) * p1
        else:
            res[mask1] = 0.0

        if ymax > y_high:
            res[mask2] = p2 + (vals[mask2] - y_high) / (ymax - y_high) * (1.0 - p2)
        else:
            res[mask2] = 1.0

        if y_high > y_low:
            res[mask_mid] = p1 + (vals[mask_mid] - y_low) / (y_high - y_low) * (p2 - p1)
        else:
            res[mask_mid] = p1
        return res

    def inverted(self):
        return InvertedBrokenYTransform(self)


class InvertedBrokenYTransform(mtransforms.Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self, fwd):
        super().__init__()
        self.fwd = fwd

    def transform_non_affine(self, a):
        ymin, ymax, y_low, y_high, p1, p2 = self.fwd._get_params()
        vals = np.asarray(a, dtype=float)
        res = np.zeros_like(vals)
        mask1 = vals <= p1
        mask2 = vals >= p2
        mask_mid = ~mask1 & ~mask2

        if p1 > 0:
            res[mask1] = ymin + (vals[mask1] / p1) * (y_low - ymin)
        else:
            res[mask1] = ymin

        if (1.0 - p2) > 0:
            res[mask2] = y_high + ((vals[mask2] - p2) / (1.0 - p2)) * (ymax - y_high)
        else:
            res[mask2] = ymax

        if (p2 - p1) > 0:
            res[mask_mid] = y_low + ((vals[mask_mid] - p1) / (p2 - p1)) * (y_high - y_low)
        else:
            res[mask_mid] = y_low
        return res

    def inverted(self):
        return self.fwd


class BrokenLocator(mticker.Locator):
    def __init__(self, y1, y2, axis=None):
        super().__init__()
        self.y1 = float(min(y1, y2))
        self.y2 = float(max(y1, y2))
        self.axis = axis
        self.loc1 = mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
        self.loc2 = mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])

    def tick_values(self, vmin, vmax):
        y1 = self.y1
        y2 = self.y2
        low_min, low_max = min(vmin, y1), max(vmin, y1)
        high_min, high_max = min(vmax, y2), max(vmax, y2)

        t1 = self.loc1.tick_values(low_min, low_max)
        t1 = t1[(t1 >= low_min) & (t1 <= low_max)]

        t2 = self.loc2.tick_values(high_min, high_max)
        t2 = t2[(t2 >= high_min) & (t2 <= high_max)]

        all_ticks = np.unique(np.concatenate([t1, t2]))
        tol1 = (low_max - low_min) * 0.03 if low_max > low_min else 1e-4
        tol2 = (high_max - high_min) * 0.03 if high_max > high_min else 1e-4
        filtered = [
            t for t in all_ticks
            if (t < y1 - tol1) or (t > y2 + tol2)
        ]
        if not filtered:
            filtered = [t for t in all_ticks if t <= y1 or t >= y2]
        return np.array(filtered)

    def __call__(self):
        if hasattr(self, "axis") and self.axis is not None:
            vmin, vmax = self.axis.get_view_interval()
        else:
            vmin, vmax = -55.0, 30.0
        return self.tick_values(vmin, vmax)


class BrokenYScale(mscale.ScaleBase):
    name = "broken_y"

    def __init__(self, axis, *, y1=-4, y2=20, gap_ratio=0.03, split_ratio=0.5, **kwargs):
        super().__init__(axis)
        self.axis = axis
        self.y1 = float(min(y1, y2))
        self.y2 = float(max(y1, y2))
        self.gap_ratio = float(gap_ratio)
        self.split_ratio = split_ratio
        self._transform = BrokenYTransform(axis, self.y1, self.y2, self.gap_ratio, self.split_ratio)

    def get_transform(self):
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(BrokenLocator(self.y1, self.y2, axis=axis))
        axis.set_major_formatter(mticker.ScalarFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, vmax


mscale.register_scale(BrokenYScale)


def _update_broken_y_marks(ax):
    """Draw or update broken-axis visual marks (gap mask and slash lines) on the axis."""
    if not hasattr(ax, "_broken_y_info"):
        return

    # Clean up previous artists
    for artist in getattr(ax, "_broken_y_artists", []):
        try:
            artist.remove()
        except Exception:
            pass
    ax._broken_y_artists = []

    transform = ax.yaxis.get_transform()
    if not hasattr(transform, "_get_params"):
        return

    ymin, ymax, y_low, y_high, p1, p2 = transform._get_params()

    bg_color = ax.get_facecolor()
    if isinstance(bg_color, tuple) and len(bg_color) == 4 and bg_color[3] == 0:
        bg_color = ax.figure.get_facecolor() if ax.figure else "white"
        if isinstance(bg_color, tuple) and len(bg_color) == 4 and bg_color[3] == 0:
            bg_color = "white"

    # Mask patch covering the excluded gap across the plot width
    patch = mpatches.Rectangle(
        (0.0, p1),
        1.0,
        p2 - p1,
        transform=ax.transAxes,
        facecolor=bg_color,
        edgecolor="none",
        zorder=49,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax._broken_y_artists.append(patch)

    is_twin_right = getattr(ax, "_is_twin_right", False)
    if is_twin_right:
        sides = [1.0]
    else:
        has_right_axes = getattr(ax, "_has_right_axes", False)
        sides = [0.0] if has_right_axes else [0.0, 1.0]

    fig = ax.figure
    bbox = ax.get_window_extent()
    dpi = fig.dpi if fig else 100
    if bbox.width > 0 and bbox.height > 0:
        w_pts = bbox.width / dpi * 72
        h_pts = bbox.height / dpi * 72
    else:
        w_pts = fig.get_figwidth() * 72 * 0.7 if fig else 400
        h_pts = fig.get_figheight() * 72 * 0.7 if fig else 200

    dx = 5.0 / max(w_pts, 1.0)
    dy = 3.5 / max(h_pts, 1.0)

    for x_pos in sides:
        spine_name = "right" if x_pos == 1.0 else "left"
        spine_color = "k"
        spine_lw = 1.0
        if spine_name in ax.spines:
            spine = ax.spines[spine_name]
            spine_color = spine.get_edgecolor()
            spine_lw = spine.get_linewidth() or 1.0

        # Spine gap line (colored like background)
        (gap_line,) = ax.plot(
            [x_pos, x_pos],
            [p1, p2],
            color=bg_color,
            lw=spine_lw + 2.5,
            transform=ax.transAxes,
            zorder=50,
            clip_on=False,
        )
        ax._broken_y_artists.append(gap_line)

        # Diagonal slash marks at p1 and p2
        for y_pos in [p1, p2]:
            (slash,) = ax.plot(
                [x_pos - dx, x_pos + dx],
                [y_pos - dy, y_pos + dy],
                color=spine_color,
                lw=spine_lw,
                transform=ax.transAxes,
                zorder=51,
                clip_on=False,
            )
            ax._broken_y_artists.append(slash)


def _finalize_broken_axes(fig):
    """Update all broken axis marks in the figure to reflect final layout and limits."""
    for ax in fig.axes:
        if hasattr(ax, "_broken_y_info"):
            _update_broken_y_marks(ax)


def exclude_y(ax, y1, y2, gap_ratio=0.03, split_ratio=0.5, **kwargs):
    """Set a broken y-axis on an axis, excluding the interval [y1, y2].

    Args
    ----
    ax: Matplotlib Axes object
    y1, y2: Boundary values of the interval to exclude.
    gap_ratio: Fraction of the axis height allocated to the visual break gap (default: 0.03).
    split_ratio: Height fraction allocated to the lower section (default: 0.5 for equal split,
                 or 'proportional' for proportional to data range).
    """
    y_low = min(float(y1), float(y2))
    y_high = max(float(y1), float(y2))

    cur_ylim = ax.get_ylim()

    ax.set_yscale(
        "broken_y",
        y1=y_low,
        y2=y_high,
        gap_ratio=gap_ratio,
        split_ratio=split_ratio,
        **kwargs,
    )

    if cur_ylim != (0.0, 1.0):
        ax.set_ylim(cur_ylim)

    ax._broken_y_info = {
        "y1": y_low,
        "y2": y_high,
        "gap_ratio": gap_ratio,
        "split_ratio": split_ratio,
    }

    if not hasattr(ax, "_broken_y_cid"):
        ax._broken_y_cid = ax.callbacks.connect(
            "ylim_changed", lambda a: _update_broken_y_marks(a)
        )

    _update_broken_y_marks(ax)


def broken_axis(ax, *args, **kwargs):
    """Configure a broken axis by specifying the two intervals to display.

    Usage:
        broken_axis(ax, -45, -35, 20, 30)
        broken_axis(ax, (-45, -35), (20, 30))
        broken_axis(ax, -45, -35, 20, 30, split_ratio=0.5, gap_ratio=0.03)

    Args:
        ax: Matplotlib Axes object.
        *args: Either 4 numbers (y1_min, y1_max, y2_min, y2_max)
               or 2 tuples/lists ((y1_min, y1_max), (y2_min, y2_max)).
        gap_ratio: Fraction of axis height for the break gap (default: 0.03).
        split_ratio: Height fraction for the lower section (default: 0.5 for equal split,
                     or 'proportional').
    """
    gap_ratio = kwargs.pop("gap_ratio", 0.03)
    split_ratio = kwargs.pop("split_ratio", 0.5)

    if len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 4:
        y1_min, y1_max, y2_min, y2_max = args[0]
    elif len(args) == 2 and isinstance(args[0], (tuple, list)) and isinstance(args[1], (tuple, list)):
        y1_min, y1_max = args[0]
        y2_min, y2_max = args[1]
    elif len(args) >= 4:
        y1_min, y1_max, y2_min, y2_max = args[:4]
    else:
        raise ValueError(
            "broken_axis requires 4 boundary values or 2 pairs defining intervals: "
            "e.g. broken_axis(-45, -35, 20, 30) or broken_axis((-45, -35), (20, 30))"
        )

    interval_a = (min(float(y1_min), float(y1_max)), max(float(y1_min), float(y1_max)))
    interval_b = (min(float(y2_min), float(y2_max)), max(float(y2_min), float(y2_max)))
    if interval_a[0] > interval_b[0]:
        interval_a, interval_b = interval_b, interval_a

    ymin, y_low = interval_a
    y_high, ymax = interval_b

    ax.set_ylim(ymin, ymax)
    exclude_y(ax, y_low, y_high, gap_ratio=gap_ratio, split_ratio=split_ratio, **kwargs)


broken_y = broken_axis

# Attach as methods to matplotlib.axes.Axes
plt.Axes.exclude_y = exclude_y
plt.Axes.broken_axis = broken_axis
plt.Axes.broken_y = broken_axis


def _apply_matplotlib_options(ax, options):
    """Apply arbitrary matplotlib method calls to an axis.

    Args
    ----
    ax: Matplotlib axis object
    options: String or list/tuple containing matplotlib method calls.
             Separated by commas if multiple calls are within a single string.
             Examples:
                 "set_ylim(1e-10, 1e-5), set_title('My Title')"
                 ["broken_axis(-45, -35, 20, 30)"]
                 ["set_ylim(-55, 30)", "exclude_y(-4, 20)"]

    The function safely parses and executes each method call on the provided axis.
    Each method call should be in the format: method_name(arg1, arg2, ...)
    Multiple calls can be separated by commas.
    """
    if not options:
        return

    if isinstance(options, str):
        raw_items = [options]
    elif isinstance(options, (list, tuple)):
        raw_items = list(options)
    else:
        raw_items = [str(options)]

    method_calls = []
    for item in raw_items:
        if not isinstance(item, str):
            item = str(item)
        item = item.strip()
        if not item:
            continue

        # Split by comma to get individual method calls, respecting parentheses
        current_call = ""
        paren_depth = 0
        for char in item:
            if char == "(":
                paren_depth += 1
                current_call += char
            elif char == ")":
                paren_depth -= 1
                current_call += char
            elif char == "," and paren_depth == 0:
                if current_call.strip():
                    method_calls.append(current_call.strip())
                current_call = ""
            else:
                current_call += char
        if current_call.strip():
            method_calls.append(current_call.strip())

    # Execute each method call
    for call in method_calls:
        call = call.strip()
        if not call:
            continue

        # Parse method name and arguments
        if "(" not in call:
            method_name = call
            args_str = ""
        else:
            method_name = call[: call.index("(")].strip()
            args_str = call[call.index("(") + 1 : call.rindex(")")].strip()

        # Intercept broken_axis / broken_y
        if method_name in ("broken_axis", "broken_y"):
            try:
                safe_dict = {
                    "ax": ax,
                    "broken_axis": broken_axis,
                    "broken_y": broken_axis,
                    "__builtins__": {},
                }
                eval(f"broken_axis(ax, {args_str})", safe_dict)
            except Exception as e:
                warnings.warn(f"Failed to execute {method_name}({args_str}): {e}")
            continue

        # Intercept exclude_y
        if method_name == "exclude_y":
            try:
                safe_dict = {"ax": ax, "exclude_y": exclude_y, "__builtins__": {}}
                eval(f"exclude_y(ax, {args_str})", safe_dict)
            except Exception as e:
                warnings.warn(f"Failed to execute exclude_y({args_str}): {e}")
            continue

        # Check if method exists on axis
        if not hasattr(ax, method_name):
            warnings.warn(f"Axis does not have method '{method_name}', skipping")
            continue

        # Execute the method call using eval in a context where 'ax' is local
        try:
            safe_dict = {"ax": ax, "__builtins__": {}}
            eval(f"ax.{method_name}({args_str})", safe_dict)
        except Exception as e:
            warnings.warn(f"Failed to execute {method_name}({args_str}): {e}")


def _get_default_plot_description(df):
    """Generate default plot description for backward compatibility.

    This function creates a plot_description dictionary that replicates
    the original hard-coded plot structure from the legacy implementation.

    Args
    ----
    df: DataFrame containing the data to plot

    Returns
    -------
    dict: A plot_description dictionary with the default plot structure:
        - First subplot: Concentrations (SO4, H2S, FeS2, O2, OM, Fe)
        - Second subplot (if isotopes=True): Isotope deltas (dSO4, dH2S, dFeS2)
        - Last subplot: Reaction rates (f_o2, f_SO4, f_FeS2, f_h2s, f_poc)
    """
    plot_desc = {
        "first": {
            "xaxis": [df.z, "Depth [m]"],
            "left": [
                [df.c_SO4, r"SO$_{4}$", {"color": "C0"}],
                [
                    df.c_h2s,
                    r"H$_{2}$S",
                    {
                        "color": "C0",
                        "linestyle": (0, (0.1, 2)),
                        "dash_capstyle": "round",
                    },
                ],
            ],
            "left_ylabel": r"SO$_{4}$ & H$_{2}$S  [mmol/l]",
            "right": [[df.c_FeS2, r"FeS$_{2}$", {"color": "k"}]],
            "right_ylabel": r"FeS$_{2}$ [mmol/l]",
            "right2": [[df.c_o2, r"O$_{2}$", {"color": "C2"}]],
            "right2_ylabel": r"O$_{2}$ [$\mu$mol/l]",
            "right3": [[df.c_poc, "OM [mol/l]", {"color": "C1"}]],
            "right3_ylabel": "OM [mol/l]",
            "right4": [[df.c_Fe3, "Fe [mol/l]", {"color": "C3"}]],
            "right4_ylabel": "Fe [mol/l]",
        },
    }

    # Reaction rates plot
    subplot_key = "second"
    plot_desc[subplot_key] = {
        "xaxis": [df.z, "Depth [m]"],
        "left": [
            [df.f_o2, "f_o2", {"color": "C2"}],
            [df.f_SO4, "f_SO4", {"color": "C0"}],
            [df.f_FeS2, "f_FeS2", {"color": "k"}],
            [df.f_h2s, "f_h2s", {"color": "C0", "linestyle": ":"}],
            [df.f_poc, "f_poc", {"color": "C1"}],
        ],
        "left_ylabel": r"f [mol/m$^{3}$ s$^{-1}$]",
        "yscale": "log",
        "legend": True,
    }

    return plot_desc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot diagenetic modeling results with optional measured data overlay.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("input_file", help="Path to the model output CSV file.")
    parser.add_argument(
        "-d",
        "--display-length",
        dest="display_length",
        default=0,
        type=float,
        help="Depth limit for the x-axis (default: full length of data).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        default=None,
        type=str,
        help="Path for the output PDF (default: <input_file_stem>.pdf).",
    )
    parser.add_argument(
        "-m",
        "--measured-data",
        dest="measured_data",
        default=None,
        type=str,
        help="Path to CSV containing measured data to overlay as scatter points.",
    )
    parser.add_argument(
        "-l",
        "--layout",
        dest="layout_file",
        default=None,
        type=str,
        help=(
            "Path to a Python file defining the plot layout.\n"
            "The file must contain a 'get_layout(df)' function.\n\n"
            "Example layout file content:\n"
            "--------------------------------------------------\n"
            "def get_layout(df):\n"
            "    return {\n"
            "        'subplot1': {\n"
            "            'xaxis': [df.z, 'Depth [m]'],\n"
            "            'left':  [[df.c_SO4, 'SO4', {'color': 'C0'}]]\n"
            "        }\n"
            "    }\n"
            "--------------------------------------------------"
        ),
    )
    parser.add_argument(
        "--hide",
        action="store_false",
        dest="show",
        help="If set, do not show the plot window, only save the PDF.",
    )
    parser.set_defaults(show=True)

    args = parser.parse_args(argv)

    input_path = pl.Path(args.input_file)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Determine display length
    if args.display_length > 0:
        display_length = args.display_length
    elif "z" in df.columns:
        display_length = df.z.iat[-1]
    else:
        display_length = len(df)

    # Determine output file
    if args.output_file:
        outfile = pl.Path(args.output_file)
    else:
        outfile = input_path.with_suffix(".pdf")

    # Load custom layout if provided
    plt_desc = None
    if args.layout_file:
        plt_desc = load_layout_from_file(df, args.layout_file, args.measured_data)

    plot(
        df,
        display_length,
        outfile,
        show=args.show,
        plot_description=plt_desc,
        measured_data_path=args.measured_data,
    )
    print(f"Plot generated: {outfile}")


if __name__ == "__main__":
    main()


def show_grid(
    ax, grid, step=50, thickness="0.1 pt", color="lightgrey", alpha=0.3, **kwargs
):
    """Plot a vertical line each multiple of a mesh coordinate.

    Add this to the plot referenced by ax

    Line thickness and color, and transparency can be modified by the
    above parameters.
    """
    import numpy as np

    # Handle thickness -> linewidth conversion
    if "linewidth" in kwargs:
        lw = kwargs.pop("linewidth")
    elif "lw" in kwargs:
        lw = kwargs.pop("lw")
    else:
        lw = thickness
        if isinstance(thickness, str):
            if "pt" in thickness:
                lw = float(thickness.replace("pt", "").strip())
            else:
                try:
                    lw = float(thickness)
                except ValueError:
                    lw = 0.1

    # Get defaults for color and alpha if not in kwargs
    c = kwargs.pop("color", color)
    a = kwargs.pop("alpha", alpha)

    # Get z values from grid (could be array or fipy mesh)
    if hasattr(grid, "cellCenters"):
        z_vals = np.array(grid.cellCenters[0])
    elif hasattr(grid, "value"):
        z_vals = np.array(grid.value)
    else:
        z_vals = np.array(grid)

    # Plot vertical lines at every 'step' index
    for i in range(0, len(z_vals), step):
        ax.axvline(z_vals[i], color=c, linewidth=lw, alpha=a, zorder=-1, **kwargs)
