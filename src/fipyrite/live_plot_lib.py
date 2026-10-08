from __future__ import annotations

import multiprocessing as mp
import time
import queue
import signal
import sys
from typing import TYPE_CHECKING, Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    import pathlib as pl


def parse_time_to_seconds(val: Any) -> Optional[float]:
    """
    Parse a time expression into seconds (float).
    Supports Pint Quantity, string expressions (e.g. '1 year', '10 days'),
    numeric seconds (int, float), or None.
    """
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in ("none", "false", ""):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if hasattr(val, "to") and hasattr(val, "magnitude"):
        return float(val.to("seconds").magnitude)
    if isinstance(val, str):
        import pint
        ureg = pint.UnitRegistry()
        return float(ureg.Quantity(val).to("seconds").magnitude)
    return float(val)


class BoundedVideoQueue:
    """
    A process-safe bounded queue with backpressure for video encoding.
    Capacity defaults to maxsize=20.
    When full (count >= maxsize), put() blocks until count <= resume_threshold
    (defaults to 10, i.e. half empty).
    """

    def __init__(
        self,
        maxsize: int = 20,
        resume_threshold: int = 10,
        ctx: Optional[mp.context.BaseContext] = None,
    ):
        if ctx is None:
            ctx = mp.get_context("spawn")
        self.ctx = ctx
        self.maxsize = maxsize
        self.resume_threshold = resume_threshold
        self._queue = ctx.Queue()
        self._lock = ctx.Lock()
        self._cond = ctx.Condition(self._lock)
        self._count = ctx.Value("i", 0)
        self._stop_event = ctx.Event()

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Puts an item into the queue. If queue is full (>= maxsize), blocks
        until occupancy drops to <= resume_threshold.
        Sentinel item None bypasses occupancy check so shutdown never deadlocks.
        """
        if item is None:
            self._queue.put(None)
            return True

        start_time = time.time()
        with self._cond:
            while self._count.value >= self.maxsize:
                if self._stop_event.is_set():
                    return False
                if not block:
                    raise queue.Full
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        raise queue.Full
                    wait_time = min(0.2, remaining)
                else:
                    wait_time = 0.2
                self._cond.wait(timeout=wait_time)
                if self._stop_event.is_set():
                    return False

            self._queue.put(item)
            self._count.value += 1
            return True

    def put_nowait(self, item: Any) -> bool:
        return self.put(item, block=False)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """
        Gets an item from the queue and decrements occupancy.
        If occupancy drops <= resume_threshold, wakes up waiting producers.
        """
        item = self._queue.get(block=block, timeout=timeout)
        if item is not None:
            with self._cond:
                self._count.value = max(0, self._count.value - 1)
                if self._count.value <= self.resume_threshold:
                    self._cond.notify_all()
        return item

    def get_nowait(self) -> Any:
        return self.get(block=False)

    def qsize(self) -> int:
        with self._lock:
            return self._count.value

    def empty(self) -> bool:
        with self._lock:
            return self._count.value == 0

    def full(self) -> bool:
        with self._lock:
            return self._count.value >= self.maxsize

    def stop(self) -> None:
        """Signal queue to unblock any waiting producers."""
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()

    def cancel_join_thread(self) -> None:
        try:
            self._queue.cancel_join_thread()
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        try:
            self._queue.cancel_join_thread()
            self._queue.close()
        except Exception:
            pass


class LivePlotter:
    """Manages a background process for real-time plotting via a bounded Queue."""

    def __init__(
        self,
        layout_path: str,
        display_length: float,
        measured_data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        video_path: Optional[str] = None,
        fps: int = 15,
        title: Optional[str] = None,
        gui: bool = False,
        report_step: int = 1,
        max_queue_size: int = 20,
        resume_threshold: int = 10,
    ):
        self.layout_path = layout_path
        self.display_length = display_length
        self.measured_data_path = measured_data_path
        self.output_path = output_path
        self.video_path = video_path
        self.fps = fps
        self.codec = "libvpx-vp9"
        self.gui = gui
        self.report_step = report_step
        self.max_queue_size = max_queue_size
        self.resume_threshold = resume_threshold
        # Use 'spawn' to avoid inheriting PETSc/MPI signal handlers and state
        self._ctx = mp.get_context("spawn")
        self._queue = BoundedVideoQueue(
            maxsize=max_queue_size,
            resume_threshold=resume_threshold,
            ctx=self._ctx,
        )
        self._process: Optional[mp.Process] = None
        self.title = title

    @property
    def queue(self):
        return self._queue

    def start(self) -> None:
        """Launch the background plotting process."""
        self._process = self._ctx.Process(target=self._run_plot_loop, daemon=False)
        self._process.start()

    def stop(self) -> None:
        """Stop the background plotting process."""
        if self._process and self._process.is_alive():
            try:
                self._queue.put(None, timeout=2)  # Sentinel for exit
            except Exception:
                pass
            self._process.join(timeout=15)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
        # Prevent Python atexit from hanging on queue flush thread
        try:
            self._queue.cancel_join_thread()
            self._queue.close()
        except Exception:
            pass

    def _run_plot_loop(self) -> None:
        """Internal loop running in the background process."""
        import os

        # Cache parent PID so we can detect if parent process dies
        parent_pid = os.getppid()

        # Establish a new process group for the child process.
        # This prevents Ctrl-C (SIGINT) sent to the parent's terminal process group
        # from propagating to the child process and its spawned ffmpeg subprocess.
        try:
            os.setpgrp()
        except OSError as e:
            print(f"[LivePlotter] Failed to set process group: {e}", flush=True)

        # Ignore SIGINT (Ctrl-C) in child process to allow clean parent-initiated shutdown
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Handle SIGTERM cleanly to avoid PETSc signal handler dumping MPI_Abort
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

        # Reset signal handlers to default to avoid PETSc's SIGPIPE handling
        if hasattr(signal, "SIGPIPE"):
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)

        import matplotlib

        if self.gui:
            matplotlib.use("TkAgg")
        else:
            matplotlib.use("Agg")
        import fipyrite.plot_data_new as plot_data_new
        from matplotlib.animation import FFMpegWriter

        # Re-assert clean SIGTERM handler after imports in case libraries modified it
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

        print(
            f"[LivePlotter] Child process starting. video_path={self.video_path}",
            flush=True,
        )

        fig = None
        ax_objects = None
        last_df = None
        plt_desc = None
        writer = None
        last_title = None

        print(
            f"[LivePlotter] Background process started (Animation: {self.video_path is not None})."
        )

        writer = None

        print(
            f"[LivePlotter] Background process started (Animation: {self.video_path is not None})."
        )

        try:

            def process_data_item(data_item) -> bool:
                """Processes a single data item. Returns True if we should stop."""
                nonlocal fig, ax_objects, last_df, plt_desc, writer, last_title
                if data_item is None:
                    print("[LivePlotter] Termination signal received.")
                    if fig and last_df is not None and self.output_path:
                        print(f"[LivePlotter] Saving final plot to {self.output_path}")
                        plot_data_new.plot(
                            last_df,
                            self.display_length,
                            outfile=self.output_path,
                            show=False,
                            fig_handle=fig,
                            plot_description=plt_desc,
                            measured_data_path=self.measured_data_path,
                            keep_open=True,
                            title=last_title or self.title,
                        )
                    return True

                data, title = data_item
                last_title = title
                df = pd.DataFrame(data)
                last_df = df

                try:
                    plt_desc = plot_data_new.load_layout_from_file(df, self.layout_path, self.measured_data_path)
                    outfile_path = None
                    if fig is None:
                        fig, ax_objects = plot_data_new.plot(
                            df,
                            self.display_length,
                            outfile=outfile_path,
                            show=False,
                            plot_description=plt_desc,
                            measured_data_path=self.measured_data_path,
                            keep_open=True,
                            title=title or self.title,
                        )
                    else:
                        plot_data_new.plot(
                            df,
                            self.display_length,
                            outfile=outfile_path,
                            show=False,
                            fig_handle=fig,
                            plot_description=plt_desc,
                            measured_data_path=self.measured_data_path,
                            keep_open=True,
                            title=title or self.title,
                        )
                except Exception as e:
                    if "invalid command name" not in str(e):
                        print(f"[LivePlotter] Plot update error: {e}")
                    fig = None

                if fig:
                    if writer is None and self.video_path:
                        print(
                            f"[LivePlotter] Initializing FFMpegWriter for {self.video_path}...",
                            flush=True,
                        )
                        try:
                            writer = FFMpegWriter(
                                fps=self.fps,
                                metadata=dict(artist="LivePlotter"),
                            )
                        except Exception as e:
                            print(
                                f"[LivePlotter] Failed to initialize FFMpegWriter: {e}. Falling back to GUI.",
                                flush=True,
                            )
                            writer = None

                    if writer:
                        if not hasattr(writer, "_saving"):
                            print(
                                f"[LivePlotter] Setting up writer for {self.video_path}...",
                                flush=True,
                            )
                            writer.setup(fig, self.video_path, dpi=300)
                            writer._saving = True
                        try:
                            writer.grab_frame()
                        except Exception as ge:
                            print(f"[LivePlotter] Grab frame error: {ge}", flush=True)

                    if self.gui:
                        try:
                            # Only setup GUI if not in video mode
                            plt.ion()
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                        except Exception as e:
                            print(f"[LivePlotter] Draw error: {e}")
                            fig = None
                        plt.pause(0.01)
                return False

            while True:
                try:
                    data_item = self._queue.get(timeout=0.1)
                    if process_data_item(data_item):
                        break
                except queue.Empty:
                    if fig and not writer:
                        plt.pause(0.01)
                    # Check if parent process has died
                    if os.getppid() != parent_pid:
                        print("[LivePlotter] Parent process died. Exiting.")
                        break
                    continue
                except Exception as e:
                    print(f"[LivePlotter] Loop error: {e}")
                    break

        except KeyboardInterrupt:
            pass
        finally:
            if writer and hasattr(writer, "_saving"):
                print("[LivePlotter] Finishing writer...", flush=True)
                writer.finish()
                print("[LivePlotter] Writer finished.", flush=True)
            if not self.video_path:
                plt.ioff()
            plt.close("all")
            print("[LivePlotter] Background process exiting.")


def capture_state(
    mp_params: Any,
    c: Any,
    k: Any,
    species_list: list[str],
    z: Any,
    D_mol: Any,
    diagenetic_reactions: Any,
    equilibrium_reactions: Any,
    current_dt: float,
) -> dict[str, np.ndarray]:
    """
    Capture the current state of the model as a dictionary of numpy arrays.
    """
    from . import diff_lib

    # 1. Capture current rates in the main thread (thread-safe snap)
    c_numpy = diff_lib.data_container({s: diff_lib.ArrayProxy(val.value) for s, val in c.items()})
    mp_numpy = diff_lib.data_container(mp_params)
    mp_numpy.phi = diff_lib.ArrayProxy(mp_params.phi.value)
    f_final, RATES = diagenetic_reactions(mp_numpy, c_numpy, k, diff_lib.data_container())
    f_final, RATES = equilibrium_reactions(mp_params, c, k, f_final, RATES, current_dt)

    # 2. Snapshot values (numpy arrays)
    def snap(obj):
        if hasattr(obj, "value"):
            return obj.value.copy()
        if hasattr(obj, "copy"):
            return obj.copy()
        return obj

    data = {"z": z.copy()}

    # Collect concentrations and rates
    for species_name in species_list:
        data[f"c_{species_name}"] = snap(getattr(c, species_name))
        res_tuple = getattr(f_final, species_name)
        data[f"f_{species_name}"] = snap(res_tuple[2])

    # Capture process-specific rates dynamically
    for key in f_final.keys():
        if key not in species_list:
            res_tuple = getattr(f_final, key)
            data[f"f_{key}"] = snap(res_tuple[2])

    # Diffusion coefficients
    for d_name, d_val in D_mol.items():
        key = f"D_{d_name}" if d_name in species_list else d_name
        data[key] = snap(d_val)

    # Isotopes
    isotope_map = {
        "SO4": "SO4_32",
        "H2S": "H2S_32",
        "HS": "HS_32",
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
            data[f"d_{base}"] = diff_lib.get_delta(s_total, s32, mp_params.VCDT)

    data["w"] = np.ones(len(z)) * mp_params.w
    data["phi"] = np.ones(len(z)) * (
        mp_params.phi.value if hasattr(mp_params.phi, "value") else mp_params.phi
    )

    return data


def write_to_queue_async(
    plot_queue: Any,
    mp_params: Any,
    c: Any,
    k: Any,
    species_list: list[str],
    z: Any,
    D_mol: Any,
    diagenetic_reactions: Any,
    equilibrium_reactions: Any,
    current_dt: float,
    title: str,
) -> None:
    """
    Simultaneously snaps model state and sends it to the plot_queue.
    If plot_queue is a BoundedVideoQueue and full, this blocks until the queue
    drains to <= resume_threshold.
    """
    data = capture_state(
        mp_params, c, k, species_list, z, D_mol, diagenetic_reactions, equilibrium_reactions, current_dt
    )

    try:
        plot_queue.put((data, title))
    except Exception:
        pass


def save_final_pdf(
    mp: Any,
    c: Any,
    k: Any,
    species_list_full: list[str],
    z: np.ndarray,
    D_mol: Any,
    diagenetic_reactions: Any,
    equilibrium_reactions: Any,
    outfile: Optional[str] = None,
) -> Optional[str]:
    """
    Produce a static final PDF plot of the model state.
    """
    import fipyrite.plot_data_new as plot_data_new

    if outfile is None:
        outfile = f"{mp.plot_name}.pdf"

    layout_file = getattr(mp, "layout_file", "plot_layout.py")
    display_length = getattr(mp, "display_length", 2)
    measured_data_path = getattr(mp, "measured_data_path", None)

    try:
        final_data = capture_state(
            mp,
            c,
            k,
            species_list_full,
            z,
            D_mol,
            diagenetic_reactions,
            equilibrium_reactions,
            current_dt=0.0,
        )
        final_df = pd.DataFrame(final_data)
        plt_desc = plot_data_new.load_layout_from_file(final_df, layout_file, measured_data_path)
        plot_data_new.plot(
            final_df,
            display_length,
            outfile=outfile,
            show=False,
            plot_description=plt_desc,
            measured_data_path=measured_data_path,
        )
        print(f"[Parent] Saved final PDF plot: {outfile}", flush=True)
        return outfile
    except Exception as e:
        print(f"[Parent] Warning: Could not save final PDF plot ({outfile}): {e}", flush=True)
        return None

