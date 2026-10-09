import os
import tempfile
import time
import numpy as np
import pytest

from fipyrite.parallel_video import (
    detect_ffmpeg_encoder,
    get_default_video_workers,
    IndexedVideoQueue,
    ParallelVideoManager,
)
from fipyrite.live_plot_lib import BoundedVideoQueue, LivePlotter


def test_detect_ffmpeg_encoder():
    name, args = detect_ffmpeg_encoder(".mp4")
    assert isinstance(name, str)
    assert len(name) > 0
    assert isinstance(args, list)
    assert "-c:v" in args


def test_default_video_workers():
    workers = get_default_video_workers()
    cores = os.cpu_count() or 1
    assert workers >= 1
    assert workers <= cores
    expected = max(1, int(cores * 0.6))
    assert workers == expected


def test_indexed_video_queue():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    bq = BoundedVideoQueue(maxsize=10, resume_threshold=5, ctx=ctx)
    iq = IndexedVideoQueue(bq)

    # Put a couple items
    iq.put(({"z": np.array([0, 1])}, "Frame 0"))
    iq.put(({"z": np.array([0, 1])}, "Frame 1"))

    assert iq.last_snapshot is not None
    assert iq.last_snapshot[1] == "Frame 1"

    # Get from underlying queue
    item0 = bq.get(timeout=1.0)
    assert item0[0] == 0  # frame_idx == 0
    assert item0[2] == "Frame 0"

    item1 = bq.get(timeout=1.0)
    assert item1[0] == 1  # frame_idx == 1
    assert item1[2] == "Frame 1"

    bq.close()


def test_parallel_video_manager_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal layout file
        layout_code = """
def get_layout(df):
    return {
        "fig_width": 6,
        "Subplot1": {
            "height": 4,
            "xaxis": [df.z, "Depth [m]"],
            "left": [[df.c_SO4, "SO4", {"color": "blue"}]],
        }
    }
"""
        layout_file = os.path.join(tmpdir, "test_layout.py")
        with open(layout_file, "w") as f:
            f.write(layout_code)

        video_file = os.path.join(tmpdir, "test_output.mp4")

        # Initialize ParallelVideoManager with 2 workers for fast testing
        mgr = ParallelVideoManager(
            layout_path=layout_file,
            display_length=1.0,
            video_path=video_file,
            fps=15,
            video_dpi=80,
            video_workers=2,
            max_queue_size=20,
            resume_threshold=10,
        )

        mgr.start()

        # Send 10 synthetic frames
        z = np.linspace(0, 1, 20)
        for i in range(10):
            data = {
                "z": z,
                "c_SO4": np.sin(z + i * 0.2),
            }
            mgr.queue.put((data, f"Time: {i} s"))

        # Stop manager
        mgr.stop()

        # Verify video file exists and is non-empty
        assert os.path.exists(video_file)
        assert os.path.getsize(video_file) > 1000


def test_live_plotter_parallel_video_and_pdf():
    with tempfile.TemporaryDirectory() as tmpdir:
        layout_code = """
def get_layout(df):
    return {
        "fig_width": 6,
        "Subplot1": {
            "height": 4,
            "xaxis": [df.z, "Depth [m]"],
            "left": [[df.c_SO4, "SO4", {"color": "blue"}]],
        }
    }
"""
        layout_file = os.path.join(tmpdir, "test_layout.py")
        with open(layout_file, "w") as f:
            f.write(layout_code)

        video_file = os.path.join(tmpdir, "plotter_output.mp4")
        pdf_file = os.path.join(tmpdir, "plotter_output.pdf")

        plotter = LivePlotter(
            layout_path=layout_file,
            display_length=1.0,
            output_path=pdf_file,
            video_path=video_file,
            fps=15,
            video_dpi=80,
            video_workers=2,
            max_queue_size=20,
            resume_threshold=10,
        )

        plotter.start()

        z = np.linspace(0, 1, 20)
        for i in range(5):
            data = {
                "z": z,
                "c_SO4": np.sin(z + i * 0.2),
            }
            plotter.queue.put((data, f"Step {i}"))

        plotter.stop()

        assert os.path.exists(video_file)
        assert os.path.getsize(video_file) > 1000
        assert os.path.exists(pdf_file)
        assert os.path.getsize(pdf_file) > 1000
