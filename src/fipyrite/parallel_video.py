from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fipyrite.live_plot_lib import BoundedVideoQueue

if TYPE_CHECKING:
    import pathlib as pl


def get_default_video_workers() -> int:
    """
    Return the default number of worker processes for parallel video rendering.
    Restricted to 60% of all available CPU cores, minimum 1.
    """
    cores = os.cpu_count() or 1
    return max(1, int(cores * 0.6))


def detect_ffmpeg_encoder(target_ext: str = ".mp4") -> Tuple[str, List[str]]:
    """
    Detect the most suitable FFmpeg video encoder available on the system.
    Tests candidate encoders with a minimal 1-frame probe to ensure compatibility.
    Prioritizes libx264, libopenh264, mpeg4, and libvpx-vp9.
    """
    is_webm = target_ext.lower() == ".webm"

    if is_webm:
        candidates = [
            ("libvpx-vp9", ["-c:v", "libvpx-vp9", "-b:v", "4M", "-pix_fmt", "yuv420p"]),
            ("libvpx", ["-c:v", "libvpx", "-b:v", "4M", "-pix_fmt", "yuv420p"]),
        ]
    else:
        candidates = [
            ("libx264", ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]),
            ("libopenh264", ["-c:v", "libopenh264", "-pix_fmt", "yuv420p", "-b:v", "4M"]),
            ("mpeg4", ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"]),
            ("libvpx-vp9", ["-c:v", "libvpx-vp9", "-b:v", "4M", "-pix_fmt", "yuv420p"]),
        ]

    try:
        res = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        stdout = res.stdout
    except Exception:
        fallback = ("mpeg4", ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"])
        return fallback

    for name, args in candidates:
        matched = any(
            name in line and line.strip().startswith("V")
            for line in stdout.splitlines()
        )
        if not matched:
            continue

        # Probe with a 16x16 raw frame to verify encoder works
        probe_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", "16x16",
            "-pix_fmt", "rgba",
            "-r", "1",
            "-i", "-",
            *args,
            "-frames:v", "1",
            "-f", "null", "-",
        ]
        try:
            probe = subprocess.run(
                probe_cmd,
                input=b"\x00" * (16 * 16 * 4),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            if probe.returncode == 0:
                return name, args
        except Exception:
            continue

    # Fallback to mpeg4
    return "mpeg4", ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"]


class IndexedVideoQueue:
    """
    Queue wrapper that intercepts put() calls from the simulation solver,
    attaches an incrementing frame index (0, 1, 2, ...), caches the latest snapshot,
    and delegates to an underlying BoundedVideoQueue.
    """

    def __init__(self, bounded_queue: BoundedVideoQueue):
        self._bq = bounded_queue
        self._frame_idx = 0
        self._last_snapshot: Optional[Tuple[Dict[str, np.ndarray], str]] = None

    @property
    def last_snapshot(self) -> Optional[Tuple[Dict[str, np.ndarray], str]]:
        return self._last_snapshot

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        if item is None:
            return self._bq.put(None, block=block, timeout=timeout)

        data, title = item
        self._last_snapshot = (data, title)
        indexed_item = (self._frame_idx, data, title)
        self._frame_idx += 1
        return self._bq.put(indexed_item, block=block, timeout=timeout)

    def put_nowait(self, item: Any) -> bool:
        return self.put(item, block=False)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        return self._bq.get(block=block, timeout=timeout)

    def get_nowait(self) -> Any:
        return self._bq.get_nowait()

    def qsize(self) -> int:
        return self._bq.qsize()

    def empty(self) -> bool:
        return self._bq.empty()

    def full(self) -> bool:
        return self._bq.full()

    def stop(self) -> None:
        self._bq.stop()

    def cancel_join_thread(self) -> None:
        self._bq.cancel_join_thread()

    def close(self) -> None:
        self._bq.close()


def _video_render_worker(
    worker_id: int,
    layout_path: str,
    display_length: float,
    measured_data_path: Optional[str],
    dpi: int,
    task_queue: Any,
    result_queue: Any,
) -> None:
    """
    Worker process target function.
    Reads (frame_idx, data, title) from task_queue, renders plot into Matplotlib Agg,
    extracts raw RGBA bytes, and pushes (frame_idx, buf, width, height) to result_queue.
    Reuses figure handle across frames for maximum speed.
    """
    try:
        os.setpgrp()
    except OSError:
        pass

    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.dpi"] = dpi
    matplotlib.rcParams["savefig.dpi"] = dpi
    import matplotlib.pyplot as plt
    import fipyrite.plot_data_new as plot_data_new

    fig = None
    ax_objects = None
    plt_desc = None

    while True:
        try:
            item = task_queue.get()
        except Exception:
            break

        if item is None:
            # Termination sentinel
            break

        frame_idx, data, title = item
        try:
            df = pd.DataFrame(data)
            plt_desc = plot_data_new.load_layout_from_file(df, layout_path, measured_data_path)

            if fig is None:
                fig, ax_objects = plot_data_new.plot(
                    df,
                    display_length,
                    outfile=None,
                    show=False,
                    plot_description=plt_desc,
                    measured_data_path=measured_data_path,
                    keep_open=True,
                    title=title,
                )
                if dpi:
                    fig.set_dpi(dpi)
            else:
                plot_data_new.plot(
                    df,
                    display_length,
                    outfile=None,
                    show=False,
                    fig_handle=fig,
                    plot_description=plt_desc,
                    measured_data_path=measured_data_path,
                    keep_open=True,
                    title=title,
                )

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = bytes(fig.canvas.buffer_rgba())
            result_queue.put((frame_idx, buf, w, h))
        except Exception as e:
            print(f"[ParallelVideoWorker {worker_id}] Error rendering frame {frame_idx}: {e}", flush=True)
            result_queue.put((frame_idx, None, 0, 0))

    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass

    # Signal completion of this worker
    try:
        result_queue.put(None)
    except Exception:
        pass


def _video_sequencer(
    video_path: str,
    fps: int,
    num_workers: int,
    result_queue: Any,
    stop_event: Any,
    codec_name: str,
    codec_args: List[str],
) -> None:
    """
    Sequencer process target function.
    Collects rendered frames from workers, reassembles them in strict sequential order
    (0, 1, 2, ...), and streams raw RGBA buffers directly into FFmpeg's stdin pipe.
    """
    try:
        os.setpgrp()
    except OSError:
        pass

    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    next_frame_idx = 0
    pending_frames: Dict[int, Tuple[Optional[bytes], int, int]] = {}
    ffmpeg_proc: Optional[subprocess.Popen] = None
    workers_finished = 0
    last_valid_buf: Optional[bytes] = None
    total_encoded_frames = 0

    def _ensure_ffmpeg(w: int, h: int) -> None:
        nonlocal ffmpeg_proc
        if ffmpeg_proc is None:
            dirname = os.path.dirname(os.path.abspath(video_path))
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{w}x{h}",
                "-pix_fmt", "rgba",
                "-r", str(fps),
                "-i", "-",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                *codec_args,
                video_path,
            ]
            ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

    while True:
        try:
            res = result_queue.get(timeout=0.2)
        except queue.Empty:
            if stop_event.is_set() and result_queue.empty():
                break
            continue

        if res is None:
            workers_finished += 1
            if workers_finished >= num_workers:
                break
            continue

        frame_idx, buf, w, h = res
        pending_frames[frame_idx] = (buf, w, h)

        while next_frame_idx in pending_frames:
            f_buf, fw, fh = pending_frames.pop(next_frame_idx)
            if f_buf is not None:
                _ensure_ffmpeg(fw, fh)
                last_valid_buf = f_buf
                try:
                    ffmpeg_proc.stdin.write(f_buf)  # type: ignore[union-attr]
                    total_encoded_frames += 1
                except (BrokenPipeError, OSError) as e:
                    print(f"[ParallelVideoSequencer] BrokenPipeError writing frame {next_frame_idx}: {e}", flush=True)
                    break
            elif last_valid_buf is not None and ffmpeg_proc is not None:
                # Fallback: duplicate last valid frame on worker render error
                try:
                    ffmpeg_proc.stdin.write(last_valid_buf)
                    total_encoded_frames += 1
                except (BrokenPipeError, OSError):
                    break
            next_frame_idx += 1

    # Drain any remaining sequential pending frames
    while next_frame_idx in pending_frames:
        f_buf, fw, fh = pending_frames.pop(next_frame_idx)
        if f_buf is not None:
            _ensure_ffmpeg(fw, fh)
            try:
                ffmpeg_proc.stdin.write(f_buf)  # type: ignore[union-attr]
                total_encoded_frames += 1
            except (BrokenPipeError, OSError):
                break
        next_frame_idx += 1

    if ffmpeg_proc is not None:
        try:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait(timeout=30)
            if ffmpeg_proc.returncode != 0:
                err_msg = ffmpeg_proc.stderr.read().decode("utf-8", errors="replace") if ffmpeg_proc.stderr else ""
                print(f"[ParallelVideoSequencer] FFmpeg returned error {ffmpeg_proc.returncode}: {err_msg}", flush=True)
            else:
                print(
                    f"[ParallelVideoSequencer] Successfully encoded {total_encoded_frames} frames to {video_path} using {codec_name}",
                    flush=True,
                )
        except Exception as e:
            ffmpeg_proc.kill()
            print(f"[ParallelVideoSequencer] Exception finishing FFmpeg: {e}", flush=True)


class ParallelVideoManager:
    """
    Manages multi-core concurrent video rendering for FiPyrite.
    Spawns worker processes (defaulting to 60% of all CPU cores) and a sequencer process.
    Provides backpressure through an IndexedVideoQueue (default max 100 entries).
    """

    def __init__(
        self,
        layout_path: str,
        display_length: float,
        video_path: str,
        measured_data_path: Optional[str] = None,
        fps: int = 15,
        video_dpi: int = 120,
        video_workers: Optional[int] = None,
        max_queue_size: int = 100,
        resume_threshold: int = 50,
    ):
        self.layout_path = layout_path
        self.display_length = display_length
        self.video_path = video_path
        self.measured_data_path = measured_data_path
        self.fps = fps
        self.video_dpi = video_dpi

        total_cores = os.cpu_count() or 1
        if video_workers is None or video_workers <= 0:
            self.num_workers = get_default_video_workers()
        else:
            self.num_workers = max(1, min(video_workers, total_cores))

        self.max_queue_size = max_queue_size
        self.resume_threshold = resume_threshold

        self._ctx = mp.get_context("spawn")
        self._raw_task_queue = BoundedVideoQueue(
            maxsize=self.max_queue_size,
            resume_threshold=self.resume_threshold,
            ctx=self._ctx,
        )
        self._queue_wrapper = IndexedVideoQueue(self._raw_task_queue)
        self._result_queue = self._ctx.Queue()
        self._stop_event = self._ctx.Event()

        self._workers: List[mp.Process] = []
        self._sequencer: Optional[mp.Process] = None
        self._started = False

    @property
    def queue(self) -> IndexedVideoQueue:
        return self._queue_wrapper

    def start(self) -> None:
        """Spawn sequencer and worker processes."""
        if self._started:
            return
        self._started = True

        ext = os.path.splitext(self.video_path)[1]
        codec_name, codec_args = detect_ffmpeg_encoder(target_ext=ext)
        print(
            f"[ParallelVideoManager] Starting {self.num_workers} workers (60% cores) "
            f"with encoder '{codec_name}', dpi={self.video_dpi}, queue_size={self.max_queue_size}...",
            flush=True,
        )

        self._sequencer = self._ctx.Process(
            target=_video_sequencer,
            args=(
                self.video_path,
                self.fps,
                self.num_workers,
                self._result_queue,
                self._stop_event,
                codec_name,
                codec_args,
            ),
            daemon=False,
        )
        self._sequencer.start()

        for wid in range(self.num_workers):
            p = self._ctx.Process(
                target=_video_render_worker,
                args=(
                    wid,
                    self.layout_path,
                    self.display_length,
                    self.measured_data_path,
                    self.video_dpi,
                    self._raw_task_queue,
                    self._result_queue,
                ),
                daemon=False,
            )
            p.start()
            self._workers.append(p)

    def stop(self) -> None:
        """Gracefully terminate workers and sequencer, flushing all queued frames."""
        if not self._started:
            return

        print(f"[ParallelVideoManager] Stopping: sending {self.num_workers} sentinels...", flush=True)

        # Send sentinels to all workers
        for _ in range(self.num_workers):
            try:
                self._raw_task_queue.put(None, timeout=2.0)
            except Exception:
                pass

        # Wait for workers to finish
        for p in self._workers:
            p.join(timeout=30)
            if p.is_alive():
                print(f"[ParallelVideoManager] Worker {p.pid} did not terminate; killing.", flush=True)
                p.terminate()
                p.join(timeout=2)

        # Wait for sequencer to finish
        if self._sequencer and self._sequencer.is_alive():
            self._sequencer.join(timeout=45)
            if self._sequencer.is_alive():
                print(f"[ParallelVideoManager] Sequencer did not terminate; killing.", flush=True)
                self._sequencer.terminate()
                self._sequencer.join(timeout=2)

        self._stop_event.set()
        try:
            self._raw_task_queue.cancel_join_thread()
            self._raw_task_queue.close()
        except Exception:
            pass

        try:
            self._result_queue.cancel_join_thread()
            self._result_queue.close()
        except Exception:
            pass

        self._started = False
        print("[ParallelVideoManager] Stopped successfully.", flush=True)
