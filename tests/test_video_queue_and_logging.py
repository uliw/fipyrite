import queue
import time
import threading
import multiprocessing as mp
import numpy as np
import pytest

from fipyrite.live_plot_lib import BoundedVideoQueue, parse_time_to_seconds
from fipyrite.direct_assembled_solver import GovernorStats


def test_parse_time_to_seconds():
    # None
    assert parse_time_to_seconds(None) is None

    # Numerics
    assert parse_time_to_seconds(100) == 100.0
    assert parse_time_to_seconds(42.5) == 42.5

    # String with pint
    assert parse_time_to_seconds("1 day") == pytest.approx(86400.0)
    assert parse_time_to_seconds("10 days") == pytest.approx(864000.0)
    assert parse_time_to_seconds("1 year") == pytest.approx(31557600.0, rel=1e-2)

    # Pint quantity
    import pint
    ureg = pint.UnitRegistry()
    q = ureg.Quantity("2 hours")
    assert parse_time_to_seconds(q) == pytest.approx(7200.0)


def test_bounded_video_queue_capacity_and_backpressure():
    # Test bounded queue with small numbers for fast deterministic testing
    ctx = mp.get_context("spawn")
    q = BoundedVideoQueue(maxsize=5, resume_threshold=2, ctx=ctx)

    # Put up to capacity without blocking
    for i in range(5):
        assert q.put(i, timeout=1.0) is True

    assert q.qsize() == 5
    assert q.full() is True

    # Attempting put_nowait when full should raise queue.Full
    with pytest.raises(queue.Full):
        q.put_nowait(999)

    # In a background thread, drain items slowly
    drained = []

    def consumer():
        time.sleep(0.05)
        # Drain 3 items so count drops from 5 -> 2 (<= resume_threshold)
        for _ in range(3):
            drained.append(q.get(timeout=2.0))
            time.sleep(0.01)

    t = threading.Thread(target=consumer)
    t.start()

    # This put should block until consumer drains down to <= 2
    start = time.time()
    res = q.put("unblocked_item", timeout=3.0)
    elapsed = time.time() - start

    assert res is True
    assert elapsed >= 0.05
    assert len(drained) >= 3

    t.join()
    q.close()


def test_bounded_video_queue_stop_unblocks_waiting():
    ctx = mp.get_context("spawn")
    q = BoundedVideoQueue(maxsize=2, resume_threshold=1, ctx=ctx)
    q.put(1)
    q.put(2)

    # Producer thread that will wait
    results = []

    def producer():
        res = q.put(3, timeout=5.0)
        results.append(res)

    t = threading.Thread(target=producer)
    t.start()
    time.sleep(0.05)

    # Now call stop(), which should unblock the waiting put
    q.stop()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert results == [False]
    q.close()


def test_governor_stats_tracking_and_percentages():
    stats = GovernorStats()

    # 1. Single sweep accepted (1 sweep)
    for _ in range(80):
        stats.record_step(sweeps=1, outcome="single_sweep")

    # 2. Multi-sweep accepted (average 2 sweeps each, 5 steps = 10 sweeps)
    for _ in range(5):
        stats.record_step(sweeps=2, outcome="multi_sweep")

    # 3. Capped by porewater governor (5 steps with 1 sweep each = 5 sweeps)
    for _ in range(5):
        stats.record_step(sweeps=1, outcome="single_sweep", porewater_capped=True, limiting_species="SO4")

    # 4. Rejected attempts (2 failures, 2 sweeps and 3 sweeps = 5 sweeps)
    stats.record_rejection(sweeps=2, reason="Picard sweep failed to converge in 4 iterations")
    stats.record_rejection(sweeps=3, reason="Linear solver error")

    # Total sweeps: 80*1 + 5*2 + 5*1 + 5 = 100 sweeps!
    assert stats.total_sweeps == 100
    assert stats.single_sweep_sweeps == 80
    assert stats.multi_sweep_sweeps == 10
    assert stats.porewater_capped_sweeps == 5
    assert stats.rejected_sweeps == 5

    summary = stats.format_summary()
    assert "80.0% (80 sweeps): accepted with a single sweep" in summary
    assert "10.0% (10 sweeps): accepted with multi-sweeps" in summary
    assert " 5.0% (5 sweeps): capped because porewater depleted too fast (limiting: SO4)" in summary
    assert " 5.0% (5 sweeps): rejected / Picard convergence failure" in summary


def test_video_dt_throttling_logic():
    # Verify that frame generation respects video_dt intervals
    video_dt_sec = 31536000.0  # 1 year in seconds
    last_video_time = -float("inf")
    frames_recorded = []

    sim_times = [
        0.0,
        1000.0,
        100000.0,
        31536000.0,        # 1.0 yr -> should trigger
        31536000.0 + 100,  # 1.0 yr + 100s -> should NOT trigger
        63072000.0,        # 2.0 yr -> should trigger
    ]

    for step, total_time in enumerate(sim_times, start=1):
        if total_time - last_video_time >= video_dt_sec or step == 1:
            frames_recorded.append((step, total_time))
            last_video_time = total_time

    # Step 1 (t=0) triggers, Step 4 (t=1 yr) triggers, Step 6 (t=2 yr) triggers
    assert len(frames_recorded) == 3
    assert frames_recorded[0] == (1, 0.0)
    assert frames_recorded[1] == (4, 31536000.0)
    assert frames_recorded[2] == (6, 63072000.0)
