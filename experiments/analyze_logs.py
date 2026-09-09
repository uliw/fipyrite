#!/usr/bin/env python3
"""
Log analysis script for FiPyrite simulation runs.

Parses compressed (.log.gz) or plain (.log) log files, extracts step-by-step
metrics, wall-clock time, simulation time, time-step size, sweeps, and failure/acceptance
events, and prints structured comparison tables and phase-by-phase throughput diagnostics.

Usage:
    python experiments/analyze_logs.py
    python experiments/analyze_logs.py experiments/run_velde_slow.log.gz experiments/run_velde_slow_picard_symmetrical.log.gz
"""

import sys
import os
import gzip
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def open_log(path: str | Path):
    """Open plain text or gzip log transparently."""
    p = Path(path)
    if not p.exists():
        # Check if .gz exists
        if p.with_suffix(p.suffix + ".gz").exists():
            p = p.with_suffix(p.suffix + ".gz")
        elif p.suffix == ".gz" and p.with_suffix("").exists():
            p = p.with_suffix("")
        else:
            raise FileNotFoundError(f"Log file not found: {path}")

    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace"), p
    return open(p, "r", encoding="utf-8", errors="replace"), p


def parse_wall_time(s: str) -> float:
    """Parse string like '01h 45m', '05m 48s', ' 2.1s' into seconds."""
    s = s.strip()
    total = 0.0
    for part in s.split():
        if part.endswith("h"):
            total += float(part[:-1]) * 3600.0
        elif part.endswith("m"):
            total += float(part[:-1]) * 60.0
        elif part.endswith("s"):
            total += float(part[:-1])
    return total


def format_wall_time(seconds: float) -> str:
    """Format seconds into readable human string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m"
    if m > 0:
        return f"{m:02d}m {s:04.1f}s"
    return f"{s:5.1f}s"


def to_years(val: float, unit: str) -> float:
    """Convert any simulation time unit to years."""
    u = unit.lower().strip()
    if u in ["ka", "kyr"]:
        return val * 1000.0
    if u in ["a", "yr", "year", "years"]:
        return val
    if u in ["month", "months"]:
        return val / 12.0
    if u in ["week", "weeks"]:
        return val / 52.1429
    if u in ["d", "day", "days"]:
        return val / 365.25
    if u in ["h", "hr", "hour", "hours"]:
        return val / (365.25 * 24.0)
    if u in ["min", "minute", "minutes"]:
        return val / (365.25 * 1440.0)
    if u in ["s", "sec", "second", "seconds"]:
        return val / (365.25 * 86400.0)
    return val


def to_days(val: float, unit: str) -> float:
    """Convert any time-step unit to days."""
    u = unit.lower().strip()
    if u in ["ka", "kyr"]:
        return val * 365250.0
    if u in ["a", "yr", "year", "years"]:
        return val * 365.25
    if u in ["month", "months"]:
        return val * (365.25 / 12.0)
    if u in ["week", "weeks"]:
        return val * 7.0
    if u in ["d", "day", "days"]:
        return val
    if u in ["h", "hr", "hour", "hours"]:
        return val / 24.0
    if u in ["min", "minute", "minutes"]:
        return val / 1440.0
    if u in ["s", "sec", "second", "seconds"]:
        return val / 86400.0
    return val


class LogSummary:
    def __init__(self, file_path: str):
        self.raw_path = file_path
        self.file_path = ""
        self.name = Path(file_path).name
        self.steps: List[Dict[str, Any]] = []
        self.hard_failures: List[Tuple[float, float, str]] = []  # (wall_s, sim_yr, msg)
        self.rollbacks: List[Tuple[float, float, str]] = []
        self.near_converged: List[Tuple[float, float, str]] = []
        self.graceful_accepted: List[Tuple[float, float, str]] = []
        self.final_report: Optional[str] = None
        self.parse()

    def parse(self):
        f, actual_path = open_log(self.raw_path)
        self.file_path = str(actual_path)
        self.name = actual_path.name

        step_re = re.compile(
            r"\[([^\]]+)\]\s+Step\s+(\d+),\s+Time:\s+([0-9\.]+)\s+([a-zA-Z]+),\s+dt:\s+([0-9\.]+)\s+([a-zA-Z]+)\s+\|\s+([0-9\.]+)\s+s/step,\s+([0-9\.]+)\s+([a-zA-Z/]+)/min(?:,\s+sweeps:\s+(\d+))?"
        )
        near_re = re.compile(r"\[([^\]]+)\]\s+Near-convergence accepted at sweep (\d+)")
        graceful_re = re.compile(r"\[([^\]]+)\]\s+Graceful acceptance at sweep (\d+)")
        fail_re = re.compile(r"\[([^\]]+)\]\s+Step failed at dt=([^:]+):(.*)")
        rollback_re = re.compile(r"\[([^\]]+)\]\s+Step rejected at dt=([^:]+):(.*)")

        current_sim_yr = 0.0

        with f:
            for line in f:
                line_str = line.strip()

                # Check step line
                m = step_re.search(line_str)
                if m:
                    wall_s = parse_wall_time(m.group(1))
                    step_num = int(m.group(2))
                    sim_val = float(m.group(3))
                    sim_unit = m.group(4)
                    sim_yr = to_years(sim_val, sim_unit)
                    current_sim_yr = sim_yr

                    dt_val = float(m.group(5))
                    dt_unit = m.group(6)
                    dt_d = to_days(dt_val, dt_unit)

                    s_per_step = float(m.group(7))
                    speed_val = float(m.group(8))
                    speed_unit = m.group(9)

                    sweeps_val = int(m.group(10)) if m.group(10) else 1

                    self.steps.append(
                        {
                            "step": step_num,
                            "wall_s": wall_s,
                            "sim_yr": sim_yr,
                            "dt_d": dt_d,
                            "s_per_step": s_per_step,
                            "speed_val": speed_val,
                            "speed_unit": speed_unit,
                            "sweeps": sweeps_val,
                        }
                    )
                    continue

                if "Near-convergence accepted" in line_str:
                    m_n = near_re.search(line_str)
                    wall_s = parse_wall_time(m_n.group(1)) if m_n else 0.0
                    self.near_converged.append((wall_s, current_sim_yr, line_str))
                elif "Graceful acceptance" in line_str:
                    m_g = graceful_re.search(line_str)
                    wall_s = parse_wall_time(m_g.group(1)) if m_g else 0.0
                    self.graceful_accepted.append((wall_s, current_sim_yr, line_str))
                elif "Step failed" in line_str:
                    m_f = fail_re.search(line_str)
                    wall_s = parse_wall_time(m_f.group(1)) if m_f else 0.0
                    self.hard_failures.append((wall_s, current_sim_yr, line_str))
                elif "Step rejected" in line_str or "Rollback" in line_str:
                    m_r = rollback_re.search(line_str)
                    wall_s = parse_wall_time(m_r.group(1)) if m_r else 0.0
                    self.rollbacks.append((wall_s, current_sim_yr, line_str))
                elif "Final Report:" in line_str:
                    self.final_report = line_str

    @property
    def total_wall_s(self) -> float:
        return self.steps[-1]["wall_s"] if self.steps else 0.0

    @property
    def total_sim_yr(self) -> float:
        return self.steps[-1]["sim_yr"] if self.steps else 0.0

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def total_sweeps(self) -> int:
        return sum(s["sweeps"] for s in self.steps)

    def stats_in_range(self, t_min_yr: float, t_max_yr: float) -> Dict[str, Any]:
        """Compute metrics for steps within a given simulated time range."""
        subset = [s for s in self.steps if t_min_yr <= s["sim_yr"] <= t_max_yr]
        if not subset:
            return {
                "count": 0,
                "wall_s": 0.0,
                "sim_yr": 0.0,
                "avg_dt_d": 0.0,
                "avg_s_per_step": 0.0,
                "avg_sweeps": 0.0,
                "throughput_d_s": 0.0,
                "throughput_yr_min": 0.0,
                "hard_fails": 0,
                "rollbacks": 0,
                "near_conv": 0,
                "graceful": 0,
            }

        first_w = subset[0]["wall_s"]
        last_w = subset[-1]["wall_s"]
        wall_span = max(last_w - first_w, 1e-4)

        sim_span_yr = subset[-1]["sim_yr"] - subset[0]["sim_yr"]
        sim_span_d = sim_span_yr * 365.25

        avg_dt = sum(s["dt_d"] for s in subset) / len(subset)
        avg_pace = sum(s["s_per_step"] for s in subset) / len(subset)
        avg_swp = sum(s["sweeps"] for s in subset) / len(subset)

        hf = sum(1 for f in self.hard_failures if t_min_yr <= f[1] <= t_max_yr)
        rb = sum(1 for r in self.rollbacks if t_min_yr <= r[1] <= t_max_yr)
        nc = sum(1 for n in self.near_converged if t_min_yr <= n[1] <= t_max_yr)
        ga = sum(1 for g in self.graceful_accepted if t_min_yr <= g[1] <= t_max_yr)

        return {
            "count": len(subset),
            "wall_s": wall_span,
            "sim_yr": sim_span_yr,
            "avg_dt_d": avg_dt,
            "avg_s_per_step": avg_pace,
            "avg_sweeps": avg_swp,
            "throughput_d_s": sim_span_d / wall_span,
            "throughput_yr_min": (sim_span_yr / wall_span) * 60.0,
            "hard_fails": hf,
            "rollbacks": rb,
            "near_conv": nc,
            "graceful": ga,
        }

    def milestone(self, target_yr: float) -> Optional[Dict[str, Any]]:
        """Find the step where sim_yr first reached target_yr."""
        for s in self.steps:
            if s["sim_yr"] >= target_yr:
                return s
        return None


def print_comparison(logs: List[LogSummary]):
    print("=" * 95)
    print(f"{'EXPERIMENT LOG ANALYSIS & COMPARISON':^95}")
    print("=" * 95)

    # 1. Overall Summary Table
    print(f"\n1. Overall Run Summary:")
    header = f"{'Run / Log File':<35} | {'Steps':>7} | {'Sweeps':>8} | {'Wall Time':>11} | {'Sim Time':>12} | {'Swp/s':>6}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for log in logs:
        swp_rate = log.total_sweeps / max(log.total_wall_s, 1e-4)
        print(
            f"{log.name:<35} | {log.total_steps:>7d} | {log.total_sweeps:>8d} | "
            f"{format_wall_time(log.total_wall_s):>11} | {log.total_sim_yr:>10.2f} yr | {swp_rate:>6.2f}"
        )
    print("-" * len(header))

    # 2. Events breakdown Table
    print(f"\n2. Solver Events & Failure Breakdown:")
    header2 = f"{'Run / Log File':<35} | {'Hard Fails':>11} | {'Rollbacks':>10} | {'Near-Conv':>10} | {'Graceful':>9}"
    print("-" * len(header2))
    print(header2)
    print("-" * len(header2))
    for log in logs:
        print(
            f"{log.name:<35} | {len(log.hard_failures):>11d} | {len(log.rollbacks):>10d} | "
            f"{len(log.near_converged):>10d} | {len(log.graceful_accepted):>9d}"
        )
    print("-" * len(header2))

    # 3. Milestone Reach Time
    print(f"\n3. Wall-Clock Time to Reach Simulation Milestones:")
    milestones = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0]
    col_names = [log.name[:25] for log in logs]
    m_header = f"{'Target (yr)':>12} | " + " | ".join(f"{cn:>26}" for cn in col_names)
    print("-" * len(m_header))
    print(m_header)
    print("-" * len(m_header))
    for m in milestones:
        cells = []
        for log in logs:
            st = log.milestone(m)
            if st:
                cells.append(f"{format_wall_time(st['wall_s'])} (st {st['step']}, dt {st['dt_d']:.1f}d)")
            else:
                cells.append("N/A")
        print(f"{m:>12.1f} | " + " | ".join(f"{c:>26}" for c in cells))
    print("-" * len(m_header))

    # 4. Phase-by-Phase Performance
    phases = [
        ("Phase 1: FeS Onset & Migration (0.0 to 10.0 yr)", 0.0, 10.0),
        ("Phase 2: Established FeS Diagenesis (10.0 to 50.0 yr)", 10.0, 50.0),
    ]
    for p_name, t0, t1 in phases:
        print(f"\n4. Performance in {p_name}:")
        p_header = f"{'Run':<32} | {'Steps':>6} | {'Avg dt':>8} | {'Pace':>8} | {'Sweeps':>6} | {'Throughput':>16} | {'Fails/Roll':>10}"
        print("-" * len(p_header))
        print(p_header)
        print("-" * len(p_header))
        for log in logs:
            st = log.stats_in_range(t0, t1)
            th_str = f"{st['throughput_d_s']:.1f} d/s ({st['throughput_yr_min']:.2f} yr/m)"
            fr_str = f"{st['hard_fails']}/{st['rollbacks']}"
            print(
                f"{log.name:<32} | {st['count']:>6d} | {st['avg_dt_d']:>6.1f} d | {st['avg_s_per_step']:>6.2f}s | "
                f"{st['avg_sweeps']:>6.2f} | {th_str:>16} | {fr_str:>10}"
            )
        print("-" * len(p_header))

    # 5. Failure and Rollback Diagnostics
    print(f"\n5. Failure & Rollback Diagnostics:")
    for log in logs:
        if not log.hard_failures and not log.rollbacks:
            continue
        print(f"\n  [{log.name}]")
        if log.hard_failures:
            print(f"    Hard Failures ({len(log.hard_failures)} total):")
            # Group or sample
            sample = log.hard_failures[:5]
            for wall_s, sim_yr, msg in sample:
                clean_msg = msg.split("Step failed at")[-1].strip() if "Step failed at" in msg else msg
                print(f"      - [{format_wall_time(wall_s)}, t={sim_yr:6.2f} yr] {clean_msg}")
            if len(log.hard_failures) > 5:
                print(f"      ... and {len(log.hard_failures) - 5} more.")
        if log.rollbacks:
            print(f"    Rollbacks / Rate Violations ({len(log.rollbacks)} total):")
            sample = log.rollbacks[:5]
            for wall_s, sim_yr, msg in sample:
                clean_msg = msg.split("Step rejected at")[-1].strip() if "Step rejected at" in msg else msg
                print(f"      - [{format_wall_time(wall_s)}, t={sim_yr:6.2f} yr] {clean_msg}")
            if len(log.rollbacks) > 5:
                print(f"      ... and {len(log.rollbacks) - 5} more.")


def main():
    args = sys.argv[1:]
    if not args:
        # Default files to check
        candidates = [
            "experiments/run_velde_slow.log.gz",
            "experiments/run_velde_slow_picard_symmetrical.log.gz",
        ]
        args = [c for c in candidates if Path(c).exists() or Path(c.replace(".gz", "")).exists()]
        if not args:
            print("No log files provided and default log files not found.")
            sys.exit(1)

    logs = []
    for arg in args:
        try:
            logs.append(LogSummary(arg))
        except Exception as e:
            print(f"Error loading {arg}: {e}", file=sys.stderr)

    if not logs:
        sys.exit(1)

    print_comparison(logs)


if __name__ == "__main__":
    main()
